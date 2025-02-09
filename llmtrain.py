import os
import sys
import gradio as gr
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
try:
    from peft import prepare_model_for_int8_training
except ImportError:
    prepare_model_for_int8_training = None
from huggingface_hub import login
import pandas as pd

# GPU ayarları için
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def create_prompt(system_text, user_text, assistant_text):
    prompt = f"[SYSTEM]\n{system_text}\n[USER]\n{user_text}\n[ASSISTANT]\n"
    return prompt, assistant_text

def chunk_list(seq, chunk_size):
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

def train_model(
    model_type,
    hf_token,
    base_model_name,
    dataset_choice,
    dataset_path,
    local_format,  # "json" veya "csv"
    dataset_repo,
    dataset_privacy,
    max_length,
    gradient_accumulation_steps,
    per_device_train_batch_size,
    num_epochs,
    quantization_choice,
    bit_choice,
    precision_choice,
    system_col,
    user_col,
    assistant_col,
    progress=gr.Progress()
):
    progress(0, desc="Initializing...")

    if model_type == "private" and hf_token:
        login(token=hf_token)

    # Dataset yükleme
    progress(0.1, desc="Loading dataset...")
    if dataset_choice == "local":
        raw_datasets = load_dataset(local_format, data_files={"train": dataset_path})
        if "train" in raw_datasets and len(raw_datasets) == 1:
            split_dataset = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
            raw_datasets = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
    else:
        if dataset_privacy == "private" and hf_token:
            raw_datasets = load_dataset(dataset_repo, use_auth_token=hf_token)
        else:
            raw_datasets = load_dataset(dataset_repo)
        if "train" in raw_datasets and len(raw_datasets) == 1:
            split_dataset = raw_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
            raw_datasets = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})

    # Cihazı belirleyelim: GPU varsa "cuda", yoksa "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress(0.2, desc="Loading model...")
    # Model yükleme ve quantization/precision seçenekleri
    if quantization_choice:
        if bit_choice == "4":
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                use_auth_token=hf_token if model_type == "private" else None,
                load_in_4bit=True,
                trust_remote_code=True,
                low_cpu_mem_usage=False  # Tam yükleme, device_map kullanılmıyor.
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                use_auth_token=hf_token if model_type == "private" else None,
                load_in_8bit=True,
                trust_remote_code=True,
                low_cpu_mem_usage=False
            )
            if prepare_model_for_int8_training:
                model = prepare_model_for_int8_training(model)
    else:
        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }[precision_choice]

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            use_auth_token=hf_token if model_type == "private" else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False
        )
        model.config.use_cache = False

    # Modeli mevcut cihaza taşıyalım (GPU varsa "cuda")
    model = model.to(device)

    # LoRA ayarları
    lora_config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    progress(0.3, desc="Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_auth_token=hf_token if model_type == "private" else None,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        # Esnek sütun erişimi: Girilen sütun adını bulamazsa case-insensitive arama yapar.
        def get_column(examples, col_name):
            if col_name in examples:
                return examples[col_name]
            for key in examples.keys():
                if key.lower() == col_name.lower():
                    return examples[key]
            return [""] * len(examples[next(iter(examples))])
        
        system_texts = get_column(examples, system_col)
        user_texts = get_column(examples, user_col)
        assistant_texts = get_column(examples, assistant_col)

        final_input_ids = []
        final_attention_masks = []
        final_labels = []
        
        for sys_text, usr_text, asst_text in zip(system_texts, user_texts, assistant_texts):
            sys_text = sys_text or ""
            usr_text = usr_text or ""
            asst_text = asst_text or ""
            prompt, answer = create_prompt(sys_text.strip(), usr_text.strip(), asst_text.strip())
            full_text = prompt + answer
            
            tokenized = tokenizer(full_text, truncation=False, padding=False)
            input_ids = tokenized["input_ids"]
            prompt_tokenized = tokenizer(prompt, truncation=False, padding=False)
            prompt_length = len(prompt_tokenized["input_ids"])
            
            labels = [-100] * len(input_ids)
            for i in range(prompt_length, len(input_ids)):
                labels[i] = input_ids[i]
                
            if len(input_ids) > max_length:
                input_chunks = chunk_list(input_ids, max_length)
                label_chunks = chunk_list(labels, max_length)
                for inp_chunk, lbl_chunk in zip(input_chunks, label_chunks):
                    pad_len = max_length - len(inp_chunk)
                    if pad_len > 0:
                        inp_chunk += [tokenizer.pad_token_id] * pad_len
                        lbl_chunk += [-100] * pad_len
                    attn_mask = [1 if t != tokenizer.pad_token_id else 0 for t in inp_chunk]
                    final_input_ids.append(inp_chunk)
                    final_attention_masks.append(attn_mask)
                    final_labels.append(lbl_chunk)
            else:
                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    input_ids += [tokenizer.pad_token_id] * pad_len
                    labels += [-100] * pad_len
                attn_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]
                final_input_ids.append(input_ids)
                final_attention_masks.append(attn_mask)
                final_labels.append(labels)
                
        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_masks,
            "labels": final_labels
        }

    progress(0.4, desc="Preprocessing datasets...")
    tokenized_train = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    tokenized_dev = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    progress(0.5, desc="Setting up training...")
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=200,
        save_steps=200,
        logging_steps=100,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=(not quantization_choice and precision_choice == "fp16"),
        bf16=(not quantization_choice and precision_choice == "bf16"),
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        push_to_hub=False,
        optim="adamw_torch_fused"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=DefaultDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    progress(0.6, desc="Training model...")
    trainer.train()
    
    progress(0.9, desc="Saving model...")
    trainer.save_model("./finetuned_model")
    tokenizer.save_pretrained("./finetuned_model")
    
    progress(1.0, desc="Training completed!")
    return "Training completed successfully! Model saved in './finetuned_model' directory."

def explore_dataset(
    dataset_choice,
    dataset_path,
    local_format,
    dataset_repo,
    dataset_privacy,
    hf_token
):
    try:
        if dataset_choice == "local":
            ds = load_dataset(local_format, data_files={"train": dataset_path})
            if "train" in ds:
                dataset = ds["train"]
            else:
                dataset = ds[list(ds.keys())[0]]
        else:
            if dataset_privacy == "private" and hf_token:
                ds = load_dataset(dataset_repo, use_auth_token=hf_token)
            else:
                ds = load_dataset(dataset_repo)
            if "train" in ds:
                dataset = ds["train"]
            else:
                dataset = ds[list(ds.keys())[0]]
    except Exception as e:
        return f"Dataset yüklenirken hata oluştu: {e}"

    info = f"Dataset Columns: {dataset.column_names}\n"
    info += f"Total Examples: {len(dataset)}\n\n"
    # İlk 5 örneği pandas DataFrame olarak alıp stringe çeviriyoruz.
    sample = dataset.select(range(min(5, len(dataset))))
    try:
        sample_df = sample.to_pandas()
    except Exception:
        sample_df = pd.DataFrame(sample)
    info += "Sample Data:\n" + sample_df.head().to_string()
    return info

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# LLM Fine-tuning and Dataset Exploration Interface")

        with gr.Tabs():
            with gr.TabItem("Dataset Exploration"):
                gr.Markdown("### Dataset Yükleme ve Keşif")
                with gr.Row():
                    with gr.Column():
                        dataset_choice_exp = gr.Radio(
                            choices=["local", "huggingface"],
                            label="Dataset Source",
                            value="huggingface"
                        )
                        dataset_path_exp = gr.Textbox(
                            label="Local Dataset Path",
                            placeholder="Yerel dosya yolu (örn: data.json veya data.csv)",
                            visible=False
                        )
                        local_format_exp = gr.Radio(
                            choices=["json", "csv"],
                            label="Local Dataset Format",
                            value="json",
                            visible=False
                        )
                        dataset_repo_exp = gr.Textbox(
                            label="Hugging Face Dataset Repository",
                            placeholder="örn: username/dataset-repo"
                        )
                        dataset_privacy_exp = gr.Radio(
                            choices=["public", "private"],
                            label="Dataset Privacy",
                            value="public"
                        )
                        hf_token_exp = gr.Textbox(
                            label="Hugging Face Token (private datasets)",
                            placeholder="HF token",
                            type="password"
                        )
                    with gr.Column():
                        explore_output = gr.Textbox(label="Dataset Bilgisi", lines=15)
                def toggle_dataset_inputs_exp(choice):
                    return {
                        dataset_path_exp: gr.update(visible=choice == "local"),
                        local_format_exp: gr.update(visible=choice == "local"),
                        dataset_repo_exp: gr.update(visible=choice == "huggingface"),
                        dataset_privacy_exp: gr.update(visible=choice == "huggingface")
                    }
                dataset_choice_exp.change(toggle_dataset_inputs_exp, dataset_choice_exp, 
                                            [dataset_path_exp, local_format_exp, dataset_repo_exp, dataset_privacy_exp])
                explore_btn = gr.Button("Explore Dataset")
                explore_btn.click(
                    explore_dataset,
                    inputs=[dataset_choice_exp, dataset_path_exp, local_format_exp, dataset_repo_exp, dataset_privacy_exp, hf_token_exp],
                    outputs=explore_output
                )

            with gr.TabItem("Fine-Tuning"):
                gr.Markdown("### Model Fine-Tuning Ayarları")
                with gr.Row():
                    with gr.Column():
                        model_type = gr.Radio(
                            choices=["public", "private"],
                            label="Model Type",
                            value="public"
                        )
                        hf_token = gr.Textbox(
                            label="Hugging Face Token (for private models)",
                            placeholder="HF token",
                            type="password"
                        )
                        base_model_name = gr.Textbox(
                            label="Base Model Name",
                            placeholder="örn: username/model-name"
                        )
                        dataset_choice = gr.Radio(
                            choices=["local", "huggingface"],
                            label="Dataset Source",
                            value="huggingface"
                        )
                        dataset_path = gr.Textbox(
                            label="Local Dataset Path",
                            placeholder="Yerel dosya yolu (örn: data.json veya data.csv)",
                            visible=False
                        )
                        local_format = gr.Radio(
                            choices=["json", "csv"],
                            label="Local Dataset Format",
                            value="json",
                            visible=False
                        )
                        dataset_repo = gr.Textbox(
                            label="Hugging Face Dataset Repository",
                            placeholder="örn: username/dataset-repo"
                        )
                        dataset_privacy = gr.Radio(
                            choices=["public", "private"],
                            label="Dataset Privacy",
                            value="public"
                        )
                        system_col = gr.Textbox(
                            label="System Column Name",
                            placeholder="örn: context veya istenilen sütun",
                            value="context"
                        )
                        user_col = gr.Textbox(
                            label="User Column Name",
                            placeholder="örn: soru",
                            value="soru"
                        )
                        assistant_col = gr.Textbox(
                            label="Assistant Column Name",
                            placeholder="örn: cevap",
                            value="cevap"
                        )
                    with gr.Column():
                        max_length = gr.Number(
                            label="Maximum Sequence Length",
                            value=1024,
                            minimum=1,
                            maximum=8192
                        )
                        gradient_accumulation_steps = gr.Number(
                            label="Gradient Accumulation Steps",
                            value=1,
                            minimum=1
                        )
                        per_device_train_batch_size = gr.Number(
                            label="Per Device Train Batch Size",
                            value=1,
                            minimum=1
                        )
                        num_epochs = gr.Number(
                            label="Number of Epochs",
                            value=3,
                            minimum=1
                        )
                        quantization_choice = gr.Checkbox(
                            label="Use Quantization",
                            value=False
                        )
                        bit_choice = gr.Radio(
                            choices=["4", "8"],
                            label="Quantization Bits",
                            value="4",
                            visible=False
                        )
                        precision_choice = gr.Radio(
                            choices=["fp16", "bf16", "fp32"],
                            label="Precision Mode",
                            value="fp16"
                        )
                def toggle_dataset_inputs(choice):
                    return {
                        dataset_path: gr.update(visible=choice == "local"),
                        local_format: gr.update(visible=choice == "local"),
                        dataset_repo: gr.update(visible=choice == "huggingface"),
                        dataset_privacy: gr.update(visible=choice == "huggingface")
                    }
                def toggle_quantization(choice):
                    return {
                        bit_choice: gr.update(visible=choice),
                        precision_choice: gr.update(visible=not choice)
                    }
                dataset_choice.change(toggle_dataset_inputs, dataset_choice, [dataset_path, local_format, dataset_repo, dataset_privacy])
                quantization_choice.change(toggle_quantization, quantization_choice, [bit_choice, precision_choice])

                output = gr.Textbox(label="Training Status", lines=8)
                submit_btn = gr.Button("Start Training")
                submit_btn.click(
                    train_model,
                    inputs=[
                        model_type,
                        hf_token,
                        base_model_name,
                        dataset_choice,
                        dataset_path,
                        local_format,
                        dataset_repo,
                        dataset_privacy,
                        max_length,
                        gradient_accumulation_steps,
                        per_device_train_batch_size,
                        num_epochs,
                        quantization_choice,
                        bit_choice,
                        precision_choice,
                        system_col,
                        user_col,
                        assistant_col
                    ],
                    outputs=output
                )

        gr.Markdown("### Notlar:")
        gr.Markdown("- Lokal dataset kullanıyorsanız, dosya formatını (json/csv) seçmeyi unutmayın.")
        gr.Markdown("- Hugging Face datasetlerinde gizlilik durumuna göre HF Token girmeniz gerekebilir.")
        gr.Markdown("- Veri setinizdeki sütun isimleri farklı ise, ilgili alanlardan doğru isimleri belirleyin (örneğin, sizin datasetinizde 'soru', 'cevap', 'context' gibi sütunlar mevcut).")

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
