LLMTrain
Bu proje, büyük dil modellerinin (LLM) ince ayar (fine-tuning) işlemlerini ve veri seti keşfini (dataset exploration) gerçekleştiren esnek ve kullanıcı dostu bir arayüz sunmaktadır. Proje, yerel veya Hugging Face üzerinden veri seti yükleyerek, modelinizi kolayca eğitmenize olanak tanır. Quantization, precision ve LoRA gibi gelişmiş ayar seçenekleriyle, farklı senaryolara uyum sağlayan kapsamlı bir çözüm sunar.

Özellikler
Veri Seti Keşfi:

Yerel (JSON/CSV) veya Hugging Face veri setlerini yükleyip, veri setinin sütun isimlerini, toplam örnek sayısını ve örnek verileri gösterir.
Model Fine-Tuning:

Kullanıcı dostu Gradio arayüzü ile model tipi (public/private), Hugging Face Token, temel model adı, veri seti kaynakları, sütun isimleri (ör. soru, cevap, context) gibi parametreler kolayca ayarlanır.
Eğitim parametreleri (epoch, batch size, gradient accumulation, learning rate, vs.) yapılandırılabilir.
Quantization & Precision:

Modelinizi 4-bit veya 8-bit quantization modlarında eğitme imkanı.
fp16, bf16 veya fp32 precision seçenekleriyle esnek hesaplama seçenekleri.
LoRA Desteği:

İnce ayar işlemleri için LoRA (Low-Rank Adaptation) konfigürasyonu sayesinde, model parametrelerinde hafif ve verimli güncellemeler yapılır.
Gereksinimler
Python 3.8+
CUDA Uyumlu PyTorch:
GPU üzerinde eğitim yapmak için uygun CUDA sürümünü destekleyen PyTorch kurulmalıdır. Örneğin, CUDA 11.7 kullanıyorsanız:
bash
Kopyala
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
Diğer Gerekli Kütüphaneler:
bash
Kopyala
pip install transformers datasets gradio accelerate peft huggingface_hub pandas
NVIDIA Sürücüleri ve CUDA Toolkit:
GPU üzerinde çalışabilmek için uygun NVIDIA sürücülerinin ve CUDA Toolkit'in kurulu olması gerekmektedir.
Kurulum
Depoyu Klonlayın:
bash
Kopyala
git clone https://github.com/kullanici_adiniz/LLMTrain.git
cd LLMTrain
Sanal Ortam Oluşturun (isteğe bağlı):
bash
Kopyala
python -m venv myenv
source myenv/bin/activate   # Windows: myenv\Scripts\activate
Gerekli Kütüphaneleri Yükleyin: Eğer bir requirements.txt dosyanız varsa:
bash
Kopyala
pip install -r requirements.txt
Alternatif olarak, yukarıdaki kütüphane yükleme komutlarını çalıştırın.
Kullanım
Projeyi çalıştırmak için aşağıdaki komutu kullanın:

bash
Kopyala
python llm.py
Bu komut Gradio arayüzünü başlatır. Arayüz, yerel olarak http://127.0.0.1:7860 adresinde çalışır; ayrıca paylaşım linki üzerinden de erişilebilir.

Arayüz Bölümleri
Dataset Exploration:

Veri Kaynağı Seçimi: Lokal dosya veya Hugging Face dataset.
Bilgi Görüntüleme: Veri seti sütunları, toplam örnek sayısı ve ilk 5 örneğin detayları gösterilir.
Fine-Tuning:

Model Ayarları: Model tipi, Hugging Face Token, temel model adı.
Veri Seti Ayarları: Veri seti kaynağı, dosya yolu veya repo bilgileri, sütun isimleri (örneğin; soru, cevap, context).
Eğitim Parametreleri: Maksimum sequence length, gradient accumulation, batch size, epoch sayısı, learning rate vb.
Quantization & Precision: Quantization modları (4-bit/8-bit) ve precision seçenekleri (fp16, bf16, fp32).
LoRA Desteği: İnce ayar için LoRA konfigürasyonu uygulanır.
Proje Akışı
Veri Seti Yükleme & Keşif:
Seçilen veri seti (yerel veya Hugging Face) yüklenir ve arayüzde sütun bilgileri, toplam örnek sayısı ve örnek veriler görüntülenir.

Model Yükleme:
Kullanıcı tarafından belirlenen ayarlara göre model, quantization/precision seçenekleriyle yüklenir. GPU veya CPU üzerinde çalışması için gerekli ayarlamalar yapılır.

Ön İşleme:
Veri setindeki her örnek, belirlenen sütun isimlerine göre tokenize edilip modelin eğitimi için gerekli formata dönüştürülür.

Eğitim:
Hugging Face Trainer API’si kullanılarak model eğitimi gerçekleştirilir. Erken durdurma (early stopping) gibi callback'ler ile eğitim süreci optimize edilir.

Model Kaydetme:
Eğitim tamamlandıktan sonra model ve tokenizer belirtilen dizine kaydedilir.

Notlar
Windows Kullanıcıları:
Windows ortamında symlink uyarıları alabilirsiniz. Bu uyarıları önlemek için Developer Mode'u aktif edebilir veya Python'u yönetici olarak çalıştırabilirsiniz.

CUDA Uyumlu Ortam:
GPU üzerinde eğitim yapabilmek için uygun NVIDIA sürücülerinizin, CUDA Toolkit'in ve uyumlu PyTorch sürümünüzün kurulu olduğundan emin olun.

Katkıda Bulunma
Her türlü katkı, hata bildirimi veya öneriler memnuniyetle karşılanmaktadır. Lütfen pull request veya issue açarak katkıda bulunun.
