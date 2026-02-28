# 📰 HaberÖzet — NLP Tabanlı Haber Özetleyici

Türkçe haber makalelerini otomatik olarak özetleyen, istatistiksel NLP ve derin öğrenme
yöntemlerini kullanan bir Streamlit web uygulaması. Kullanıcının yapıştırdığı haber
URL'sinden metni çeker ve üç farklı yöntemle özetler:

- **Abstractive (mT5):** Türkçe için eğitilmiş mT5 modeli ile kendi özet cümlesini üretir
- **TextRank:** Graf tabanlı extractive özetleme
- **TF-IDF:** İstatistiksel extractive özetleme

## 🚀 Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. NLTK verilerini indir
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# 3. Uygulamayı çalıştır
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

## 📂 Dosya Yapısı

| Dosya              | Açıklama                                      |
|--------------------|-----------------------------------------------|
| `app.py`           | Streamlit arayüzü                             |
| `scraper.py`       | newspaper3k ile haber çekme modülü            |
| `preprocessor.py`  | NLTK tabanlı NLP ön işleme modülü             |
| `summarizer.py`    | TF-IDF, TextRank ve Abstractive özetleme motoru |
| `requirements.txt` | Python bağımlılıkları                         |

## 🌐 Desteklenen Haber Siteleri

Uygulama, `newspaper3k` kütüphanesi sayesinde standart HTML yapısına sahip çoğu haber
sitesinden içerik çekebilir. Türkçe haber siteleri için optimize edilmiştir:

- BBC Türkçe
- TRT Haber
- Hürriyet
- Sabah
- NTV
- CNN Türk
- Sözcü
- ve standart makale yapısına sahip diğer tüm siteler

## 🔬 Özetleme Yöntemleri

| Özellik         | Abstractive (mT5)                              | TextRank                                       | TF-IDF                                         |
|-----------------|-------------------------------------------------|------------------------------------------------|------------------------------------------------|
| **Yaklaşım**    | Derin öğrenme: mT5 modeli ile kendi özet cümlesini üretir | Graf tabanlı: cümleler arası benzerlik grafi + PageRank | İstatistiksel: TF-IDF skoru ile cümle seçimi |
| **Güçlü Yanı**  | İnsan benzeri özet üretir, orijinal cümlelere bağlı değildir | Bağlam bütünlüğünü iyi korur | Hızlı, nadir ama önemli terimleri öne çıkarır |
| **Kullanım**    | En iyi kaliteli özet için                        | Uzun ve karmaşık haberlerde                    | Kısa ve bilgi yoğun haberlerde                 |

> **Not:** Abstractive yöntem ilk çalıştırmada modeli indirir (~2 GB). Sonraki
> kullanımlarda önbellekten yüklenir.

## 📝 Lisans

Bu proje eğitim ve kişisel kullanım amaçlıdır.
