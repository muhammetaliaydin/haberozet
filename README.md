# 📰 Sumlify — NLP Tabanlı Türkçe Haber Özetleyici

Türkçe haber makalelerini otomatik olarak özetleyen, istatistiksel NLP yöntemlerini kullanan bir **Streamlit** web uygulaması.

Kullanıcı bir haber URL'si girer, uygulama makaleyi çeker ve **TF-IDF** veya **TextRank** algoritmasıyla en önemli cümleleri seçerek kısa bir özet oluşturur.

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Depoyu klonla
git clone https://github.com/<kullanici>/Sumlify.git
cd Sumlify

# 2. Sanal ortam oluştur ve aktifleştir
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -r haberozet/requirements.txt

# 4. Uygulamayı başlat
streamlit run haberozet/app.py
```

Uygulama varsayılan olarak **http://localhost:8501** adresinde açılır.

> **Not:** NLTK veri paketleri (`punkt`, `punkt_tab`, `stopwords`) uygulama ilk çalıştığında otomatik olarak indirilir.

---

## 📂 Proje Yapısı

```
Sumlify/
├── haberozet/
│   ├── app.py              # Streamlit arayüzü (ana uygulama)
│   ├── scraper.py           # newspaper3k ile haber çekme modülü
│   ├── preprocessor.py      # NLTK tabanlı NLP ön işleme modülü
│   ├── summarizer.py        # TF-IDF + TextRank özetleme motoru
│   └── requirements.txt     # Python bağımlılıkları
└── README.md
```

---

## ⚙️ Modüller

### `scraper.py`
`newspaper3k` kütüphanesi ile verilen URL'den haber metnini ve başlığını çeker. Türkçe dil desteği ile çalışır.

### `preprocessor.py`
- **Cümle tokenizasyonu** — NLTK `sent_tokenize` ile Türkçe cümle bölme
- **Stop-words filtreleme** — 50+ Türkçe stop-word ve İngilizce NLTK stop-words
- **Metin temizleme** — Küçük harfe çevirme, noktalama/sayı silme

### `summarizer.py`
İki farklı **extractive** özetleme algoritması sunar:

| Özellik | TextRank | TF-IDF |
|---------|----------|--------|
| **Yaklaşım** | Graf tabanlı — cümleler arası benzerlik grafi kurar, PageRank ile skorlar | İstatistiksel — her cümlenin TF-IDF skorunu hesaplar |
| **Güçlü Yanı** | Bağlam bütünlüğünü iyi korur | Hızlı, bilgi yoğun cümleleri öne çıkarır |
| **İdeal Kullanım** | Uzun ve karmaşık haberler | Kısa ve bilgi yoğun haberler |

### `app.py`
Streamlit tabanlı web arayüzü. Sidebar'dan yöntem ve cümle sayısı seçilir, sonuçlar orijinal metin ve özet olarak yan yana gösterilir. Sıkıştırma oranı ve cümle bazlı detaylar da sunulur.

---

## 🌐 Desteklenen Haber Siteleri

`newspaper3k` sayesinde standart HTML yapısına sahip çoğu haber sitesinden içerik çekilebilir:

- BBC Türkçe · TRT Haber · Hürriyet · Sabah
- NTV · CNN Türk · Sözcü
- ve standart makale yapısına sahip diğer siteler

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Amaç |
|-----------|------|
| [Streamlit](https://streamlit.io/) | Web arayüzü |
| [newspaper3k](https://github.com/codelucas/newspaper) | Haber çekme ve ayrıştırma |
| [NLTK](https://www.nltk.org/) | Cümle tokenizasyonu, stop-words |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF vektörizasyonu |
| [NetworkX](https://networkx.org/) | TextRank graf algoritması |
| [NumPy](https://numpy.org/) | Sayısal hesaplamalar |

---

## 📝 Lisans

Bu proje eğitim ve kişisel kullanım amaçlıdır.
