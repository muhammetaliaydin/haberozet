# -*- coding: utf-8 -*-
"""NLP ön işleme modülü.

NLTK tabanlı cümle tokenizasyonu, stop-words filtreleme ve
metin temizleme fonksiyonlarını içerir. Türkçe dil desteği sağlar.
"""

import re
import logging

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# ── Metadata / gürültü kalıpları ───────────────────────────────────
_NOISE_PATTERNS: list[re.Pattern] = [
    # "Kaynak, …" — fotoğraf / görsel kaynağı
    re.compile(r"^Kaynak\s*[,:]", re.IGNORECASE),
    # "Fotoğraf altı yazısı, …"
    re.compile(r"^Fotoğra?f\s*(altı\s*yazısı|açıklaması)\s*[,:]", re.IGNORECASE),
    # "Yazan, … Unvan, …" veya sadece "Yazan, …"
    re.compile(r"^Yazan\s*[,:]", re.IGNORECASE),
    # "Unvan, …"
    re.compile(r"^Unvan\s*[,:]", re.IGNORECASE),
    # "Okuma süresi 7 dk"
    re.compile(r"^Okuma\s+süresi\s+\d+", re.IGNORECASE),
    # Sade tarih satırları: "25 Şubat 2026", "3 Mart 2025" vb.
    re.compile(
        r"^\d{1,2}\s+"
        r"(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)"
        r"\s+\d{4}$",
        re.IGNORECASE,
    ),
    # "Getty Images", "Reuters", "AFP" gibi tek başına kaynak isimleri
    re.compile(r"^(Getty\s*Images|Reuters|AFP|AP|AA|İHA|DHA)\s*$", re.IGNORECASE),
    # "BBC …" tek başına kaynak satırı
    re.compile(r"^BBC\s+\w+$", re.IGNORECASE),
]


def clean_article_text(text: str, title: str = "") -> str:
    """Haber metninden metadata / gürültü satırlarını temizler.

    newspaper3k bazen fotoğraf alt yazısı, yazar bilgisi, tarih ve
    okuma süresi gibi satırları makale metnine dahil eder. Bu fonksiyon
    bu tür satırları filtreler.

    Args:
        text: Ham makale metni.
        title: Haber başlığı (metin içinde tekrarlanıyorsa çıkarılır).

    Returns:
        Temizlenmiş makale metni.
    """
    # Başlık karşılaştırması için normalize edilmiş versiyon
    norm_title = re.sub(r"\s+", " ", title.strip().lower()) if title else ""

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()

        # Boş satırları koru (paragraf ayracı)
        if not stripped:
            cleaned_lines.append("")
            continue

        # Bilinen gürültü kalıplarına uyan satırları at
        if any(pat.search(stripped) for pat in _NOISE_PATTERNS):
            continue

        # Başlık ile aynı veya çok benzer satırları at
        if norm_title:
            norm_line = re.sub(r"\s+", " ", stripped.lower())
            if norm_line == norm_title or norm_title in norm_line or norm_line in norm_title:
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()



def setup_nltk() -> None:
    """Gerekli NLTK paketlerini sessizce indirir.

    İndirilen paketler: punkt, punkt_tab, stopwords.
    Uygulama başında bir kez çağrılmalıdır.
    """
    for package in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(package, quiet=True)


def tokenize_sentences(text: str) -> list[str]:
    """Metni cümlelere böler ve kısa cümleleri filtreler.

    Args:
        text: Cümlelere bölünecek ham metin.

    Returns:
        20 karakterden uzun cümlelerin listesi.
    """
    raw_sentences = sent_tokenize(text, language="turkish")
    return [s.strip() for s in raw_sentences if len(s.strip()) >= 20]


def preprocess_sentence(sentence: str, stop_words: set) -> str:
    """Cümleyi NLP için ön işlemden geçirir.

    İşlemler:
        1. Küçük harfe çevirme (Türkçe karakter duyarlı)
        2. Noktalama ve sayıları silme
        3. Stop-words filtreleme

    Args:
        sentence: İşlenecek ham cümle.
        stop_words: Filtrelenecek stop-words seti.

    Returns:
        Temizlenmiş ve filtrelenmiş cümle metni.
    """
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    sentence = re.sub(r"\d+", "", sentence)
    words = sentence.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def get_turkish_stopwords() -> set:
    """Türkçe ve İngilizce stop-words setini döndürür.

    NLTK İngilizce stop-words listesine ek olarak 50'den fazla
    Türkçe stop-word içerir.

    Returns:
        Birleştirilmiş stop-words seti.
    """
    try:
        english_sw = set(stopwords.words("english"))
    except LookupError:
        setup_nltk()
        english_sw = set(stopwords.words("english"))

    turkish_sw = {
        "ve", "veya", "ama", "ile", "bu", "şu", "o", "bir", "için",
        "de", "da", "ki", "mi", "mı", "mu", "mü", "gibi", "kadar",
        "daha", "çok", "az", "her", "hiç", "en", "ne", "olan",
        "olarak", "edildi", "yapıldı", "göre", "sonra", "önce", "ancak",
        "fakat", "lakin", "yani", "çünkü", "eğer", "ise", "hem", "ya",
        "diye", "üzere", "karşı", "rağmen", "doğru", "beri",
        "itibaren", "arasında", "içinde", "dışında", "üzerinde", "altında",
        "ben", "sen", "biz", "siz", "onlar", "benim", "senin", "onun",
        "bizim", "sizin", "onların", "bana", "sana", "ona", "bize", "size",
        "onlara", "beni", "seni", "onu", "bizi", "sizi", "onları",
        "var", "yok", "değil", "bile", "sadece", "artık", "henüz",
    }

    return english_sw | turkish_sw


if __name__ == "__main__":
    setup_nltk()

    sample_text = (
        "Türkiye'de hava durumu bu hafta değişkenlik gösterecek. "
        "Meteoroloji Genel Müdürlüğü, İstanbul ve Ankara başta olmak üzere "
        "birçok ilde yağış beklendiğini açıkladı. Vatandaşlara şemsiye "
        "taşımaları tavsiye ediliyor. Sıcaklıklar mevsim normallerinin "
        "altında seyredecek."
    )

    sentences = tokenize_sentences(sample_text)
    print(f"Cümle sayısı: {len(sentences)}")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s}")

    sw = get_turkish_stopwords()
    print(f"\nStop-words sayısı: {len(sw)}")

    if sentences:
        cleaned = preprocess_sentence(sentences[0], sw)
        print(f"\nOrijinal: {sentences[0]}")
        print(f"Temiz   : {cleaned}")
