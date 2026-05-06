# -*- coding: utf-8 -*-
"""TF-IDF, TextRank ve Abstractive özetleme modülü.

Üç farklı özetleme algoritması sunar:
  - TF-IDF: Extractive (cümle seçme)
  - TextRank: Extractive (cümle seçme)
  - Abstractive: mT5 tabanlı Türkçe model ile kendi cümlesini üretir
"""

import logging

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from preprocessor import setup_nltk, tokenize_sentences, preprocess_sentence, get_turkish_stopwords

logger = logging.getLogger(__name__)

# ── Abstractive model sabitleri ────────────────────────────────────
_MODEL_NAME = "yeniguno/turkish-abstractive-summary-mt5"
_tokenizer = None
_model = None
_MAX_INPUT_TOKENS = 512


def tfidf_summarize(
    sentences: list[str],
    clean_sentences: list[str],
    n: int,
) -> list[str]:
    """TF-IDF skorlarına göre en önemli cümleleri seçer.

    Her cümle için TF-IDF vektörünün satır toplamı cümle skoru olarak
    hesaplanır. En yüksek skorlu n cümle, orijinal metindeki sırası
    korunarak döndürülür.

    Args:
        sentences: Orijinal (ham) cümle listesi.
        clean_sentences: Ön işlemden geçmiş cümle listesi.
        n: Seçilecek özet cümle sayısı.

    Returns:
        Seçilen orijinal cümlelerin listesi (orijinal sırada).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)

    sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    ranked_indices = sentence_scores.argsort()[::-1][:n]
    selected_indices = sorted(ranked_indices)

    return [sentences[i] for i in selected_indices]


def textrank_summarize(
    sentences: list[str],
    clean_sentences: list[str],
    n: int,
) -> list[str]:
    """TextRank algoritması ile en önemli cümleleri seçer.

    TF-IDF vektörleriyle cümle benzerlik matrisi oluşturur, networkx
    ile graf kurar ve PageRank ile cümle skorlarını hesaplar.

    Args:
        sentences: Orijinal (ham) cümle listesi.
        clean_sentences: Ön işlemden geçmiş cümle listesi.
        n: Seçilecek özet cümle sayısı.

    Returns:
        Seçilen orijinal cümlelerin listesi (orijinal sırada).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)

    sim_matrix = cosine_similarity(tfidf_matrix)

    graph = nx.from_numpy_array(sim_matrix)

    scores = nx.pagerank(graph, max_iter=200)

    ranked_indices = sorted(scores, key=scores.get, reverse=True)[:n]
    selected_indices = sorted(ranked_indices)

    return [sentences[i] for i in selected_indices]


def load_abstractive_model():
    """mT5 modelini ve tokenizer'ı yükler (lazy loading).

    İlk çağrıda model indirilir/yüklenir, sonraki çağrılarda
    cache'ten döndürülür.

    Returns:
        (tokenizer, model) tuple.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info("Abstractive model yükleniyor: %s", _MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
        logger.info("Model başarıyla yüklendi.")
    return _tokenizer, _model


def abstractive_summarize(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
) -> list[str]:
    """mT5 tabanlı abstractive özetleme yapar.

    Metin, modelin token limitine göre parçalara bölünür.
    Her parça ayrı özetlenir ve sonuçlar birleştirilir.

    Args:
        text: Özetlenecek ham metin.
        max_length: Üretilecek özet için maksimum token sayısı.
        min_length: Üretilecek özet için minimum token sayısı.

    Returns:
        Üretilen özet cümlelerinin listesi.
    """
    tokenizer, model = load_abstractive_model()

    # Metni parçalara böl (model input limiti)
    prefix = "summarize: "
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )
    input_ids = inputs["input_ids"][0]
    total_tokens = len(input_ids)

    if total_tokens <= _MAX_INPUT_TOKENS:
        # Tek parça — doğrudan özetle
        chunks_text = [prefix + text]
    else:
        # Uzun metin — cümle bazlı parçala
        sentences = tokenize_sentences(text)
        chunks_text = []
        current_chunk = prefix
        for sent in sentences:
            candidate = current_chunk + " " + sent if current_chunk != prefix else prefix + sent
            tok_len = len(tokenizer.encode(candidate, add_special_tokens=True))
            if tok_len > _MAX_INPUT_TOKENS and current_chunk != prefix:
                chunks_text.append(current_chunk)
                current_chunk = prefix + sent
            else:
                current_chunk = candidate
        if current_chunk and current_chunk != prefix:
            chunks_text.append(current_chunk)

    summaries = []
    for chunk in chunks_text:
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=_MAX_INPUT_TOKENS,
            truncation=True,
        )
        output_ids = model.generate(
            enc["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if decoded.strip():
            summaries.append(decoded.strip())

    return summaries


# ── Özel Sezgisel Skorlama Algoritması ─────────────────────────────
# Hiçbir makine öğrenmesi kütüphanesi kullanmadan, haber metinlerinin
# doğasına özgü 5 farklı kriter ile cümle puanlama yapan algoritmamız.

def _position_score(index: int, total: int) -> float:
    """Ters Piramit Kuralı — cümlenin metindeki konumuna göre puan.

    Gazetecilikte en önemli bilgi haberin başında verilir (ters piramit).
    Metnin son cümlesi de genellikle toparlama niteliği taşır.

    İlk %20  → 1.0  (en önemli bölge)
    Son cümle → 0.6  (toparlama)
    Ortalar   → Doğrusal azalma ile 0.2 – 0.5 arası
    """
    if total <= 1:
        return 1.0
    relative = index / (total - 1)  # 0.0 (baş) – 1.0 (son)

    if relative <= 0.2:
        return 1.0
    if index == total - 1:
        return 0.6
    # Doğrusal azalma: 0.2 konumunda 0.5, 0.8 konumunda 0.2
    return max(0.2, 0.5 - (relative - 0.2) * 0.5)


def _title_overlap_score(sentence_words: list[str], title_words: set[str]) -> float:
    """Başlık Örtüşmesi — cümledeki kelimelerin başlıkla ne kadar örtüştüğü.

    Haberin başlığı ana fikri yansıtır. Başlıkta geçen anlamlı
    kelimeleri barındıran cümleler daha önemlidir.

    Returns:
        0.0 – 1.0 arası örtüşme oranı.
    """
    if not title_words or not sentence_words:
        return 0.0
    overlap = sum(1 for w in sentence_words if w in title_words)
    return min(1.0, overlap / len(title_words))


def _numerical_score(sentence: str) -> float:
    """Sayısal Veri Bonusu — istatistik, tarih, para birimi içeren cümlelere bonus.

    Haberlerde sayısal veriler (yüzdeler, tarihler, para miktarları)
    genellikle kritik bilgi taşır.

    Returns:
        0.0, 0.3 veya 0.5 — bulunan sayısal öğe sayısına göre.
    """
    import re as _re
    indicators = [
        _re.compile(r'\d+'),               # herhangi bir rakam
        _re.compile(r'%\s?\d+|\d+\s?%'),    # yüzde ifadesi
        _re.compile(r'[₺$€£]'),             # para birimi sembolleri
        _re.compile(r'\b(milyon|milyar|trilyon|bin)\b', _re.IGNORECASE),
        _re.compile(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b'),  # tarih formatı
    ]
    hit_count = sum(1 for pat in indicators if pat.search(sentence))
    if hit_count >= 3:
        return 0.5
    if hit_count >= 1:
        return 0.3
    return 0.0


def _keyword_density_score(
    sentence_words: list[str],
    top_keywords: dict[str, float],
) -> float:
    """Anahtar Kelime Yoğunluğu — kendi TF hesabımızla en sık geçen kelimeleri skorla.

    Metin genelindeki en sık tekrarlanan anlamlı kelimeleri (stop-words
    hariç) bulur ve cümledeki bu kelimelerin toplam frekans ağırlığını
    hesaplar.

    Args:
        sentence_words: Cümledeki kelime listesi (küçük harf, temiz).
        top_keywords: {kelime: normalize_frekans} sözlüğü.

    Returns:
        0.0 – 1.0 arası normalize skor.
    """
    if not top_keywords or not sentence_words:
        return 0.0
    score = sum(top_keywords.get(w, 0.0) for w in sentence_words)
    # Normalize: cümle uzunluğuna böl, max 1.0 ile sınırla
    return min(1.0, score / max(len(sentence_words), 1))


def _length_penalty(word_count: int) -> float:
    """Uzunluk Cezası — çok kısa veya çok uzun cümlelere ceza uygular.

    Özet için ideal cümle uzunluğu 8-30 kelime arasıdır.
    Bunun dışındaki cümleler puan kaybeder.

    Returns:
        0.0 – 1.0 arası çarpan.
    """
    if word_count < 5:
        return 0.1   # çok kısa → neredeyse ele
    if word_count < 8:
        return 0.6
    if word_count <= 30:
        return 1.0   # ideal aralık
    if word_count <= 40:
        return 0.7
    return 0.4        # çok uzun


def _compute_top_keywords(
    all_words: list[str],
    stop_words: set[str],
    top_n: int = 15,
) -> dict[str, float]:
    """Metin genelinde en sık geçen anlamlı kelimeleri bulur (kendi TF).

    scikit-learn veya benzeri kütüphane kullanmadan, sade Python ile
    kelime frekanslarını hesaplar ve normalize eder.

    Args:
        all_words: Tüm metindeki kelime listesi.
        stop_words: Filtrelenecek stop-word seti.
        top_n: Döndürülecek en popüler kelime sayısı.

    Returns:
        {kelime: 0.0-1.0 arası normalize frekans} sözlüğü.
    """
    import re as _re
    freq: dict[str, int] = {}
    for w in all_words:
        w = _re.sub(r'[^\w]', '', w.lower())
        if len(w) < 3 or w in stop_words or w.isdigit():
            continue
        freq[w] = freq.get(w, 0) + 1

    if not freq:
        return {}

    # En yüksek frekansa göre normalize et
    max_freq = max(freq.values())
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {word: count / max_freq for word, count in sorted_words}


def custom_heuristic_summarize(
    sentences: list[str],
    clean_sentences: list[str],
    n: int,
    title: str = "",
) -> list[str]:
    """Özel Çoklu-Kriter Sezgisel Skorlama Algoritması.

    Hiçbir makine öğrenmesi kütüphanesi kullanmadan, haber metinlerinin
    doğasına özgü 5 farklı kriteri birleştirerek her cümleye bir önem
    puanı atar ve en yüksek puanlı cümleleri seçer.

    Kriterler ve Ağırlıkları:
        1. Konum Skoru       (w=0.25) — Ters piramit kuralı
        2. Başlık Örtüşmesi  (w=0.25) — Başlıktaki anahtar kelimelerle örtüşme
        3. Sayısal Veri      (w=0.10) — Rakam, yüzde, para birimi varlığı
        4. Anahtar Kelime     (w=0.30) — Metindeki en sık anlamlı kelimelere yakınlık
        5. Uzunluk Cezası    (w=0.10) — Çok kısa/uzun cümleleri cezalandırma

    Args:
        sentences: Orijinal (ham) cümle listesi.
        clean_sentences: Ön işlemden geçmiş cümle listesi.
        n: Seçilecek özet cümle sayısı.
        title: Haber başlığı (başlık örtüşmesi için).

    Returns:
        Seçilen orijinal cümlelerin listesi (orijinal sırada).
    """
    import re as _re

    total = len(sentences)
    if total == 0:
        return []
    if n >= total:
        return list(sentences)

    # ── Hazırlık ───────────────────────────────────────────────────
    stop_words = get_turkish_stopwords()

    # Başlık kelimelerini hazırla (stop-words hariç)
    title_words: set[str] = set()
    if title:
        for w in _re.sub(r'[^\w\s]', '', title.lower()).split():
            if w not in stop_words and len(w) >= 3:
                title_words.add(w)

    # Tüm metindeki kelimelerden anahtar kelime tablosu oluştur
    all_words = []
    for cs in clean_sentences:
        all_words.extend(cs.split())
    top_keywords = _compute_top_keywords(all_words, stop_words)

    # ── Ağırlıklar ─────────────────────────────────────────────────
    W_POSITION = 0.25
    W_TITLE    = 0.25
    W_NUMERIC  = 0.10
    W_KEYWORD  = 0.30
    W_LENGTH   = 0.10

    # ── Her cümleyi puanla ─────────────────────────────────────────
    scores: list[tuple[int, float]] = []
    for i, (orig, clean) in enumerate(zip(sentences, clean_sentences)):
        words = clean.split()

        s_position = _position_score(i, total)
        s_title    = _title_overlap_score(words, title_words)
        s_numeric  = _numerical_score(orig)
        s_keyword  = _keyword_density_score(words, top_keywords)
        s_length   = _length_penalty(len(orig.split()))

        total_score = (
            W_POSITION * s_position
            + W_TITLE  * s_title
            + W_NUMERIC * s_numeric
            + W_KEYWORD * s_keyword
            + W_LENGTH  * s_length
        )
        scores.append((i, total_score))

    # En yüksek puanlı n cümleyi seç, orijinal sırayı koru
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    selected_indices = sorted(idx for idx, _ in ranked)

    return [sentences[i] for i in selected_indices]


def summarize(
    text: str,
    n: int = 3,
    method: str = "textrank",
    title: str = "",
) -> dict:
    """Metni seçilen yöntemle özetler.

    Args:
        text: Özetlenecek ham metin.
        n: İstenen özet cümle sayısı (extractive yöntemler için).
        method: Özetleme yöntemi — "textrank", "tfidf", "custom" veya "abstractive".
        title: Haber başlığı (custom yöntemi için kullanılır).

    Returns:
        Şu anahtarları içeren dict:
            - summary (str): Özet metni
            - sentences (list[str]): Özet cümleleri liste olarak
            - sentence_count (int): Toplam cümle sayısı
            - compression_ratio (float): Özet / toplam cümle oranı (yüzde)
            - error (str | None): Hata varsa mesaj
    """
    result: dict = {
        "summary": "",
        "sentences": [],
        "sentence_count": 0,
        "compression_ratio": 0.0,
        "error": None,
    }

    try:
        sentences = tokenize_sentences(text)
        result["sentence_count"] = len(sentences)

        if not sentences:
            result["error"] = "Metinde yeterli cümle bulunamadı."
            return result

        stop_words = get_turkish_stopwords()
        clean_sentences = [
            preprocess_sentence(s, stop_words) for s in sentences
        ]

        # Boş temizlenmiş cümleleri filtrele (TF-IDF hata verebilir)
        valid_pairs = [
            (s, c) for s, c in zip(sentences, clean_sentences) if c.strip()
        ]
        if not valid_pairs:
            result["error"] = "Ön işleme sonrası yeterli içerik kalmadı."
            return result

        sentences_filtered = [p[0] for p in valid_pairs]
        clean_filtered = [p[1] for p in valid_pairs]

        actual_n = min(n, len(sentences_filtered))

        if method == "abstractive":
            selected = abstractive_summarize(text)
        elif method == "custom":
            selected = custom_heuristic_summarize(
                sentences_filtered, clean_filtered, actual_n, title=title
            )
        elif method == "tfidf":
            selected = tfidf_summarize(sentences_filtered, clean_filtered, actual_n)
        else:
            selected = textrank_summarize(sentences_filtered, clean_filtered, actual_n)

        result["sentences"] = selected
        result["summary"] = " ".join(selected)
        result["compression_ratio"] = round(
            (len(selected) / result["sentence_count"]) * 100, 1
        )

    except Exception as exc:
        logger.error("Özetleme sırasında hata: %s", exc)
        result["error"] = f"Özetleme hatası: {exc}"

    return result


if __name__ == "__main__":
    setup_nltk()

    sample = (
        "Türkiye'de ekonomi gündeminde önemli gelişmeler yaşanıyor. "
        "Merkez Bankası faiz kararını açıkladı ve piyasalar bu karara "
        "olumlu tepki verdi. Borsa İstanbul günü yükselişle kapattı. "
        "Dolar kuru ise gün içinde dalgalı bir seyir izledi. "
        "Analistler, önümüzdeki dönemde enflasyonun kontrol altına "
        "alınabileceğini öngörüyor. Hükümet yetkilileri de ekonomik "
        "büyüme hedeflerini yeniden değerlendirdiklerini belirtti. "
        "Uluslararası yatırımcılar Türkiye piyasalarına olan ilgilerini "
        "artırdı."
    )

    for m in ("textrank", "tfidf", "custom"):
        print(f"\n{'='*60}")
        print(f"Yöntem: {m.upper()}")
        print("=" * 60)
        out = summarize(
            sample, n=3, method=m,
            title="Türkiye Ekonomi Gündeminde Merkez Bankası Faiz Kararı",
        )
        if out["error"]:
            print(f"Hata: {out['error']}")
        else:
            print(f"Toplam cümle: {out['sentence_count']}")
            print(f"Sıkıştırma: %{out['compression_ratio']}")
            print(f"Özet:\n{out['summary']}")
