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


def summarize(
    text: str,
    n: int = 3,
    method: str = "textrank",
) -> dict:
    """Metni seçilen yöntemle özetler.

    Args:
        text: Özetlenecek ham metin.
        n: İstenen özet cümle sayısı (extractive yöntemler için).
        method: Özetleme yöntemi — "textrank", "tfidf" veya "abstractive".

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

    for m in ("textrank", "tfidf"):
        print(f"\n{'='*60}")
        print(f"Yöntem: {m.upper()}")
        print("=" * 60)
        out = summarize(sample, n=3, method=m)
        if out["error"]:
            print(f"Hata: {out['error']}")
        else:
            print(f"Toplam cümle: {out['sentence_count']}")
            print(f"Sıkıştırma: %{out['compression_ratio']}")
            print(f"Özet:\n{out['summary']}")
