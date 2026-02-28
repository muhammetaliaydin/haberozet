# -*- coding: utf-8 -*-
"""Streamlit arayüzü — HaberÖzet.

Kullanıcının haber URL'si girip TF-IDF, TextRank veya Abstractive
yöntemiyle özet oluşturmasını sağlayan web uygulaması.
"""

import streamlit as st

from preprocessor import setup_nltk
from scraper import fetch_article
from summarizer import summarize

# ── NLTK verilerini bir kez indir ──────────────────────────────────
@st.cache_resource
def _init_nltk() -> None:
    """NLTK verilerini Streamlit context'i içinde bir kez indirir."""
    setup_nltk()

_init_nltk()

# ── Sayfa Ayarları ─────────────────────────────────────────────────
st.set_page_config(
    page_title="HaberÖzet",
    page_icon="📰",
    layout="wide",
)

st.title("📰 HaberÖzet")
st.caption("Türkçe haber makalelerini yapay zekâ destekli NLP ile özetleyin.")

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ayarlar")

    method = st.radio(
        "Özetleme Yöntemi",
        options=["Abstractive (mT5)", "TextRank", "TF-IDF"],
        index=0,
        help="Özet oluşturmak için kullanılacak algoritmayı seçin.",
    )
    method_map = {
        "Abstractive (mT5)": "abstractive",
        "TextRank": "textrank",
        "TF-IDF": "tfidf",
    }
    method_key = method_map[method]

    # Abstractive yöntemde cümle sayısı slider'ı gereksiz
    if method_key != "abstractive":
        n_sentences = st.slider(
            "Özet Cümle Sayısı",
            min_value=1,
            max_value=10,
            value=3,
        )
    else:
        n_sentences = 3  # extractive fallback değeri, abstractive'de kullanılmaz

    st.divider()
    st.subheader("ℹ️ Yöntem Bilgisi")
    st.info(
        "**Abstractive (mT5):** Türkçe haber özetleme için eğitilmiş mT5 "
        "dil modeli kullanır. Orijinal cümleleri seçmek yerine **kendi "
        "özet cümlelerini üretir**. İlk çalıştırmada model indirilir (~2 GB)."
    )
    st.info(
        "**TextRank:** Cümleler arası benzerlik grafi kurar ve PageRank "
        "algoritmasıyla en merkezi cümleleri seçer. Bağlam bütünlüğünü "
        "iyi korur."
    )
    st.info(
        "**TF-IDF:** Her cümlenin terim frekans–ters doküman frekans "
        "skorunu hesaplar ve en bilgi yoğun cümleleri seçer. Hızlı ve "
        "etkilidir."
    )

# ── Cacheleme ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_fetch(url: str) -> dict:
    """Aynı URL için tekrar istek atmamak adına sonucu önbelleğe alır."""
    return fetch_article(url)


@st.cache_data(show_spinner=False)
def cached_summarize(text: str, n: int, method: str) -> dict:
    """Aynı metin/parametre kombinasyonu için sonucu önbelleğe alır."""
    return summarize(text, n=n, method=method)


# ── Ana Alan ───────────────────────────────────────────────────────
url = st.text_input(
    "🔗 Haber URL'si",
    placeholder="https://www.bbc.com/turkce/articles/c0kvl0ry0rdo",
)

summarize_btn = st.button("🔍 Haberi Özetle", type="primary", use_container_width=True)

if summarize_btn:
    if not url or not url.strip():
        st.warning("Lütfen bir URL girin.")
    else:
        try:
            # 1. Haberi çek
            with st.spinner("Haber çekiliyor..."):
                article = cached_fetch(url.strip())

            if article["error"]:
                st.error(f"❌ Haber çekme hatası: {article['error']}")
            else:
                # 2. Özetle
                with st.spinner(
                    "Model yükleniyor ve özetleniyor..."
                    if method_key == "abstractive"
                    else "Özetleniyor..."
                ):
                    result = cached_summarize(
                        article["text"], n_sentences, method_key
                    )

                if result["error"]:
                    st.error(f"❌ Özetleme hatası: {result['error']}")
                else:
                    # ── Sonuç Layout'u ─────────────────────────────
                    st.subheader(article["title"])
                    st.divider()

                    col_left, col_right = st.columns([0.6, 0.4])

                    with col_left:
                        st.markdown("#### 📄 Orijinal Haber")
                        st.text_area(
                            "Orijinal metin",
                            value=article["text"],
                            height=400,
                            disabled=True,
                            label_visibility="collapsed",
                        )

                    with col_right:
                        st.markdown("#### ✨ Özet")
                        st.success(result["summary"])

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Toplam Cümle", result["sentence_count"])
                        m2.metric("Özet Cümle", len(result["sentences"]))
                        m3.metric(
                            "Sıkıştırma Oranı",
                            f"%{result['compression_ratio']}",
                        )

                    st.divider()

                    with st.expander("📋 Özet Cümleleri"):
                        for i, sent in enumerate(result["sentences"], 1):
                            st.markdown(f"**{i}.** {sent}")

        except Exception as exc:
            st.error(f"❌ Beklenmeyen bir hata oluştu: {exc}")
