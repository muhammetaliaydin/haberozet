# -*- coding: utf-8 -*-
"""Haber çekme modülü.

newspaper3k kütüphanesi kullanarak verilen URL'den haber metnini çeker,
parse eder ve temizlenmiş veriyi dict olarak döndürür.
"""

import logging
from newspaper import Article

from preprocessor import clean_article_text

logger = logging.getLogger(__name__)


def fetch_article(url: str) -> dict:
    """Verilen URL'den haber makalesini çeker ve parse eder.

    Args:
        url: Haber makalesinin URL'si.

    Returns:
        Şu anahtarları içeren dict:
            - title (str): Haber başlığı
            - text (str): Temiz haber metni
            - url (str): Orijinal URL
            - error (str | None): Hata varsa mesaj, yoksa None
    """
    result: dict = {
        "title": "",
        "text": "",
        "url": url,
        "error": None,
    }

    try:
        article = Article(url, language="tr")
        article.download()
        article.parse()

        result["title"] = article.title or ""
        result["text"] = clean_article_text(article.text or "", title=result["title"])
        result["url"] = url

        if len(result["text"]) < 100:
            result["error"] = "Yeterli içerik bulunamadı"
            result["title"] = ""
            result["text"] = ""
            return result

    except Exception as exc:
        logger.error("Makale çekilirken hata oluştu: %s", exc)
        result["error"] = f"Haber çekilemedi: {exc}"
        result["title"] = ""
        result["text"] = ""

    return result


if __name__ == "__main__":
    test_url = "https://www.bbc.com/turkce"
    data = fetch_article(test_url)
    if data["error"]:
        print(f"Hata: {data['error']}")
    else:
        print(f"Başlık: {data['title']}")
        print(f"Metin uzunluğu: {len(data['text'])} karakter")
        print(f"İlk 200 karakter: {data['text'][:200]}")
