# backend/utils/scraper.py
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from dataclasses import dataclass
from typing import Optional
import logging
import re

logger = logging.getLogger(__name__)

# Simple credibility list — extend with a real database later
CREDIBLE_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "economist.com", "nature.com", "science.org",
}
SUSPICIOUS_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "theonion.com",  # satire — not fake but not real news either
}


@dataclass
class ScrapedArticle:
    url: str
    headline: str
    body: str
    domain: str
    credibility_score: float   # 0=suspicious, 0.5=unknown, 1=credible
    image_url: Optional[str]
    authors: list[str]
    publish_date: Optional[str]


def scrape_article(url: str, timeout: int = 20) -> ScrapedArticle:
    """
    Scrape an article URL using newspaper3k with BeautifulSoup fallback.
    Returns a ScrapedArticle dataclass.
    """
    domain = _extract_domain(url)
    credibility = _score_domain(domain)

    try:
        article = Article(url)
        article.download()
        article.parse()

        headline = article.title or ""
        body = article.text or ""
        image_url = article.top_image or None
        authors = article.authors or []
        pub_date = str(article.publish_date) if article.publish_date else None

        # Fallback if newspaper3k got nothing useful
        if len(body) < 100:
            headline, body, image_url = _bs4_fallback(url, timeout)

    except Exception as e:
        logger.warning(f"newspaper3k failed for {url}: {e}. Trying BS4 fallback.")
        headline, body, image_url = _bs4_fallback(url, timeout)
        authors = []
        pub_date = None

    if not headline and not body:
        raise ValueError(f"Could not extract content from {url}")

    return ScrapedArticle(
        url=url,
        headline=headline.strip(),
        body=body.strip(),
        domain=domain,
        credibility_score=credibility,
        image_url=image_url,
        authors=authors,
        publish_date=pub_date,
    )


def _bs4_fallback(url: str, timeout: int):
    """Minimal BeautifulSoup fallback scraper."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeGuard/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    headline = ""
    h1 = soup.find("h1")
    if h1:
        headline = h1.get_text(strip=True)

    # Grab all paragraph text
    paragraphs = soup.find_all("p")
    body = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 40)

    # First og:image or first <img>
    image_url = None
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        image_url = og_img["content"]
    elif soup.find("img"):
        image_url = soup.find("img").get("src")

    return headline, body, image_url


def _extract_domain(url: str) -> str:
    """Pull bare domain from a URL."""
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1).lower() if match else ""


def _score_domain(domain: str) -> float:
    """Return 0.0 (suspicious), 0.5 (unknown), or 1.0 (credible)."""
    if domain in CREDIBLE_DOMAINS:
        return 1.0
    if domain in SUSPICIOUS_DOMAINS:
        return 0.0
    return 0.5
