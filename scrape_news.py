import os
import argparse
import time
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from newspaper import Article
# serpapi package can expose GoogleSearch at the top-level or under
# serpapi.google_search_results depending on the installed distribution.
try:
    from serpapi import GoogleSearch  # preferred
except ImportError:
    # fallback to the module path used by some package versions
    from serpapi.google_search_results import GoogleSearch
import pandas as pd
from dateutil import parser


# Basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CONFIG (can be overridden by command-line args)
SERPAPI_KEY = os.getenv("SERPAPI_KEY") or os.getenv("API_KEY")
QUERY = "data center Georgia"
MAX_ARTICLES = 250
HEADERS = {"User-Agent": "Mozilla/5.0"}
BAD_HOSTS = ["msn.com", "bing.com", "youtube.com", "facebook", "tiktok", "/news/search"]


def is_valid_url(url):
    if not url or not url.startswith("http"):
        return False
    return not any(bad in url for bad in BAD_HOSTS)


def parse_date(text):
    try:
        return parser.parse(text, fuzzy=True).date().isoformat()
    except Exception:
        return None


def extract_article(url):
    """Try `newspaper` first, fall back to requests+BeautifulSoup."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text or ""
        if len(text.strip()) >= 100:
            title = article.title.strip() if article.title else "No Title"
            return title, text.strip()
    except Exception:
        # continue to fallback
        pass

    try:
        resp = requests.get(url, timeout=10, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        full_text = " ".join(p.get_text() for p in paragraphs)
        if len(full_text.strip()) >= 100:
            title = soup.title.string if soup.title and soup.title.string else "No Title"
            return title, full_text.strip()
    except Exception:
        pass

    return None, None


def mock_ai_analysis(text):
    # Simulated analysis used as placeholder for a real AI call
    return {
        "stance_score": 0.1,
        "confidence": 0.5,
        "risk_score": 45,
        "stakeholders": "Unknown",
        "themes": "economic, infrastructure",
        "mobilization": "None",
        "narrative_frame": "economic_injustice",
        "key_quotes": "None",
    }


def fetch_google_news(query, max_results, api_key):
    if not api_key:
        raise ValueError("SERPAPI_KEY (or API_KEY) is not set. Set it in environment variables or .env.")

    articles = []
    # serpapi returns 10 results per page typically; iterate start values
    for start in range(0, max_results, 10):
        params = {
            "q": query,
            "engine": "google",
            "tbm": "nws",
            "start": start,
            "api_key": api_key,
        }
        try:
            search = GoogleSearch(params)
            data = search.get_dict()
        except Exception as e:
            logger.warning("SerpApi query failed at start=%s: %s", start, e)
            break

        for article in data.get("news_results", []):
            link = article.get("link")
            date_str = article.get("date")
            pub_date = parse_date(date_str)
            source = article.get("source")
            articles.append((link, pub_date, source))

        time.sleep(1)  # polite pause

    return articles


def main():
    parser_arg = argparse.ArgumentParser(description="News scraper using SerpApi + newspaper3k")
    parser_arg.add_argument("--query", default=QUERY, help="Search query")
    parser_arg.add_argument("--max", type=int, default=MAX_ARTICLES, help="Maximum number of articles to fetch")
    parser_arg.add_argument("--output", default="output.xlsx", help="Output Excel file path")
    args = parser_arg.parse_args()

    api_key = SERPAPI_KEY
    if not api_key:
        logger.error("SERPAPI_KEY (or API_KEY) not found in environment. See README.md to set it.")
        return

    results = []
    all_links = fetch_google_news(args.query, args.max, api_key)
    seen = set()

    for i, (url, pub_date, source) in enumerate(all_links, 1):
        if not url or url in seen or not is_valid_url(url):
            continue
        seen.add(url)
        logger.info("[%s] Processing: %s", i, url)
        title, full_text = extract_article(url)
        if not title or not full_text:
            logger.info("Skipped (no usable text): %s", url)
            continue

        ai_result = mock_ai_analysis(full_text)

        results.append({
            "Date": pub_date or datetime.now().date().isoformat(),
            "Source": source or "Unknown",
            "Title": title,
            "URL": url,
            "Location": "Georgia",
            "Stance Score": ai_result["stance_score"],
            "Confidence": ai_result["confidence"],
            "Risk Score": ai_result["risk_score"],
            "Stakeholders": ai_result["stakeholders"],
            "Themes": ai_result["themes"],
            "Mobilization": ai_result["mobilization"],
            "Narrative Frame": ai_result["narrative_frame"],
            "Key Quotes": ai_result["key_quotes"],
        })

        if len(results) >= args.max:
            break

        time.sleep(1)

    df = pd.DataFrame(results)
    df.to_excel(args.output, index=False)
    logger.info("Saved %s articles to %s", len(results), args.output)


if __name__ == "__main__":
    main()
