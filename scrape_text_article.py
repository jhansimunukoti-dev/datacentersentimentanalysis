import argparse
import os
import sys
import time
import tempfile
import shutil

import pandas as pd
import requests
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def load_input(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    tmp = None
    try:
        # Copy to temp to avoid OneDrive lock issues
        tmp = os.path.join(
            tempfile.gettempdir(),
            f"tmp_{int(time.time())}_{os.path.basename(path)}",
        )
        shutil.copy2(path, tmp)

        # ---- 1️⃣ Try real Excel first (.xlsx / .xlsm) ----
        try:
            return pd.read_excel(tmp, engine="openpyxl")
        except Exception:
            pass

        # ---- 2️⃣ Try legacy .xls ----
        try:
            return pd.read_excel(tmp, engine="xlrd")
        except Exception:
            pass

        # ---- 3️⃣ Try xlsb ----
        try:
            return pd.read_excel(tmp, engine="pyxlsb")
        except Exception:
            pass

        # ---- 4️⃣ Try CSV (very common) ----
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
            try:
                return pd.read_csv(tmp, encoding=enc)
            except Exception:
                continue

        # ---- 5️⃣ Try HTML pretending to be Excel (MOST LIKELY) ----
        try:
            tables = pd.read_html(tmp)
            if tables:
                return tables[0]
        except Exception:
            pass

        raise RuntimeError(
            "Input file is not a real Excel file. "
            "It appears to be HTML or CSV saved with .xlsx extension.\n\n"
            "✅ FIX: Open it in Excel → Save As → CSV or XLSX → retry."
        )

    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def scrape_article_text(url: str, session: requests.Session, timeout: int = 15) -> str:
    if not url or not isinstance(url, str):
        return "No URL"

    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        return "Invalid URL"

    try:
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        for t in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
            t.decompose()

        selectors = [
            ("article", None),
            ("main", None),
            ("div", {"class": ["content", "article-body", "post-content", "story-body", "entry-content"]}),
            ("div", {"id": ["content", "article", "main-content"]}),
        ]

        container = None
        for tag, attrs in selectors:
            container = soup.find(tag, attrs=attrs) if attrs else soup.find(tag)
            if container:
                break

        paras = (
            container.find_all("p") if container else soup.find_all("p")
        )
        text = " ".join(p.get_text(" ", strip=True) for p in paras)
        return " ".join(text.split()) or "No content found"

    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="250_datacenter_news_final.xlsx")
    parser.add_argument("-o", "--output", default="250_datacenter_news_updated.xlsx")
    parser.add_argument("-d", "--delay", type=float, default=1.5)
    args = parser.parse_args()

    try:
        df = load_input(args.input)
    except Exception as e:
        print("❌ Failed to load input file:\n", e, file=sys.stderr)
        sys.exit(1)

    if "URL" not in df.columns:
        print("❌ Input must contain a 'URL' column.", file=sys.stderr)
        print("Columns found:", list(df.columns), file=sys.stderr)
        sys.exit(1)

    if "Full Text" not in df.columns:
      df["Full Text"] = ""

    if "Article Summary" not in df.columns:
      df["Article Summary"] = ""


    session = requests.Session()
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {row['URL']}")
        df.at[i, "Full Text"] = scrape_article_text(row["URL"], session)
        time.sleep(args.delay)

    df.to_excel(args.output, index=False, engine="openpyxl")
    print("✅ Saved:", args.output)


if __name__ == "__main__":
    main()
