import argparse
import json
from random import uniform, randint, choice

import pandas as pd


def read_csv_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort: keep going but replace un-decodable chars
    return pd.read_csv(path, encoding="cp1252", encoding_errors="replace")


def generate_structured_analysis(article_id, date, source, text):
    stance_score = round(uniform(-1.0, 1.0), 2)
    confidence = round(min(1.0, max(0.0, 0.5 + 0.1 * randint(-2, 4))), 2)
    mobilization_score = randint(0, 30)

    risk_score = int(
        max(
            0,
            min(
                100,
                (50 - (stance_score * 50)) * (1 + 0.02 * mobilization_score) * confidence,
            ),
        )
    )

    themes = {
        "environmental": randint(0, 40),
        "economic": randint(0, 40),
        "infrastructure": randint(0, 40),
        "land_use": randint(0, 40),
        "trust": randint(0, 40),
        "cultural": randint(0, 40),
    }

    total = sum(themes.values()) or 1
    for k in themes:
        themes[k] = int(themes[k] / total * 100)
    themes["economic"] += 100 - sum(themes.values())

    return {
        "analysis": {
            "stance_score": stance_score,
            "confidence": confidence,
            "opposition_risk_score": risk_score,
        },
        "location": {
            "primary": "Georgia",
            "secondary": ["Atlanta"],
            "geographic_level": "state",
        },
        "stakeholders": [
            {
                "name": "Reporter",
                "type": "expert",
                "stance": stance_score,
                "influence_weight": 1.0,
                "quote": "Simulated quote on the issue.",
            }
        ],
        "themes": themes,
        "mobilization": {
            "indicators_present": ["petition"],
            "mobilization_score": mobilization_score,
            "specific_actions": ["Simulated petition with 500 signatures"],
        },
        "narrative": {
            "primary_frame": choice(
                ["environmental_crisis", "democratic_deficit", "economic_injustice"]
            ),
            "key_messages": ["Data center impact discussed"],
            "slogan_if_any": None,
        },
        "key_quotes": {
            "strongest_opposition": "We will fight this data center until the end.",
            "strongest_support": "The data center brings much needed jobs.",
            "most_mobilizing": "Over 500 residents signed the petition.",
        },
    }


def apply_analysis(row, index):
    date_val = row.get("Date", "")
    source_val = row.get("Source", "Unknown")
    text_val = row.get("Full Text", "")

    date = "2025-01-01"
    if pd.notnull(date_val) and str(date_val).strip():
        try:
            date = pd.to_datetime(date_val).strftime("%Y-%m-%d")
        except Exception:
            pass

    source = str(source_val).strip() if pd.notnull(source_val) and str(source_val).strip() else "Unknown"
    text = str(text_val) if pd.notnull(text_val) else ""

    structured = generate_structured_analysis(index, date, source, text)

    return pd.Series(
        {
            "Location": json.dumps(structured["location"], ensure_ascii=False),
            "Stance Score": structured["analysis"]["stance_score"],
            "Confidence": structured["analysis"]["confidence"],
            "Risk Score": structured["analysis"]["opposition_risk_score"],
            "Stakeholders": json.dumps(structured["stakeholders"], ensure_ascii=False),
            "Themes": json.dumps(structured["themes"], ensure_ascii=False),
            "Mobilization": json.dumps(structured["mobilization"], ensure_ascii=False),
            "Narrative Frame": json.dumps(structured["narrative"], ensure_ascii=False),
            "Key Quotes": json.dumps(structured["key_quotes"], ensure_ascii=False),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Add simulated analysis columns to a CSV.")
    parser.add_argument("-i", "--input", default="250_datacenter_news_updated.csv")
    parser.add_argument("-o", "--output", default="250_datacenter_news_with_analysis.csv")
    args = parser.parse_args()

    print("📥 Loading CSV (with encoding fallback)...")
    df = read_csv_fallback(args.input)
    df.rename(columns=lambda c: str(c).strip(), inplace=True)
    print("✅ Columns detected:", list(df.columns))

    for col, default in [("Date", ""), ("Source", "Unknown"), ("Full Text", "")]:
        if col not in df.columns:
            df[col] = default

    print("🧠 Running analysis...")
    results = df.apply(lambda r: apply_analysis(r, r.name), axis=1)

    print("📊 Merging results...")
    for col in results.columns:
        df[col] = results[col]

    print(f"💾 Saving to {args.output} ...")
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("✅ Done.")


if __name__ == "__main__":
    main()
