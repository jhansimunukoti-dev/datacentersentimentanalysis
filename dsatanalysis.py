import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# OpenAI SDK (official)
from openai import OpenAI


# =========================
# Your SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are analyzing news articles to predict data center opposition risk. Your output will be used to forecast whether public discourse will translate into organized opposition capable of delaying, modifying, or blocking data center development projects.

CORE PRINCIPLE
STANCE ≠ SENTIMENT: Measure position toward data center development, not emotional tone.
"Victory! We stopped the data center!" = STANCE: -1.0 (maximum opposition)
"Sadly, the data center was approved" = STANCE: +0.5 (development proceeding)

INPUT FORMAT
You will receive: date | news_source | article_text

REQUIRED OUTPUT FORMAT
Return a JSON object with ALL fields below. Use null for missing data, never omit fields.
{
 "article_id": "source_date_location",
 "date": "YYYY-MM-DD",
 "source": "exact_source_name",
 "analysis": {
   "stance_score": -1.0 to 1.0,
   "confidence": 0.0 to 1.0,
   "opposition_risk_score": 0 to 100
 },
 "location": {
   "primary": "County/City name",
   "secondary": ["other_locations_mentioned"],
   "geographic_level": "neighborhood|city|county|region|state"
 },
 "stakeholders": [
   {
     "name": "exact_name_or_group",
     "type": "elected_official|organized_group|business|resident|expert",
     "stance": -1.0 to 1.0,
     "influence_weight": 1.0 to 1.8,
     "quote": "exact_quote_showing_stance"
   }
 ],
 "themes": {
   "environmental": 0 to 100,
   "economic": 0 to 100,
   "infrastructure": 0 to 100,
   "land_use": 0 to 100,
   "trust": 0 to 100,
   "cultural": 0 to 100
 },
 "mobilization": {
   "indicators_present": [
     "group_formation|petition|protest|legal_action|political_pressure|packed_meeting"
   ],
   "mobilization_score": 0 to 30,
   "specific_actions": ["exact_descriptions_from_article"]
 },
 "narrative": {
   "primary_frame": "environmental_crisis|democratic_deficit|david_goliath|cultural_preservation|economic_injustice|infrastructure_overload",
   "key_messages": ["exact_phrases_used"],
   "slogan_if_any": "exact_slogan_or_null"
 },
 "risk_factors": {
   "elected_opposition": true|false,
   "multi_stakeholder_coalition": true|false,
   "reference_other_success": true|false,
   "specific_demands": true|false,
   "deadline_mentioned": "date_or_null",
   "media_amplification": true|false
 },
 "key_quotes": {
   "strongest_opposition": "exact_quote|null",
   "strongest_support": "exact_quote|null",
   "most_mobilizing": "exact_quote|null"
 }
}
"""

# =========================
# Keyword rules (your spec)
# =========================
STRONG_NEGATIVE = ["stop", "fight", "block", "oppose", "moratorium", "destroy", "catastrophic", "disaster"]
MODERATE_NEGATIVE = ["concern", "worry", "question", "impact", "burden", "strain", "problem"]
NEUTRAL = ["discuss", "consider", "review", "study", "examine", "debate"]
MODERATE_POSITIVE = ["opportunity", "benefit", "potential", "growth", "jobs", "investment"]
STRONG_POSITIVE = ["need", "welcome", "transform", "essential", "critical", "support fully"]

ACTION_WORDS = ["organize", "petition", "sue", "protest", "rally", "mobilize"]
SUCCESS_WORDS = ["stopped", "blocked", "defeated", "victory", "won"]

THEME_KEYWORDS = {
    "environmental": ["water", "drought", "emissions", "pollution", "sustainable", "climate", "ecosystem"],
    "economic": ["jobs", "employment", "tax", "revenue", "investment", "growth", "economy"],
    "infrastructure": ["traffic", "roads", "power", "grid", "utilities", "capacity", "strain"],
    "land_use": ["farmland", "rural", "zoning", "agricultural", "development", "preservation"],
    "trust": ["transparency", "process", "community input", "backroom", "rushed", "excluded"],
    "cultural": ["character", "way of life", "heritage", "tradition", "community feel", "identity"],
}

STAKEHOLDER_WEIGHTS = {
    "elected_official": 1.8,
    "organized_group": 1.5,
    "business": 1.3,
    "resident": 1.0,
    "expert": 1.0,
}

MOB_POINTS = {
    "group_formation": 3,
    "petition": 2,
    "protest": 3,
    "legal_action": 5,  # filed; we’ll downshift if only threatened
    "political_pressure": 3,
    "packed_meeting": 2,
    "media_coordination": 2,
    "multi_county": 3,
}


# =========================
# Helpers
# =========================
def read_csv_fallback(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="cp1252", encoding_errors="replace")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def normalize_theme_percentages(counts: Dict[str, int]) -> Dict[str, int]:
    keys = ["environmental", "economic", "infrastructure", "land_use", "trust", "cultural"]
    total = sum(max(0, counts.get(k, 0)) for k in keys)

    if total == 0:
        # Default per spec: economic 40, infrastructure 30, other 30 split
        return {"environmental": 10, "economic": 40, "infrastructure": 30, "land_use": 10, "trust": 5, "cultural": 5}

    # Convert to int percentages
    pct = {}
    for k in keys:
        pct[k] = int(counts.get(k, 0) / total * 100)
    # Fix rounding to sum to 100
    pct["economic"] += 100 - sum(pct.values())
    return pct


def split_sentences(text: str) -> List[str]:
    # Simple sentence split; good enough for rule scoring
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def contains_any(sentence: str, keywords: List[str]) -> bool:
    s = sentence.lower()
    return any(k in s for k in keywords)


def compute_base_stance(article_text: str) -> float:
    sentences = split_sentences(article_text)
    if not sentences:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for sent in sentences:
        s = sent.lower()

        if contains_any(s, STRONG_NEGATIVE):
            score = -0.9
            weight = 1.5
        elif contains_any(s, MODERATE_NEGATIVE):
            score = -0.5
            weight = 1.0
        elif contains_any(s, MODERATE_POSITIVE):
            score = 0.5
            weight = 1.0
        elif contains_any(s, STRONG_POSITIVE):
            score = 0.9
            weight = 1.5
        else:
            score = 0.0
            weight = 0.5

        # Amplify if action words present
        if contains_any(s, ACTION_WORDS):
            weight *= 1.3

        # Invert if success against DC (and DC mentioned)
        if contains_any(s, SUCCESS_WORDS) and contains_any(s, ["data center", "datacenter", "development", "project"]):
            score *= -1

        weighted_sum += score * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight else 0.0


def highest_opposing_weight(stakeholders: List[Dict[str, Any]]) -> float:
    # Only consider opposing stakeholders (stance < 0)
    max_w = 1.0
    for sh in stakeholders or []:
        st = safe_float(sh.get("stance", 0.0), 0.0)
        if st < 0:
            typ = (sh.get("type") or "resident").strip()
            w = STAKEHOLDER_WEIGHTS.get(typ, 1.0)
            # If model provided influence_weight, respect it within 1.0..1.8
            iw = safe_float(sh.get("influence_weight", w), w)
            iw = clamp(iw, 1.0, 1.8)
            max_w = max(max_w, iw)
    return max_w


def is_opinion_piece(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["op-ed", "opinion", "editorial", "column:", "guest column"])


def has_mixed_signals(article_text: str) -> bool:
    t = article_text.lower()
    neg = any(k in t for k in STRONG_NEGATIVE + MODERATE_NEGATIVE)
    pos = any(k in t for k in STRONG_POSITIVE + MODERATE_POSITIVE)
    return neg and pos


def mostly_future_conditional(article_text: str) -> bool:
    t = article_text.lower()
    # rough heuristics
    return sum(t.count(w) for w in ["could", "may", "might", "would", "plans to", "proposed", "proposal"]) > 8


def compute_confidence(article_date: str, stakeholders: List[Dict[str, Any]], article_text: str) -> float:
    confidence = 0.5

    # +0.1 each stakeholder with clear stance
    clear = 0
    for sh in stakeholders or []:
        if sh.get("quote") and sh.get("stance") is not None:
            clear += 1
    confidence += 0.1 * min(clear, 5)  # cap contribution

    # +0.2 if outcome explicitly stated (approved/denied/voted)
    t = article_text.lower()
    if any(k in t for k in ["approved", "denied", "rejected", "voted", "passed", "blocked", "stopped"]):
        confidence += 0.2

    # +0.1 if multiple consistent indicators (many neg or many pos)
    neg_hits = sum(1 for k in (STRONG_NEGATIVE + MODERATE_NEGATIVE) if k in t)
    pos_hits = sum(1 for k in (STRONG_POSITIVE + MODERATE_POSITIVE) if k in t)
    if max(neg_hits, pos_hits) >= 5 and min(neg_hits, pos_hits) == 0:
        confidence += 0.1

    # +0.1 if recent event (within 7 days)
    try:
        d = datetime.fromisoformat(article_date)
        now = datetime.now()
        if abs((now - d).days) <= 7:
            confidence += 0.1
    except Exception:
        pass

    # Decrease:
    if has_mixed_signals(article_text):
        confidence -= 0.2
    if mostly_future_conditional(article_text):
        confidence -= 0.1
    if is_opinion_piece(article_text):
        confidence -= 0.2

    return clamp(confidence, 0.0, 1.0)


def theme_counts_from_text(article_text: str) -> Dict[str, int]:
    t = article_text.lower()
    counts = {}
    for theme, kws in THEME_KEYWORDS.items():
        counts[theme] = sum(t.count(k.lower()) for k in kws)
    return counts


def extract_number_near(text: str, keyword: str, window: int = 50) -> Optional[int]:
    # Find a number near a keyword, e.g. "petition ... 500 signatures"
    t = text.lower()
    idx = t.find(keyword)
    if idx == -1:
        return None
    snippet = t[idx : idx + window]
    m = re.search(r"(\d{1,6})", snippet)
    return int(m.group(1)) if m else None


def compute_mobilization(article_text: str, stakeholders: List[Dict[str, Any]]) -> Tuple[List[str], int, List[str]]:
    t = article_text.lower()
    indicators: List[str] = []
    actions: List[str] = []
    score = 0

    # group_formation
    group_mentions = re.findall(r"(group|coalition|alliance|organization|committee)", t)
    if group_mentions:
        indicators.append("group_formation")
        pts = MOB_POINTS["group_formation"] * min(len(group_mentions), 3)
        score += pts
        actions.append("Group formation mentioned.")

    # petition
    if "petition" in t:
        indicators.append("petition")
        score += MOB_POINTS["petition"]
        sig = extract_number_near(article_text, "petition", 120) or extract_number_near(article_text, "signatures", 80)
        if sig:
            score += int(sig / 100)  # +1 per 100 signatures
            actions.append(f"Petition mentioned with ~{sig} signatures.")
        else:
            actions.append("Petition mentioned (no signature count).")

    # protest
    if any(k in t for k in ["protest", "rally", "march", "demonstration"]):
        indicators.append("protest")
        score += MOB_POINTS["protest"]
        attendees = extract_number_near(article_text, "attendees", 80) or extract_number_near(article_text, "people", 80)
        if attendees:
            score += int(attendees / 50)
            actions.append(f"Protest/rally mentioned with ~{attendees} attendees.")
        else:
            actions.append("Protest/rally mentioned (no attendance count).")

    # legal_action
    if any(k in t for k in ["lawsuit", "sued", "filed suit", "court filing"]):
        indicators.append("legal_action")
        score += MOB_POINTS["legal_action"]
        actions.append("Legal action filed (lawsuit/court filing).")
    elif any(k in t for k in ["threatened lawsuit", "considering lawsuit", "may sue"]):
        indicators.append("legal_action")
        score += 3
        actions.append("Legal action threatened (not filed).")

    # political_pressure
    if any(k in t for k in ["commissioner", "mayor", "council", "senator", "representative", "county board"]):
        # only count as political_pressure if opposed language present nearby
        if any(k in t for k in ["oppose", "opposed", "called for", "demanded", "urged"]):
            indicators.append("political_pressure")
            # +3 per elected official opposed — approximate by counting office keywords
            office_hits = sum(t.count(k) for k in ["commissioner", "mayor", "council", "senator", "representative"])
            score += MOB_POINTS["political_pressure"] * max(1, min(office_hits, 3))
            actions.append("Political pressure via officials mentioned.")

    # packed_meeting
    if any(k in t for k in ["packed meeting", "crowded meeting", "standing room", "overflow crowd"]):
        indicators.append("packed_meeting")
        score += 4
        actions.append("Packed meeting described (standing-room/overflow).")
    else:
        attendees = extract_number_near(article_text, "meeting", 120)
        if attendees and attendees > 50:
            indicators.append("packed_meeting")
            score += 2 if attendees <= 200 else 4
            actions.append(f"Meeting attendance mentioned (~{attendees}).")

    # media_coordination
    if any(k in t for k in ["press conference", "press release", "media statement"]):
        indicators.append("media_coordination")
        score += MOB_POINTS["media_coordination"]
        actions.append("Media coordination mentioned (press release/conference).")

    # multi_county
    if "multi-county" in t or "multiple counties" in t or "regional coalition" in t:
        indicators.append("multi_county")
        score += MOB_POINTS["multi_county"]
        actions.append("Multi-county coordination mentioned.")

    score = int(clamp(score, 0, 30))
    indicators = sorted(set(indicators))

    if not actions:
        actions = ["No mobilization indicators found in text."]

    return indicators, score, actions


def mobilization_multiplier(mobilization_score: int) -> float:
    if mobilization_score > 20:
        return 1.8
    if mobilization_score > 15:
        return 1.6
    if mobilization_score > 10:
        return 1.4
    if mobilization_score > 5:
        return 1.2
    return 1.0


def compute_risk(stance_score: float, confidence: float, mob_score: int) -> int:
    base_risk = 50 - (stance_score * 50)
    risk = base_risk * mobilization_multiplier(mob_score) * confidence
    risk = clamp(risk, 0.0, 100.0)
    return int(round(risk))


def ensure_minimum_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Fill missing top-level keys with null-ish defaults, never omit
    def dget(d: Dict[str, Any], k: str, default):
        return d[k] if k in d else default

    obj = obj or {}

    obj["article_id"] = dget(obj, "article_id", None)
    obj["date"] = dget(obj, "date", None)
    obj["source"] = dget(obj, "source", None)

    obj["analysis"] = dget(obj, "analysis", {}) or {}
    obj["analysis"]["stance_score"] = obj["analysis"].get("stance_score", None)
    obj["analysis"]["confidence"] = obj["analysis"].get("confidence", None)
    obj["analysis"]["opposition_risk_score"] = obj["analysis"].get("opposition_risk_score", None)

    obj["location"] = dget(obj, "location", {}) or {}
    obj["location"]["primary"] = obj["location"].get("primary", None)
    obj["location"]["secondary"] = obj["location"].get("secondary", []) or []
    obj["location"]["geographic_level"] = obj["location"].get("geographic_level", None)

    obj["stakeholders"] = dget(obj, "stakeholders", []) or []
    if not obj["stakeholders"]:
        obj["stakeholders"] = [{
            "name": "Reporter",
            "type": "expert",
            "stance": 0.0,
            "influence_weight": 1.0,
            "quote": None,
        }]

    # Themes
    obj["themes"] = dget(obj, "themes", {}) or {}
    for k in THEME_KEYWORDS.keys():
        obj["themes"][k] = int(obj["themes"].get(k, 0) or 0)

    # Mobilization
    obj["mobilization"] = dget(obj, "mobilization", {}) or {}
    obj["mobilization"]["indicators_present"] = obj["mobilization"].get("indicators_present", []) or []
    obj["mobilization"]["mobilization_score"] = int(obj["mobilization"].get("mobilization_score", 0) or 0)
    obj["mobilization"]["specific_actions"] = obj["mobilization"].get("specific_actions", []) or []

    # Narrative
    obj["narrative"] = dget(obj, "narrative", {}) or {}
    obj["narrative"]["primary_frame"] = obj["narrative"].get("primary_frame", None)
    obj["narrative"]["key_messages"] = obj["narrative"].get("key_messages", []) or []
    obj["narrative"]["slogan_if_any"] = obj["narrative"].get("slogan_if_any", None)

    # Risk factors
    obj["risk_factors"] = dget(obj, "risk_factors", {}) or {}
    for k in ["elected_opposition", "multi_stakeholder_coalition", "reference_other_success", "specific_demands", "deadline_mentioned", "media_amplification"]:
        obj["risk_factors"][k] = obj["risk_factors"].get(k, None)

    # Key quotes
    obj["key_quotes"] = dget(obj, "key_quotes", {}) or {}
    for k in ["strongest_opposition", "strongest_support", "most_mobilizing"]:
        obj["key_quotes"][k] = obj["key_quotes"].get(k, None)

    return obj


def safe_json_load(text: str) -> Dict[str, Any]:
    # Extract JSON if model adds extra text
    text = text.strip()
    # Try direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("Model did not return valid JSON.")


def call_model(client: OpenAI, model: str, date: str, source: str, article_text: str) -> Dict[str, Any]:
    user_msg = f"{date} | {source} | {article_text}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content or ""
    return safe_json_load(content)


def finalize_with_rule_scores(obj: Dict[str, Any], article_date: str, article_text: str) -> Dict[str, Any]:
    obj = ensure_minimum_schema(obj)

    # Ensure date format
    try:
        d = pd.to_datetime(article_date)
        obj["date"] = d.strftime("%Y-%m-%d")
    except Exception:
        obj["date"] = "2025-01-01"

    # Themes by rule (override to make deterministic + sum=100)
    theme_counts = theme_counts_from_text(article_text)
    obj["themes"] = normalize_theme_percentages(theme_counts)

    # Mobilization by rule (override)
    indicators, mob_score, actions = compute_mobilization(article_text, obj["stakeholders"])
    obj["mobilization"]["indicators_present"] = indicators
    obj["mobilization"]["mobilization_score"] = mob_score
    # Keep model actions if present, but ensure we have something and include errors if needed
    if not obj["mobilization"]["specific_actions"]:
        obj["mobilization"]["specific_actions"] = actions

    # Stance by rule
    base = compute_base_stance(article_text)
    sh_weight = highest_opposing_weight(obj["stakeholders"])
    stance = base * sh_weight
    stance = clamp(stance, -1.0, 1.0)

    # Special case: No clear stance indicators
    t = article_text.lower()
    has_any = (
        any(k in t for k in STRONG_NEGATIVE + MODERATE_NEGATIVE + STRONG_POSITIVE + MODERATE_POSITIVE)
        and any(k in t for k in ["data center", "datacenter", "data centres", "development", "project"])
    )
    if not has_any:
        stance = 0.0
        conf = 0.3
    else:
        # Confidence by rule
        conf = compute_confidence(obj["date"], obj["stakeholders"], article_text)

    # Risk by rule
    risk = compute_risk(stance, conf, mob_score)

    obj["analysis"]["stance_score"] = round(float(stance), 2)
    obj["analysis"]["confidence"] = round(float(conf), 2)
    obj["analysis"]["opposition_risk_score"] = int(risk)

    # Ensure article_id if missing
    if not obj.get("article_id"):
        loc = obj.get("location", {}).get("primary") or "unknown_location"
        src = obj.get("source") or "unknown_source"
        obj["article_id"] = f"{src}_{obj['date']}_{loc}".replace(" ", "_")

    # Ensure source
    if not obj.get("source"):
        obj["source"] = "Unknown"

    # Final validation clamps
    obj["analysis"]["stance_score"] = clamp(safe_float(obj["analysis"]["stance_score"]), -1.0, 1.0)
    obj["analysis"]["confidence"] = clamp(safe_float(obj["analysis"]["confidence"]), 0.0, 1.0)
    obj["analysis"]["opposition_risk_score"] = int(clamp(safe_float(obj["analysis"]["opposition_risk_score"]), 0, 100))

    # Themes sum check
    s = sum(int(obj["themes"][k]) for k in obj["themes"])
    if s != 100:
        obj["themes"]["economic"] += (100 - s)

    # At least 1 stakeholder
    if not obj["stakeholders"]:
        obj["stakeholders"] = [{
            "name": "Reporter",
            "type": "expert",
            "stance": 0.0,
            "influence_weight": 1.0,
            "quote": None,
        }]

    return obj


# =========================
# Main pipeline
# =========================
def main():
    parser = argparse.ArgumentParser(description="DCSAT LLM extraction + deterministic scoring.")
    parser.add_argument("-i", "--input", default="250_datacenter_news_updated.csv", help="Input CSV file")
    parser.add_argument("-o", "--output", default="250_datacenter_news_with_analysis.csv", help="Output CSV file")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model name for OpenAI")
    parser.add_argument("--date_col", default="Date", help="Column name for date")
    parser.add_argument("--source_col", default="Source", help="Column name for source")
    parser.add_argument("--text_col", default="Full Text", help="Column name for article text")
    parser.add_argument("--max_rows", type=int, default=0, help="0 = all rows, else process first N rows")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var not set.", file=os.sys.stderr)
        os.sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = read_csv_fallback(args.input)
    df.rename(columns=lambda c: str(c).strip(), inplace=True)

    # Ensure expected cols exist
    for col in [args.date_col, args.source_col, args.text_col]:
        if col not in df.columns:
            df[col] = ""

    out_rows = []
    n = len(df) if args.max_rows == 0 else min(len(df), args.max_rows)

    for idx in range(n):
        row = df.iloc[idx]
        date_raw = str(row.get(args.date_col, "") or "").strip()
        source = str(row.get(args.source_col, "") or "Unknown").strip()
        text = str(row.get(args.text_col, "") or "").strip()

        # Normalize date
        try:
            date_norm = pd.to_datetime(date_raw).strftime("%Y-%m-%d")
        except Exception:
            date_norm = "2025-01-01"

        print(f"[{idx+1}/{n}] {source} | {date_norm}")

        try:
            extracted = call_model(client, args.model, date_norm, source, text)
        except Exception as e:
            extracted = {
                "article_id": None,
                "date": date_norm,
                "source": source,
                "analysis": {"stance_score": None, "confidence": None, "opposition_risk_score": None},
                "location": {"primary": None, "secondary": [], "geographic_level": None},
                "stakeholders": [],
                "themes": {k: 0 for k in THEME_KEYWORDS.keys()},
                "mobilization": {"indicators_present": [], "mobilization_score": 0, "specific_actions": [f"ERROR: Model call failed: {e}"]},
                "narrative": {"primary_frame": None, "key_messages": [], "slogan_if_any": None},
                "risk_factors": {
                    "elected_opposition": None,
                    "multi_stakeholder_coalition": None,
                    "reference_other_success": None,
                    "specific_demands": None,
                    "deadline_mentioned": None,
                    "media_amplification": None,
                },
                "key_quotes": {"strongest_opposition": None, "strongest_support": None, "most_mobilizing": None},
            }

        final = finalize_with_rule_scores(extracted, date_norm, text)

        out_rows.append({
            **row.to_dict(),
            "DCSAT_JSON": json.dumps(final, ensure_ascii=False),
            "Stance Score": final["analysis"]["stance_score"],
            "Confidence": final["analysis"]["confidence"],
            "Risk Score": final["analysis"]["opposition_risk_score"],
            "Location": json.dumps(final["location"], ensure_ascii=False),
            "Stakeholders": json.dumps(final["stakeholders"], ensure_ascii=False),
            "Themes": json.dumps(final["themes"], ensure_ascii=False),
            "Mobilization": json.dumps(final["mobilization"], ensure_ascii=False),
            "Narrative": json.dumps(final["narrative"], ensure_ascii=False),
            "Risk Factors": json.dumps(final["risk_factors"], ensure_ascii=False),
            "Key Quotes": json.dumps(final["key_quotes"], ensure_ascii=False),
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {args.output}")


if __name__ == "__main__":
    main()
