"""
Climatrack core logic — no Streamlit. Used by Flask app.
"""
from functools import lru_cache
import time
import re
import json
import logging
import os
import csv
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from html import unescape, escape
from urllib.parse import quote_plus, urlparse
import requests

# Optional heavy deps
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

import xml.etree.ElementTree as ET

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
NEWS_TTL_SEC = 180
DEBUG_AQI = False
LBS_PER_KG = 2.20462262
UNITS_DEFAULT = "Imperial (lbs CO₂)"
_ROOT = Path(__file__).resolve().parent
DATA_DIR = _ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "footprint_log.csv"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# TTL cache (replaces st.cache_data(ttl=...))
# ---------------------------------------------------------
_ttl_cache: Dict = {}
def ttl_cache(ttl_sec: int):
    def dec(f):
        def wrap(*args, **kwargs):
            key = (f.__name__, args, tuple(sorted((k, v) for k, v in kwargs.items())))
            now = time.time()
            if key in _ttl_cache:
                val, ts = _ttl_cache[key]
                if now - ts < ttl_sec:
                    return val
            val = f(*args, **kwargs)
            _ttl_cache[key] = (val, now)
            return val
        return wrap
    return dec

# ---------------------------------------------------------
# Units
# ---------------------------------------------------------
def kg_to_lbs(x: float) -> float:
    return float(x) * LBS_PER_KG

def convert_mass_value(value_kg: float, units_label: str = UNITS_DEFAULT) -> float:
    return kg_to_lbs(value_kg)

def unit_suffix(units_label: str = UNITS_DEFAULT) -> str:
    return "lb CO₂"

# ---------------------------------------------------------
# Models (Ollama → fallback to FLAN-T5)
# ---------------------------------------------------------
@lru_cache(maxsize=1)
def load_models():
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers/torch not installed")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def have_ollama():
    try:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        r = requests.get(f"{host}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def ollama_generate(prompt, model=None, timeout=30):
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    resp = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"]

def llm_complete(prompt: str, max_new_tokens=300, temp=0.3) -> str:
    if have_ollama():
        return ollama_generate(prompt, timeout=60)
    tokenizer, model = load_models()
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temp, do_sample=temp > 0.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# News utilities
# ---------------------------------------------------------
def _time_window_to_timespan(window: str) -> str:
    return {"24h": "1d", "7d": "7d", "30d": "30d"}.get(window, "7d")

def _time_window_to_google_when(window: str) -> str:
    return {"24h": "when:1d", "7d": "when:7d", "30d": "when:30d"}.get(window, "when:7d")

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""

def _fmt_time(ts: datetime) -> str:
    try:
        delta = datetime.utcnow() - ts
        if delta.days >= 1:
            return f"{delta.days}d ago"
        hours = int(delta.total_seconds() // 3600)
        if hours >= 1:
            return f"{hours}h ago"
        minutes = int(delta.total_seconds() // 60)
        return f"{minutes}m ago"
    except Exception:
        return ""

def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _clamp(text: str, n: int = 280) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"

def _is_english_text(text: str) -> bool:
    if not (text or "").strip():
        return False
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return True
    return all("a" <= c <= "z" or "A" <= c <= "Z" for c in letters)

def _normalize_items(items: List[Dict]) -> List[Dict]:
    norm = []
    for it in items:
        title = unescape(it.get("title", "")).strip()
        url = (it.get("url") or "").strip()
        if not title or not url:
            continue
        published = it.get("published")
        if isinstance(published, str):
            for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%SZ", "%Y%m%d%H%M%S"):
                try:
                    published = datetime.strptime(published, fmt)
                    break
                except Exception:
                    pass
            if isinstance(published, str):
                published = None
        raw_summary = it.get("summary") or it.get("description") or ""
        summary = _clamp(_strip_html(raw_summary), 280)
        norm.append({"title": title, "url": url, "source": it.get("source") or _domain(url), "summary": summary, "published": published})
    seen, out = set(), []
    for x in norm:
        if x["url"] in seen:
            continue
        seen.add(x["url"])
        out.append(x)
    return out

@ttl_cache(NEWS_TTL_SEC)
def fetch_google_news(query: str, window: str, limit: int = 24, fresh: int = 0) -> List[Dict]:
    q = quote_plus(f"(climate OR environment) {query}".strip())
    feed_url = f"https://news.google.com/rss/search?q={q}+{_time_window_to_google_when(window)}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(feed_url, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.findall(".//item"):
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub = item.findtext("pubDate") or ""
            desc = item.findtext("description") or ""
            source_tag = item.find("{http://news.google.com}source") or item.find("{http://www.w3.org/2005/Atom}source")
            src = (source_tag.text if source_tag is not None else "")
            items.append({"title": title, "url": link, "source": src, "published": pub, "description": desc})
        return _normalize_items(items)[:limit]
    except Exception as e:
        logger.info(f"Google News RSS fetch failed: {e}")
        return []

def get_climate_news(query: str, window: str = "7d", limit: int = 24, source_pref: str = "all", fresh: int = 0) -> List[Dict]:
    """Fetch climate/environment news from Google News only (English titles)."""
    query = (query or "").strip()
    items: List[Dict] = []
    fetch_limit = min(limit * 3, 48) if limit else 48
    try:
        items += fetch_google_news(query, window, fetch_limit, fresh=fresh)
    except Exception as e:
        logger.info(f"News fetch error: {e}")
    def _sort_key(x):
        ts = x.get("published")
        if isinstance(ts, datetime):
            return (0, ts)
        return (1, datetime.min)
    items = _normalize_items(items)
    items = [it for it in items if _is_english_text(it.get("title") or "")]
    items.sort(key=_sort_key, reverse=True)
    seen = set()
    out = []
    for it in items:
        if it["url"] in seen:
            continue
        seen.add(it["url"])
        out.append(it)
        if len(out) >= limit:
            break
    return out

# Typical daily footprint (lb CO₂) for comparison — e.g. ~16 kg
TYPICAL_DAILY_LBS = round(kg_to_lbs(16.0), 1)

# ---------------------------------------------------------
# Emission factors (real-life, cited)
# ---------------------------------------------------------
# Transport: EPA "Greenhouse Gas Emissions from a Typical Passenger Vehicle" (8,887 g CO2/gal gasoline, avg fuel economy) → ~0.404 kg CO2/mile. km = mile/1.609.
KG_CO2_PER_MILE_CAR = 0.404
KG_CO2_PER_KM_CAR = 0.404 / 1.609  # ~0.251
# Electricity: US average from EPA eGRID (2023) ~0.348 kg/kWh; use 0.35 for calculator.
KG_CO2_PER_KWH_US = 0.35
# Food: kg CO2e per typical serving (~150 g protein). Sources: Our World in Data (Poore & Nemecek), EPA. Beef ~27–43 kg/kg → ~6 kg/serving; chicken ~5–7 kg/kg → ~1 kg/serving; pork ~7–10 kg/kg → ~1.2.
KG_CO2_BEEF_SERVING = 6.0
KG_CO2_CHICKEN_SERVING = 1.0
KG_CO2_PORK_SERVING = 1.2
KG_CO2_MEAT_AVG_SERVING = 2.5  # generic "meat" when type unknown
# Default when user mentions electricity/AC/heating but no kWh: assume ~4 kWh (e.g. AC or heating for part of day).
DEFAULT_ELECTRICITY_KWH = 4.0

# ---------------------------------------------------------
# News: fetch full article text (so agents can read entire article)
# ---------------------------------------------------------
ARTICLE_FETCH_TIMEOUT = 15
ARTICLE_MAX_CHARS = 5500  # max article text passed to LLM (to fit context)

def fetch_article_text(url: str, max_chars: int = ARTICLE_MAX_CHARS) -> str:
    """Fetch URL and extract main article text. Returns plain text (empty on failure)."""
    if not (url or "").strip():
        return ""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        return ""
    text = ""
    try:
        from trafilatura import fetch_url, extract
        downloaded = fetch_url(url, timeout=ARTICLE_FETCH_TIMEOUT)
        if downloaded:
            text = extract(downloaded, include_comments=False, include_tables=False) or ""
    except Exception as e:
        logger.debug(f"trafilatura fetch/extract failed for {url[:60]}: {e}")
    if not text or len(text) < 100:
        try:
            r = requests.get(url, timeout=ARTICLE_FETCH_TIMEOUT, headers={"User-Agent": "Climatrack/1.0 (climate news reader)"})
            r.raise_for_status()
            raw = _strip_html(r.text)
            if raw and len(raw) > 200:
                text = raw[:max_chars * 2]  # allow more before final clamp
        except Exception as e:
            logger.debug(f"requests fallback failed for {url[:60]}: {e}")
    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text

# ---------------------------------------------------------
# News: AI agent opinions (Moltbook-style) — agents read full article when available
# ---------------------------------------------------------
NEWS_AGENTS = [
    {"name": "Climate scientist", "persona": "You are a climate scientist talking to Gen‑Z. Use simple, clear language, short sentences, and no heavy jargon. Explain what the article means for the planet in a friendly way."},
    {"name": "Policy advocate", "persona": "You are a climate policy advocate speaking in an easy, Gen‑Z style. Be direct, practical, and encouraging. Focus on what governments, companies, and people could actually do next."},
    {"name": "Skeptic", "persona": "You are a thoughtful, chill skeptic from Gen‑Z. You ask smart questions, look for trade‑offs and missing angles, but you're not a climate denier. Keep it short and down‑to‑earth."},
]

def get_news_agent_opinions(title: str, summary: str, url: str = "") -> List[Dict]:
    """Return a list of {agent, opinion} from different AI personas who read the article and give their view. Fetches full article from url when possible."""
    # Prefer full article text from URL so agents read the entire news
    full_article = ""
    if (url or "").strip():
        full_article = fetch_article_text(url, max_chars=ARTICLE_MAX_CHARS)
    if full_article and len(full_article) >= 150:
        text = (title or "").strip()
        if text:
            text = "Headline: " + text + "\n\nFull article (excerpt):\n\n" + full_article
        else:
            text = full_article
    else:
        text = (title or "").strip()
        if (summary or "").strip():
            text += "\n\n" + (summary or "").strip()
    if not text or len(text) < 10:
        return [{"agent": "System", "opinion": "Not enough article text to form an opinion."}]
    opinions = []
    for ag in NEWS_AGENTS:
        prompt = f"""{ag["persona"]}

Read the following climate/environment news (headline and full article or snippet). Give your brief opinion in 2–4 sentences, in casual, easy‑to‑read Gen‑Z English (think clear, friendly, no buzzwords). Don't repeat the headline; say what you think it means, why it matters, and what actions or trade‑offs stand out. Base your opinion only on the text below.

---
{text[:6000]}
---

Your opinion (in simple Gen‑Z style):"""
        try:
            opinion = llm_complete(prompt, max_new_tokens=220, temp=0.5).strip()
            if not opinion or len(opinion) < 15:
                opinion = "No comment."
            meta_phrases = ("please share", "share the", "go ahead and share", "i'm ready to", "ready to provide", "provide the", "send the article")
            if any(p in opinion.lower() for p in meta_phrases):
                opinion = "Based on the article: this is relevant to climate or environment. Key points and policy implications depend on the full context above."
            opinions.append({"agent": ag["name"], "opinion": opinion})
        except Exception as e:
            opinions.append({"agent": ag["name"], "opinion": f"(Could not generate: {e})"})
    return opinions

# ---------------------------------------------------------
# Calculator logic (one-shot)
# ---------------------------------------------------------
def run_calculator(user_input: str, goal_lbs: float) -> dict:
    """Returns {distance_kg, electricity_kg, meat_kg, total_kg, total_lbs, goal_kg, under_goal, ai_analysis, classification_error }."""
    user_input = (user_input or "").strip()
    goal_kg = goal_lbs / LBS_PER_KG
    # Transport: EPA typical passenger vehicle (~0.404 kg CO2/mile)
    miles = sum(int(m) for m in re.findall(r'(\d+)\s*miles?\b', user_input.lower()))
    kms = sum(int(m) for m in re.findall(r'(\d+)\s*km\b', user_input.lower()))
    distance_kg = miles * KG_CO2_PER_MILE_CAR + kms * KG_CO2_PER_KM_CAR
    # Electricity: US grid average ~0.35 kg/kWh (EPA eGRID). If no number, default 4 kWh for AC/heating mention.
    kwh_vals = re.findall(r'(\d+)\s*(?:kwh|kilowatt)', user_input.lower())
    if kwh_vals:
        electricity_kg = sum(int(k) for k in kwh_vals) * KG_CO2_PER_KWH_US
    else:
        mentions = len(re.findall(r'\b(?:electricity|ac|heating|kwh)\b', user_input.lower()))
        electricity_kg = mentions * DEFAULT_ELECTRICITY_KWH * KG_CO2_PER_KWH_US if mentions else 0.0
    # Food: different factors per meat type (Our World in Data / Poore & Nemecek)
    meat_kg = 0.0
    n_beef = len(re.findall(r'\bbeef\b', user_input.lower()))
    n_pork = len(re.findall(r'\bpork\b', user_input.lower()))
    n_chicken = len(re.findall(r'\bchicken\b', user_input.lower()))
    n_meat_generic = len(re.findall(r'\bmeat\b', user_input.lower()))
    meat_kg = n_beef * KG_CO2_BEEF_SERVING + n_pork * KG_CO2_PORK_SERVING + n_chicken * KG_CO2_CHICKEN_SERVING
    if meat_kg == 0 and n_meat_generic > 0:
        meat_kg = n_meat_generic * KG_CO2_MEAT_AVG_SERVING
    total_kg = distance_kg + electricity_kg + meat_kg
    total_lbs = kg_to_lbs(total_kg)
    under_goal = total_kg <= goal_kg
    ai_analysis = None
    classification_error = None
    if user_input:
        try:
            classification_prompt = f"""Is this text describing a daily routine or daily activities?\n\nText: {user_input}\n\nAnswer with ONLY: DAILY ROUTINE or NOT DAILY ROUTINE"""
            classification = llm_complete(classification_prompt, max_new_tokens=8, temp=0.0).strip()
            if "DAILY ROUTINE" not in classification.upper():
                classification_error = "That doesn't look like a typical day. Try describing specific activities."
            else:
                total_lbs_1 = round(total_lbs, 1)
                goal_lbs_1 = round(goal_lbs, 1)
                over = round(total_lbs - goal_lbs, 1) if not under_goal else 0
                analysis_prompt = f"""You are helping someone understand their daily carbon footprint in plain language. Use the numbers below.

**Numbers (use these):**
- This day's estimated footprint: **{total_lbs_1} lb CO₂**
- Their daily goal: **{goal_lbs_1} lb CO₂**
- Typical person (for comparison): **{TYPICAL_DAILY_LBS} lb CO₂/day**
- They are {"under" if under_goal else "over"} their goal{f' by {over} lb' if not under_goal else ''}.

**Daily routine they described:**
{user_input}

Write a short analysis that really helps them understand. Use this structure and format (use * for bullet points and ** for bold):

**In a nutshell**
One sentence: what today's footprint means and how it compares to their goal and to a typical day.

**What each part of your day means**
For each activity they mentioned: one line in plain language (what it does to the climate, and roughly how much it adds if you can).

**Put it in perspective**
One or two comparisons they can relate to (e.g. "That's like…", "Equivalent to…", "Roughly X% of a typical day").

**Quick wins**
2–3 specific, doable changes (e.g. "Carpool once a week" or "Try one meat-free day") — not generic advice.

**Bottom line**
One sentence: positive note if they're under goal, or one clear next step if they're over.

Keep it concise and friendly. Use * for bullets and ** for bold only."""
                ai_analysis = llm_complete(analysis_prompt, max_new_tokens=720, temp=0.5)
        except Exception as e:
            classification_error = str(e)
    return {
        "distance_kg": distance_kg, "electricity_kg": electricity_kg, "meat_kg": meat_kg,
        "total_kg": total_kg, "total_lbs": total_lbs, "goal_kg": goal_kg, "goal_lbs": goal_lbs,
        "under_goal": under_goal, "ai_analysis": ai_analysis, "classification_error": classification_error,
    }

# ---------------------------------------------------------
# Region-aware optimizer helpers
# ---------------------------------------------------------
FOOD_KG_PER_SERVING = {"Chicken": 1.0, "Pork": 1.2, "Vegetarian": 0.7}
TRANSPORT_KG_PER_MILE = {"Gasoline car": 0.404, "Hybrid": 0.25, "EV": 0.0, "Bus": 0.089, "Rail": 0.041, "Bike/Walk": 0.0}

def _read_csv_map(path: str, key_col: str, val_col: str) -> dict:
    out = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                k = str(row.get(key_col, "")).strip()
                v = str(row.get(val_col, "")).strip()
                if k and v:
                    out[k] = v
    except FileNotFoundError:
        pass
    return out

@lru_cache(maxsize=1)
def load_grid_data():
    subregion_to_kg = {"US_AVG": 0.35, "EU_AVG": 0.28, "GLOBAL": 0.47}
    zip_to_subregion = {}
    csv_sub = _read_csv_map(str(DATA_DIR / "egrid_subregion_factors.csv"), "subregion", "kg_per_kwh")
    if csv_sub:
        try:
            subregion_to_kg.update({k.upper(): float(v) for k, v in csv_sub.items()})
        except Exception:
            pass
    csv_zip = _read_csv_map(str(DATA_DIR / "zip_to_egrid.csv"), "zip", "subregion")
    if csv_zip:
        zip_to_subregion = {k.zfill(5): v.upper() for k, v in csv_zip.items()}
    return subregion_to_kg, zip_to_subregion

def get_grid_factor(zip_code: Optional[str], location_label: str) -> Tuple[float, str, bool]:
    subregion_to_kg, zip_to_subregion = load_grid_data()
    loc = (location_label or "United States").lower()
    if zip_code:
        z = "".join([c for c in zip_code if c.isdigit()])
        z = z[:5].zfill(5) if z else ""
        if z and z in zip_to_subregion:
            sr = zip_to_subregion[z]
            if sr in subregion_to_kg:
                return float(subregion_to_kg[sr]), f"{sr} grid", False
    if "europe" in loc:
        return float(subregion_to_kg.get("EU_AVG", 0.28)), "EU average", True
    if "global" in loc:
        return float(subregion_to_kg.get("GLOBAL", 0.47)), "Global average", True
    return float(subregion_to_kg.get("US_AVG", 0.35)), "US average", True

def ev_kg_per_mile(grid_kg_per_kwh: float, kwh_per_mile: float = 0.30) -> float:
    return grid_kg_per_kwh * kwh_per_mile

def compute_scenario(grid_kg_per_kwh: float, kwh_per_day: float, commute_miles: float,
                     commute_mode: str, meals_per_day: float, meal_type: str,
                     ev_kwh_per_mile: float = 0.30) -> dict:
    elec_kg = kwh_per_day * grid_kg_per_kwh
    if commute_mode == "EV":
        trans_kg = commute_miles * ev_kg_per_mile(grid_kg_per_kwh, ev_kwh_per_mile)
    else:
        trans_kg = commute_miles * TRANSPORT_KG_PER_MILE.get(commute_mode, 0.0)
    food_kg = meals_per_day * FOOD_KG_PER_SERVING.get(meal_type, 0.0)
    total = elec_kg + trans_kg + food_kg
    return {"electricity": elec_kg, "transport": trans_kg, "food": food_kg, "total": total}

# ---------------------------------------------------------
# Reddit search (Community tab: climate/environment only)
# ---------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AI-Climate-Impact-Calculator/3.0 (contact: you@example.com)"})

# Only search these subreddits so results stay on-topic (climate, environment, sustainability, outdoor)
ALLOWED_COMMUNITY_SUBREDDITS = [
    "climate", "environment", "sustainability", "ZeroWaste", "ClimateActionPlan",
    "renewableenergy", "solar", "electricvehicles", "bikecommuting", "permaculture",
    "homesteading", "gardening", "outdoors", "CampingGear", "hiking", "snowboarding",
    "skiing", "survival", "preppers", "collapse", "climatechange", "ecology",
    "conservation", "environmental_science", "Green", "vegan", "PlantBasedDiet",
]

def reddit_search(query: str, limit: int = 12, time_window: str = "year", subs=None):
    results = []
    # Never search all of Reddit: use only allowed climate/environment subreddits
    if not subs:
        subs = ALLOWED_COMMUNITY_SUBREDDITS
    else:
        allowed_set = {s.lower() for s in ALLOWED_COMMUNITY_SUBREDDITS}
        subs = [s for s in subs if s and s.lstrip("r/").lower() in allowed_set]
        if not subs:
            subs = ALLOWED_COMMUNITY_SUBREDDITS
    def _collect(url):
        try:
            r = SESSION.get(url, timeout=12)
            r.raise_for_status()
            data = r.json()
            for child in data.get("data", {}).get("children", []):
                d = child.get("data", {})
                if d.get("over_18"):
                    continue
                results.append({
                    "id": d.get("id"), "title": d.get("title", ""), "subreddit": d.get("subreddit", ""),
                    "author": d.get("author", ""), "url": f"https://www.reddit.com{d.get('permalink','')}",
                    "selftext": (d.get("selftext") or "")[:1800], "score": int(d.get("score", 0)),
                    "created": datetime.utcfromtimestamp(d.get("created_utc", 0))
                })
        except Exception as e:
            logger.warning(f"Reddit fetch failed: {e}")
    enc_q = quote_plus(query)
    for s in subs:
        url = f"https://www.reddit.com/r/{s}/search.json?q={enc_q}&restrict_sr=1&sort=relevance&t={time_window}&limit={limit}"
        _collect(url)
    dedup = {}
    for p in results:
        if p["id"]:
            dedup[p["id"]] = p
    return list(dedup.values())

def reddit_top_comments(post_id: str, limit: int = 4):
    out = []
    try:
        url = f"https://www.reddit.com/comments/{post_id}.json?limit=25&sort=top"
        r = SESSION.get(url, timeout=12)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, list) and len(payload) > 1:
            comments = payload[1].get("data", {}).get("children", [])
            for c in comments:
                if c.get("kind") != "t1":
                    continue
                body = c.get("data", {}).get("body", "")
                if body and len(body) > 30:
                    out.append(body.strip().replace("\n", " ")[:600])
                if len(out) >= limit:
                    break
    except Exception as e:
        logger.info(f"Could not fetch comments for {post_id}: {e}")
    return out

# ---------------------------------------------------------
# AI helpers for Reddit (deep analysis + strict relevance)
# ---------------------------------------------------------
def ai_deep_analyze_problem(issue: str) -> dict:
    """Deeply analyze the user's problem to drive targeted Reddit search and strict filtering.
    Returns: intent, queries, must_terms, subreddits, relevance_criteria."""
    prompt = f"""You are an expert at understanding climate/environment/outdoor problems and finding Reddit posts that STRONGLY match.

User's problem (describe in their words):
\"\"\"{issue}\"\"\"

Analyze deeply:
1. What is the user's exact situation and what do they need? (one short sentence = "intent")
2. Which Reddit search phrases would find posts where people had the SAME problem or gave direct solutions? (3-5 short search queries, use quotes for phrases)
3. Which words MUST appear in a post for it to be strongly related? (5-10 must_terms: specific to this problem, e.g. snow, ice, walk, tips, boots, grip)
4. Which subreddits are most likely to have this exact topic? Pick from: climate, environment, sustainability, ZeroWaste, outdoors, hiking, skiing, snowboarding, survival, gardening, permaculture, CampingGear, renewableenergy, bikecommuting, ecology (list 2-4)
5. In one sentence, what makes a post "strongly related"? (relevance_criteria)

Return ONLY valid JSON, no other text:
{{"intent": "<one sentence>", "queries": ["q1", "q2", "q3"], "must_terms": ["term1", "term2", ...], "subreddits": ["sub1", "sub2"], "relevance_criteria": "<one sentence>"}}"""
    raw = llm_complete(prompt, max_new_tokens=380, temp=0.2)
    try:
        jtxt = re.search(r"\{.*\}", raw, flags=re.S).group(0)
        plan = json.loads(jtxt)
    except Exception:
        plan = {}
    intent = (plan.get("intent") or issue[:200]).strip()
    queries = [q.strip() for q in plan.get("queries", []) if q and len(q.strip()) > 1][:6]
    if not queries:
        queries = [issue[:80]]
    # Always include user's own words as a query so Reddit returns a broader pool
    user_query = issue[:70].strip()
    if user_query and user_query not in queries:
        queries = [user_query] + queries[:5]
    must_terms = [t.strip().lower() for t in plan.get("must_terms", []) if t][:12]
    # Ensure key words from the issue appear in must_terms so last-resort can match
    issue_words = [w for w in re.findall(r"[a-z]{3,}", issue.lower()) if w not in ("the", "and", "for", "you", "your", "can", "cant", "looking")]
    for w in issue_words[:5]:
        if w not in must_terms:
            must_terms.append(w)
    must_terms = must_terms[:14]
    subreddits = [s.strip().lstrip("r/") for s in plan.get("subreddits", []) if s][:5]
    relevance_criteria = (plan.get("relevance_criteria") or "Post addresses the same problem or gives direct, practical advice for it.").strip()
    return {
        "intent": intent,
        "queries": queries,
        "must_terms": must_terms,
        "subreddits": subreddits,
        "relevance_criteria": relevance_criteria,
        "time_window": "year",
    }


def ai_build_search_plan(issue: str) -> dict:
    """Build search plan from deep analysis (kept for compatibility; prefers deep_analyze)."""
    return ai_deep_analyze_problem(issue)


def ai_is_strongly_related(issue: str, analysis: dict, post_title: str, post_text: str, min_terms: int = 2) -> bool:
    """Strict filter: only YES if the post is STRONGLY related to the user's specific problem.
    min_terms: require at least this many must_terms in the post (use 1 for fallback when no strict matches)."""
    intent = analysis.get("intent", issue)
    criteria = analysis.get("relevance_criteria", "")
    must_terms = analysis.get("must_terms", [])
    base = (post_title + " " + post_text).lower()
    term_hits = sum(1 for t in must_terms if t.lower() in base) if must_terms else 0
    if term_hits < min_terms:
        return False
    mt_str = ", ".join(must_terms[:10]) if must_terms else "none"
    prompt = f"""You are a strict relevance filter. Say YES only if the post is clearly related to the user's problem and could help.

User's problem: {intent}

What counts as related: {criteria}

Post title: {post_title}
Post body (excerpt): {post_text[:700]}

Is this post about the same or a very similar issue and does it offer useful advice? Answer ONLY: YES or NO."""
    try:
        ans = llm_complete(prompt, max_new_tokens=8, temp=0.0).strip().upper()
        return "YES" in ans
    except Exception:
        return term_hits >= max(min_terms, 2)


def ai_is_somewhat_related(issue: str, analysis: dict, post_title: str, post_text: str) -> bool:
    """Looser filter: post must be somewhat related to the user's problem (same topic area, could be vaguely helpful).
    Reject random or off-topic posts. Used when we want to show something related but not strictly matching."""
    must_terms = analysis.get("must_terms", [])
    base = (post_title + " " + post_text).lower()
    term_hits = sum(1 for t in must_terms if t.lower() in base) if must_terms else 0
    if term_hits < 1:
        return False
    intent = analysis.get("intent", issue)
    prompt = f"""User's problem (topic): {intent}

Post title: {post_title}
Post body (excerpt): {post_text[:600]}

Is this post at least somewhat related to the user's topic (same general area: e.g. winter/outdoor, climate, environment, practical tips)? Say NO if it's clearly random, unrelated (e.g. gaming, relationships, politics), or spam. Answer ONLY: YES or NO."""
    try:
        ans = llm_complete(prompt, max_new_tokens=8, temp=0.0).strip().upper()
        return "YES" in ans
    except Exception:
        return term_hits >= 2


def ai_is_relevant(issue: str, post_title: str, post_text: str, must_terms=None) -> bool:
    """Legacy: use ai_is_strongly_related with a minimal analysis dict when analysis not available."""
    analysis = {"intent": issue, "relevance_criteria": "Same problem or direct practical advice.", "must_terms": must_terms or []}
    return ai_is_strongly_related(issue, analysis, post_title, post_text)


def ai_generate_problem_solution(issue: str, analysis: dict) -> str:
    """Generate an AI-written solution to the user's problem (practical advice, tips, cautions). Shown first before Reddit posts."""
    intent = analysis.get("intent", issue)
    prompt = f"""The user has this climate/environment/outdoor-related problem:

"{issue}"

(Interpreted as: {intent})

Write a direct, practical solution. Use this structure (use **bold** for section labels, not # or ###):
- **What you can do:** 3–5 concrete steps or tips.
- **Key things to keep in mind:** 2–4 short bullets.
- **Cautions:** 2–3 things to avoid or be careful about.

Use simple language. No markdown headings (no # or ###)."""
    try:
        return llm_complete(prompt, max_new_tokens=400, temp=0.4).strip()
    except Exception:
        return ""


def ai_summarize_findings(issue: str, posts: list, skip_comments: bool = True):
    """Summarize Reddit findings. skip_comments=True avoids extra HTTP calls for speed."""
    context_chunks = []
    for p in posts[:6]:
        if skip_comments:
            snippet = f"Title: {p['title']}\nPost: {p.get('selftext', '')}"
        else:
            comments = reddit_top_comments(p["id"], limit=3)
            snippet = f"Title: {p['title']}\nPost: {p['selftext']}\nTop comments: {' | '.join(comments)}"
        context_chunks.append(snippet[:1000])
    context = "\n\n---\n".join(context_chunks) if context_chunks else "No context."
    prompt = (
        f"User climate/health issue:\n{issue}\n\n"
        "From the Reddit context below, extract concrete advice that actually helped people.\n"
        "Return:\n- Key themes (3–5 bullets)\n- What others did that worked (5–8 bullets)\n- Cautions (2–4 bullets)\n\n"
        "Context:\n" + context
    )
    try:
        return llm_complete(prompt, max_new_tokens=320, temp=0.6)
    except Exception:
        return ""


# ---------------------------------------------------------
# Web search for related articles (Ask Climi)
# ---------------------------------------------------------
def _search_web_duckduckgo_html(query: str, max_results: int) -> List[Dict]:
    """Fallback: scrape DuckDuckGo HTML for links when duckduckgo_search package unavailable."""
    import re
    from urllib.parse import urlparse, quote_plus
    out = []
    seen_urls = set()
    try:
        # DuckDuckGo HTML accepts GET with ?q= as well
        base = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        # Prefer POST like a form to avoid encoding issues
        r = requests.post(base, data={"q": query[:200]}, headers=headers, timeout=12)
        r.raise_for_status()
        html = r.text
        # Any <a ... href="https://..." that is not duckduckgo, in order
        all_links = re.findall(r'href="(https?://[^"]+)"', html)
        for href in all_links:
            href = href.strip()
            if "duckduckgo.com" in href or href in seen_urls:
                continue
            if len(out) >= max_results:
                break
            try:
                parsed = urlparse(href)
                if not parsed.netloc or parsed.netloc.startswith("duckduckgo"):
                    continue
                # Use domain + path as readable title
                title = (parsed.netloc or "") + (parsed.path[:50] if parsed.path else "")
                if not title:
                    title = href[:80]
                seen_urls.add(href)
                out.append({"title": title[:200], "url": href, "snippet": ""})
            except Exception:
                continue
    except Exception as e:
        logger.warning("DuckDuckGo HTML fallback failed: %s", e)
    return out[:max_results]


def search_web_articles(query: str, max_results: int = 5) -> List[Dict]:
    """Search the web for articles relevant to the user's question. Returns list of {title, url, snippet}."""
    query = (query or "").strip()[:200]
    if not query:
        return []
    out: List[Dict] = []
    # 1) Try duckduckgo_search package (pip install duckduckgo-search)
    for pkg in ("duckduckgo_search", "ddgs"):
        try:
            if pkg == "ddgs":
                from ddgs import DDGS
            else:
                from duckduckgo_search import DDGS
            # Newer API: DDGS() is context manager; .text() returns iterable of dicts
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            for r in results:
                title = (r.get("title") or "").strip()
                url = (r.get("href") or r.get("url") or "").strip()
                body = (r.get("body") or r.get("snippet") or "").strip()
                if url and title and not url.startswith("https://duckduckgo.com"):
                    out.append({"title": title[:200], "url": url, "snippet": body[:280] if body else ""})
            if out:
                return out[:max_results]
        except ImportError:
            continue
        except Exception as e:
            logger.warning("Web search (%s) failed: %s", pkg, e)
            out = []
    # 2) Fallback: DuckDuckGo HTML scrape
    if len(out) < max_results:
        fallback = _search_web_duckduckgo_html(query, max_results)
        seen = {x["url"] for x in out}
        for item in fallback:
            if item["url"] not in seen and len(out) < max_results:
                out.append(item)
                seen.add(item["url"])
    # 3) Last resort: curated "learn more" links so user always sees something
    if len(out) < 2:
        curated = [
            ("EPA – Climate Change", "https://www.epa.gov/climate-change"),
            ("NASA – Climate", "https://climate.nasa.gov/"),
            ("WWF – Climate & Energy", "https://www.worldwildlife.org/threats/effects-of-climate-change"),
            ("UN – Climate Action", "https://www.un.org/en/climatechange"),
        ]
        for title, url in curated[: max_results - len(out)]:
            out.append({"title": title, "url": url, "snippet": "Learn more about climate and environment."})
    return out[:max_results]


# ---------------------------------------------------------
# Progress logging
# ---------------------------------------------------------
def load_footprint_log():
    """Load footprint log (alias for _load_log for Flask)"""
    if LOG_PATH.exists():
        try:
            import pandas as pd
            return pd.read_csv(LOG_PATH, parse_dates=["ts"])
        except Exception:
            import pandas as pd
            return pd.DataFrame(columns=["ts", "kg"])
    import pandas as pd
    return pd.DataFrame(columns=["ts", "kg"])

# ---------------------------------------------------------
# Weather/Risk helpers
# ---------------------------------------------------------
AQI_BREAKPOINTS_PM25 = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def _pm25_to_aqi(pm25: float) -> int:
    if pm25 is None:
        return 0
    x = float(pm25)
    for c_low, c_high, aqi_low, aqi_high in AQI_BREAKPOINTS_PM25:
        if c_low <= x <= c_high:
            return int(round((aqi_high - aqi_low) / (c_high - c_low) * (x - c_low) + aqi_low))
    return 500

OPENMETEO = requests.Session()
OPENMETEO.headers.update({"User-Agent": "Climatrack/1.0 (Flask app)"})

def geocode_place(q: str) -> Optional[dict]:
    q = (q or "").strip()
    if not q:
        return None
    try:
        r = OPENMETEO.get("https://geocoding-api.open-meteo.com/v1/search",
                          params={"name": q, "count": 1, "language": "en", "format": "json"}, timeout=20)
        r.raise_for_status()
        js = r.json()
        res = (js.get("results") or [None])[0]
        if res:
            return {"name": res.get("name"), "lat": float(res["latitude"]), "lon": float(res["longitude"]),
                    "admin": res.get("admin1"), "country": res.get("country")}
    except Exception:
        pass
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": 1},
                         headers={"User-Agent": "Climatrack/1.0"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data:
            d = data[0]
            return {"name": d.get("display_name", q), "lat": float(d["lat"]), "lon": float(d["lon"]),
                    "admin": None, "country": None}
    except Exception as e:
        logger.info(f"Fallback geocode failed: {e}")
    return None

@ttl_cache(15 * 60)
def openmeteo_air_quality(lat: float, lon: float):
    import pandas as pd
    r = OPENMETEO.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                      params={"latitude": lat, "longitude": lon, "hourly": "pm2_5", "timezone": "UTC", "forecast_days": 4},
                      timeout=20)
    r.raise_for_status()
    js = r.json()
    times = js.get("hourly", {}).get("time", []) or []
    pm25 = js.get("hourly", {}).get("pm2_5", []) or []
    df = pd.DataFrame({"ts": pd.to_datetime(times), "pm25": pm25})
    df["aqi"] = df["pm25"].apply(_pm25_to_aqi)
    return df

@ttl_cache(15 * 60)
def openmeteo_weather(lat: float, lon: float):
    import pandas as pd
    r = OPENMETEO.get("https://api.open-meteo.com/v1/forecast",
                      params={"latitude": lat, "longitude": lon,
                              "hourly": "temperature_2m,relative_humidity_2m,apparent_temperature",
                              "timezone": "UTC", "forecast_days": 4}, timeout=20)
    r.raise_for_status()
    js = r.json()
    H = js.get("hourly", {})
    df = pd.DataFrame({
        "ts": pd.to_datetime(H.get("time", []) or []),
        "temp_c": H.get("temperature_2m", []) or [],
        "rh": H.get("relative_humidity_2m", []) or [],
        "apparent_c": H.get("apparent_temperature", []) or []
    })
    return df

@ttl_cache(60 * 60)
def openmeteo_recent_baseline(lat: float, lon: float) -> float:
    import pandas as pd
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=35)
    r = OPENMETEO.get("https://archive-api.open-meteo.com/v1/archive",
                      params={"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                              "end_date": end.isoformat(), "daily": "temperature_2m_max", "timezone": "UTC"},
                      timeout=20)
    r.raise_for_status()
    js = r.json()
    daily = js.get("daily", {})
    vals = daily.get("temperature_2m_max", []) or []
    if not vals:
        return float("nan")
    return float(pd.Series(vals).mean())

def heat_index_c(temp_c: float, rh: float) -> float:
    Tf = temp_c * 9/5 + 32
    if Tf < 80:
        HI_f = 0.5 * (Tf + 61.0 + ((Tf - 68.0) * 1.2) + (rh * 0.094))
    else:
        HI_f = (-42.379 + 2.04901523 * Tf + 10.14333127 * rh
                - 0.22475541 * Tf * rh - 6.83783e-3 * Tf**2
                - 5.481717e-2 * rh**2 + 1.22874e-3 * Tf**2 * rh
                + 8.5282e-4 * Tf * rh**2 - 1.99e-6 * Tf**2 * rh**2)
    return (HI_f - 32) / 1.8

def score_from_aqi(aqi: float) -> float:
    return max(0.0, min(100.0, (aqi / 300.0) * 100.0))

def score_from_heatindex_c(hi_c: float) -> float:
    hi_f = hi_c * 9/5 + 32
    if hi_f < 80:
        return 0.0
    if hi_f < 90:
        return 20.0 * (hi_f - 80) / 10.0
    if hi_f < 103:
        return 20 + 30.0 * (hi_f - 90) / 13.0
    if hi_f < 124:
        return 50 + 30.0 * (hi_f - 103) / 21.0
    return 100.0

def score_from_temp_anomaly(anom_c: float) -> float:
    if anom_c <= 0:
        return 0.0
    if anom_c >= 8.0:
        return 100.0
    return 12.5 * anom_c

def composite_risk(aqi_score: float, hi_score: float, anom_score: float) -> float:
    return round(0.45 * aqi_score + 0.35 * hi_score + 0.20 * anom_score, 1)
