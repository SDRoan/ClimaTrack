import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import json
import matplotlib.pyplot as plt
import requests
import os
import logging
from datetime import datetime, timedelta, timezone, date
from urllib.parse import quote_plus, urlparse
import csv
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from html import unescape, escape
import xml.etree.ElementTree as ET
from transformers import pipeline
import torch
import pandas as pd
import plotly.express as px

try:
    from shapely.geometry import shape as shapely_shape  # noqa
except Exception:
    shapely_shape = None

# ✅ Auto-refresh helper (separate package)
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*args, **kwargs):
        return None

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
NEWS_TTL_SEC = 180  # cache TTL for news sources (3 minutes)
DEBUG_AQI = False   

# ---- Units helpers ----
LBS_PER_KG = 2.20462262
UNITS_DEFAULT = "Imperial (lbs CO₂)"  
def kg_to_lbs(x: float) -> float:
    return float(x) * LBS_PER_KG

def convert_mass_value(value_kg: float, units_label: str = UNITS_DEFAULT) -> float:
    return kg_to_lbs(value_kg)

def unit_suffix(units_label: str = UNITS_DEFAULT) -> str:
    return "lb CO₂"

# ---------------------------------------------------------
# Lightweight summarizer used in one helper (kept as-is)
# ---------------------------------------------------------
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
def detect_activities_with_ai(user_input):
    prompt = (
        "Classify the user's activities based on the following routine:\n"
        f"{user_input}\n\n"
        "Categories: transportation, electricity usage, meat consumption\n"
        "Respond in JSON with confidence scores from 0.0 to 1.0. Example:\n"
        '{"transportation": 0.7, "electricity usage": 0.4, "meat consumption": 0.9}'
    )
    try:
        result = summarizer(prompt, max_length=100, min_length=30, do_sample=False)
        text = result[0]["summary_text"]
        return eval(text.strip())
    except:
        return {"transportation": 0.3, "electricity usage": 0.3, "meat consumption": 0.3}

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------
st.set_page_config(
  page_title="Climatrack",
  page_icon="🌍",
  layout="wide",
  initial_sidebar_state="collapsed"
)

if st.session_state.get("units") != UNITS_DEFAULT:
    st.session_state["units"] = UNITS_DEFAULT

# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------
st.markdown("""
<style>
   .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; margin: 1rem 0; }
   .metric-card { background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0; }
   .metric-value { font-size: 3rem; font-weight: bold; margin: 1rem 0; }
   .metric-label { font-size: 1.2rem; color: #666; margin-bottom: 0.5rem; }
   .status-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 25px; font-weight: bold; margin: 0.5rem; }
   .status-success { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; }
   .status-warning { background: linear-gradient(135deg, #f39c12, #e67e22); color: white; }
   .status-info { background: linear-gradient(135deg, #3498db, #2980b9); color: white; }
   .interactive-card:hover { transform: translateY(-5px); }
   h1, h2, h3 { color: #2c3e50; font-weight: bold; }
   .stButton > button { background: linear-gradient(135deg, #2ecc71, #27ae60); color: white; border: none; padding: 1rem 2rem; border-radius: 25px; font-weight: bold; font-size: 1.1rem; transition: all 0.3s ease; }
   .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3); }
   .stTextInput > div > input { border: 2px solid #e1e8ed; border-radius: 10px; padding: 1rem; font-size: 1rem; }
   .stTextInput > div > input:focus { border-color: #2ecc71; box-shadow: 0 0 0 3px rgba(46, 204, 113, 0.1); }
   .stSlider > div > div > div > div { background: linear-gradient(90deg, #2ecc71, #3498db); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper: themed notice boxes
# ---------------------------------------------------------
def notice(message: str, level: str = "info"):
    colors = {"info": "#2E86C1","success": "#1E8449","warning": "#B9770E","error": "#922B21"}
    color = colors.get(level, colors["info"])
    st.markdown(
        f"""
        <div style="margin:10px 0;padding:12px 14px;border-left:6px solid {color};
                    border-radius:10px;background: rgba(255,255,255,0.04);">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Models (Ollama → fallback to FLAN-T5)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
  flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
  flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
  return flan_tokenizer, flan_model

flan_tokenizer, flan_model = load_models()

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
  inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
  outputs = flan_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temp, do_sample=temp > 0.0)
  return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# Reddit (no API key)
# ---------------------------------------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "AI-Climate-Impact-Calculator/3.0 (contact: you@example.com)"})

def reddit_search(query: str, limit: int = 12, time_window: str = "year", subs=None):
  results = []
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
  if subs:
      for s in subs:
          url = f"https://www.reddit.com/r/{s}/search.json?q={enc_q}&restrict_sr=1&sort=relevance&t={time_window}&limit={limit}"
          _collect(url)
  else:
      url = f"https://www.reddit.com/search.json?q={enc_q}&restrict_sr=0&sort=relevance&t={time_window}&limit={limit}"
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
# AI helpers
# ---------------------------------------------------------
def ai_build_search_plan(issue: str) -> dict:
  prompt = f"""
You are planning a Reddit search based on a user's environmental/climate problem.

User issue:
\"\"\"{issue}\"\"\"

Return ONLY compact JSON with this exact schema, no commentary:

{{"queries": ["<q1>", "<q2>", "<q3>"], "subreddits": ["<s1>", "<s2>"], "must_terms": ["<k1>", "<k2>"], "time_window": "week|month|year"}}
"""
  raw = llm_complete(prompt, max_new_tokens=260, temp=0.2)
  try:
      jtxt = re.search(r"\{.*\}", raw, flags=re.S).group(0)
      plan = json.loads(jtxt)
  except Exception:
      plan = {"queries": [issue[:80]], "subreddits": [], "must_terms": [], "time_window": "year"}
  plan["queries"] = [q.strip() for q in plan.get("queries", []) if q and len(q.strip()) > 1][:5]
  plan["subreddits"] = [s.strip().lstrip("r/") for s in plan.get("subreddits", []) if s][:5]
  plan["must_terms"] = [t.strip().lower() for t in plan.get("must_terms", []) if t][:8]
  plan["time_window"] = plan.get("time_window") or "year"
  if not plan["queries"]:
      plan["queries"] = [issue[:80]]
  return plan

def ai_is_relevant(issue: str, post_title: str, post_text: str, must_terms=None) -> bool:
  mt = ", ".join(must_terms or [])
  prompt = f"""
User issue:
{issue}

Candidate post:
Title: {post_title}
Body: {post_text[:700]}

Key terms to consider: {mt if mt else "None"}.
Answer ONLY: YES or NO
"""
  try:
      ans = llm_complete(prompt, max_new_tokens=10, temp=0.0).strip()
      return "YES" in ans.upper()
  except Exception:
      base = (post_title + " " + post_text).lower()
      hits = sum(1 for w in (must_terms or []) if w.lower() in base)
      return hits >= 1

def ai_summarize_findings(issue: str, posts: list):
  context_chunks = []
  for p in posts[:6]:
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
# Region-aware optimizer helpers
# ---------------------------------------------------------
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

@st.cache_data(show_spinner=False)
def load_grid_data():
  subregion_to_kg = {"US_AVG": 0.40, "EU_AVG": 0.28, "GLOBAL": 0.47}
  zip_to_subregion = {}
  csv_sub = _read_csv_map("data/egrid_subregion_factors.csv", "subregion", "kg_per_kwh")
  if csv_sub:
      try:
          subregion_to_kg.update({k.upper(): float(v) for k, v in csv_sub.items()})
      except Exception:
          pass
  csv_zip = _read_csv_map("data/zip_to_egrid.csv", "zip", "subregion")
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
  return float(subregion_to_kg.get("US_AVG", 0.40)), "US average", True

FOOD_KG_PER_SERVING = {"Chicken": 1.7, "Vegetarian": 0.7}
TRANSPORT_KG_PER_MILE = {"Gasoline car": 0.404, "Hybrid": 0.25, "Bus": 0.089, "Rail": 0.041, "Bike/Walk": 0.0}

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
# News utilities (unchanged logic)
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

      norm.append({
          "title": title,
          "url": url,
          "source": it.get("source") or _domain(url),
          "summary": summary,
          "published": published,
      })

  seen, out = set(), []
  for x in norm:
      if x["url"] in seen:
          continue
      seen.add(x["url"])
      out.append(x)
  return out

@st.cache_data(ttl=NEWS_TTL_SEC, show_spinner=False)
def fetch_gdelt_news(query: str, window: str, limit: int = 24, fresh: int = 0) -> List[Dict]:
  q = f'(climate OR environment) {query}'.strip()
  params = {"query": q, "mode": "ArtList", "maxrecords": str(limit), "format": "json", "timespan": _time_window_to_timespan(window)}
  try:
      r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
      r.raise_for_status()
      js = r.json()
      items = []
      for a in js.get("articles", []):
          items.append({
              "title": a.get("title", ""), "url": a.get("url", ""),
              "source": a.get("source") or a.get("domain"), "summary": a.get("seendate") or "",
              "published": a.get("seendate"),
          })
      return _normalize_items(items)
  except Exception as e:
      logger.info(f"GDELT fetch failed: {e}")
      return []

@st.cache_data(ttl=NEWS_TTL_SEC, show_spinner=False)
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

@st.cache_data(ttl=NEWS_TTL_SEC, show_spinner=False)
def fetch_newsapi(query: str, window: str, limit: int = 24, fresh: int = 0) -> List[Dict]:
  key = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))
  if not key:
      return []
  date_from = (datetime.utcnow() - {"24h": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30)}.get(window, timedelta(days=7))).strftime("%Y-%m-%dT%H:%M:%SZ")
  params = {"q": f"(climate OR environment) {query}".strip(), "from": date_from, "sortBy": "publishedAt", "pageSize": str(limit), "language": "en"}
  try:
      r = requests.get("https://newsapi.org/v2/everything", params=params, headers={"X-Api-Key": key}, timeout=20)
      r.raise_for_status()
      js = r.json()
      items = []
      for a in js.get("articles", []):
          items.append({
              "title": a.get("title", ""), "url": a.get("url", ""),
              "source": (a.get("source") or {}).get("name", ""), "summary": a.get("description", ""),
              "published": a.get("publishedAt", ""),
          })
      return _normalize_items(items)
  except Exception as e:
      logger.info(f"NewsAPI fetch failed: {e}")
      return []

@st.cache_data(ttl=NEWS_TTL_SEC, show_spinner=False)
def fetch_guardian(query: str, window: str, limit: int = 24, fresh: int = 0) -> List[Dict]:
  key = st.secrets.get("GUARDIAN_API_KEY", os.getenv("GUARDIAN_API_KEY", ""))
  if not key:
      return []
  date_from = (datetime.utcnow() - {"24h": timedelta(days=1), "7d": timedelta(days=7), "30d": timedelta(days=30)}.get(window, timedelta(days=7))).strftime("%Y-%m-%d")
  params = {"q": query or "climate OR environment","section": "environment","from-date": date_from,
            "page-size": str(limit),"order-by": "newest","api-key": key,"show-fields": "trailText"}
  try:
      r = requests.get("https://content.guardianapis.com/search", params=params, timeout=20)
      r.raise_for_status()
      resp = r.json().get("response", {})
      items_raw = resp.get("results", [])
      items = []
      for a in items_raw:
          fields = a.get("fields", {})
          items.append({
              "title": a.get("webTitle", ""), "url": a.get("webUrl", ""), "source": "The Guardian",
              "summary": fields.get("trailText", ""), "published": a.get("webPublicationDate", ""),
          })
      return _normalize_items(items)
  except Exception as e:
      logger.info(f"Guardian fetch failed: {e}")
      return []

def get_climate_news(query: str, window: str = "7d", limit: int = 24, source_pref: str = "all", fresh: int = 0) -> List[Dict]:
  query = (query or "").strip()
  items: List[Dict] = []
  try:
      if source_pref in ("all", "gdelt"):
          items += fetch_gdelt_news(query, window, limit, fresh=fresh)
      if source_pref in ("all", "googlenews"):
          items += fetch_google_news(query, window, limit, fresh=fresh)
      if source_pref in ("all", "newsapi"):
          items += fetch_newsapi(query, window, limit, fresh=fresh)
      if source_pref in ("all", "guardian"):
          items += fetch_guardian(query, window, limit, fresh=fresh)
  except Exception as e:
      logger.info(f"News aggregation error: {e}")

  def _sort_key(x):
      ts = x.get("published")
      if isinstance(ts, datetime):
          return (0, ts)
      return (1, datetime.min)

  items = _normalize_items(items)
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

def render_news_card(item: Dict):
  title_txt = escape(item.get("title", ""))
  url = item.get("url", "")
  source = item.get("source", "") or _domain(url)
  pub = item.get("published")
  when = _fmt_time(pub) if isinstance(pub, datetime) else ""
  summary_txt = item.get("summary", "")

  st.markdown(
      f"""
<div style="background: white; padding: 14px 16px; border-radius: 12px; margin: 8px 0; border-left: 4px solid #2ecc71;">
  <div style="display:flex; justify-content:space-between; gap:12px;">
    <div style="flex:1; min-width:0;">
      <a href="{url}" target="_blank" style="text-decoration:none;"><strong>{title_txt}</strong></a><br>
      <small style="color:#666;">{source}{(" · " + when) if when else ""}</small>
      {f"<div style='margin-top:6px;color:#333;'>{escape(summary_txt)}</div>" if summary_txt else ""}
    </div>
    <div style="text-align:right; white-space:nowrap;">
      <a href="{url}" target="_blank" style="text-decoration:none;">Open →</a>
    </div>
  </div>
</div>
""",
      unsafe_allow_html=True
  )

# ---------------------------------------------------------
# AQI conversion breakpoints (EPA PM2.5) + state mapping
# ---------------------------------------------------------
AQI_BREAKPOINTS_PM25 = [
    (0.0, 12.0,   0,  50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4,101, 150),
    (55.5,150.4,151, 200),
    (150.5,250.4,201,300),
    (250.5,350.4,301,400),
    (350.5,500.4,401,500),
]
STATE_NAME_TO_USPS = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
    "Connecticut":"CT","Delaware":"DE","District of Columbia":"DC","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
    "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE",
    "Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
    "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
    "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","Puerto Rico":"PR"}

def _pm25_to_aqi(pm25: float) -> int:
    if pm25 is None:
        return 0
    x = float(pm25)
    for c_low, c_high, aqi_low, aqi_high in AQI_BREAKPOINTS_PM25:
        if c_low <= x <= c_high:
            return int(round((aqi_high - aqi_low) / (c_high - c_low) * (x - c_low) + aqi_low))
    return 500

# ---------- OpenAQ state map fetch ----------
OPENAQ_SESSION = requests.Session()
OPENAQ_SESSION.headers.update({"User-Agent": "Climatrack/1.0 (Streamlit app)"})

@st.cache_data(ttl=10 * 60, show_spinner=False)
def fetch_openaq_state_pm25() -> pd.DataFrame:
    """Latest 24–48h PM2.5 from OpenAQ aggregated to state level."""
    def _get_df(hours_back: int, max_pages: int = 6) -> pd.DataFrame:
        date_from = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat(timespec="seconds")
        page = 1
        recs = []
        total_results = 0
        while page <= max_pages:
            params = {"country":"US","parameter":"pm25","date_from":date_from,"limit":10000,"page":page,"order_by":"date","sort":"desc"}
            try:
                r = OPENAQ_SESSION.get("https://api.openaq.org/v2/measurements", params=params, timeout=25)
                r.raise_for_status()
                js = r.json()
            except Exception as e:
                logger.info(f"OpenAQ measurements fetch failed on page {page}: {e}")
                break
            results = js.get("results", []) or []
            total_results += len(results)
            if not results: break
            for m in results:
                state_name = (m.get("state") or "").strip()
                code = STATE_NAME_TO_USPS.get(state_name)
                if not code: continue
                try:
                    val = float(m.get("value"))
                    ts_raw = (m.get("date") or {}).get("utc") or ""
                    if not ts_raw: continue
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    continue
                recs.append({"state": code, "pm25": val, "ts": ts})
            meta = js.get("meta", {})
            found = int(meta.get("found", 0))
            limit = int(meta.get("limit", 10000))
            page_count = (found // limit) + (1 if found % limit else 0)
            if page >= page_count: break
            page += 1
        if DEBUG_AQI:
            st.write(f"OpenAQ measurements fetched: {total_results} rows (hours_back={hours_back})")
        if not recs:
            return pd.DataFrame(columns=["state","pm25","aqi","updated_at"])
        df = pd.DataFrame(recs)
        df = (df.sort_values("ts")
                .groupby("state")
                .agg(pm25=("pm25","median"), updated_at=("ts","max"))
                .reset_index())
        df["aqi"] = df["pm25"].apply(_pm25_to_aqi).astype(int)
        df = df[df["state"].str.match(r"^[A-Z]{2}$", na=False)]
        return df[["state","pm25","aqi","updated_at"]]
    df = _get_df(24)
    if df.empty: df = _get_df(48)
    if DEBUG_AQI and not df.empty:
        st.dataframe(df.sort_values("updated_at", ascending=False).head(15))
    return df

def render_us_aqi_map(aqi_df: pd.DataFrame):
    if aqi_df.empty:
        return px.choropleth()
    aqi_scale = [(0.00,"#00e400"),(0.20,"#ffff00"),(0.40,"#ff7e00"),(0.60,"#ff0000"),(0.80,"#8f3f97"),(1.00,"#7e0023")]
    fig = px.choropleth(
        aqi_df, locations="state", color="aqi", locationmode="USA-states", scope="usa",
        color_continuous_scale=aqi_scale, range_color=(0,300),
        hover_data={"pm25":":.1f","aqi":":d","state":True},
        labels={"aqi":"AQI","pm25":"PM₂․₅ (µg/m³)"},
        title="U.S. PM₂․₅ → AQI (OpenAQ, latest ≤24–48h)"
    )
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
    return fig

# ---------------------------------------------------------
# Open-Meteo Risk tab helpers (Geocoding + Forecast + Archive)
# ---------------------------------------------------------
OPENMETEO = requests.Session()
OPENMETEO.headers.update({"User-Agent": "Climatrack/1.0 (Streamlit app)"})

@st.cache_data(ttl=60 * 60, show_spinner=False)
def geocode_place(q: str) -> Optional[dict]:
    q = (q or "").strip()
    if not q: return None
    try:
        r = OPENMETEO.get("https://geocoding-api.open-meteo.com/v1/search",
                          params={"name": q, "count": 1, "language":"en","format":"json"}, timeout=20)
        r.raise_for_status()
        js = r.json()
        res = (js.get("results") or [None])[0]
        if res:
            return {"name":res.get("name"),"lat":float(res["latitude"]),"lon":float(res["longitude"]),
                    "admin":res.get("admin1"),"country":res.get("country")}
    except Exception:
        pass
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q":q,"format":"json","limit":1},
                         headers={"User-Agent":"Climatrack/1.0"}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data:
            d = data[0]
            return {"name":d.get("display_name", q),"lat":float(d["lat"]),"lon":float(d["lon"]),
                    "admin":None,"country":None}
    except Exception as e:
        logger.info(f"Fallback geocode failed: {e}")
    return None

@st.cache_data(ttl=15 * 60, show_spinner=False)
def openmeteo_air_quality(lat: float, lon: float) -> pd.DataFrame:
    r = OPENMETEO.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                      params={"latitude":lat,"longitude":lon,"hourly":"pm2_5","timezone":"UTC","forecast_days":4},
                      timeout=20)
    r.raise_for_status()
    js = r.json()
    times = js.get("hourly", {}).get("time", []) or []
    pm25 = js.get("hourly", {}).get("pm2_5", []) or []
    df = pd.DataFrame({"ts": pd.to_datetime(times), "pm25": pm25})
    df["aqi"] = df["pm25"].apply(_pm25_to_aqi)
    return df

@st.cache_data(ttl=15 * 60, show_spinner=False)
def openmeteo_weather(lat: float, lon: float) -> pd.DataFrame:
    r = OPENMETEO.get("https://api.open-meteo.com/v1/forecast",
                      params={"latitude":lat,"longitude":lon,
                              "hourly":"temperature_2m,relative_humidity_2m,apparent_temperature",
                              "timezone":"UTC","forecast_days":4}, timeout=20)
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

@st.cache_data(ttl=60 * 60, show_spinner=False)
def openmeteo_recent_baseline(lat: float, lon: float) -> float:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=35)
    r = OPENMETEO.get("https://archive-api.open-meteo.com/v1/archive",
                      params={"latitude":lat,"longitude":lon,"start_date":start.isoformat(),
                              "end_date":end.isoformat(),"daily":"temperature_2m_max","timezone":"UTC"},
                      timeout=20)
    r.raise_for_status()
    js = r.json()
    daily = js.get("daily", {})
    vals = daily.get("temperature_2m_max", []) or []
    if not vals: return float("nan")
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
    if hi_f < 80: return 0.0
    if hi_f < 90: return 20.0 * (hi_f - 80) / 10.0
    if hi_f < 103: return 20 + 30.0 * (hi_f - 90) / 13.0
    if hi_f < 124: return 50 + 30.0 * (hi_f - 103) / 21.0
    return 100.0

def score_from_temp_anomaly(anom_c: float) -> float:
    if anom_c <= 0: return 0.0
    if anom_c >= 8.0: return 100.0
    return 12.5 * anom_c

def composite_risk(aqi_score: float, hi_score: float, anom_score: float) -> float:
    return round(0.45 * aqi_score + 0.35 * hi_score + 0.20 * anom_score, 1)

# ---------------------------------------------------------
# Progress logging (persisted to data/footprint_log.csv) — stored in kg internally
# ---------------------------------------------------------
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "footprint_log.csv"

def _load_log() -> pd.DataFrame:
  if LOG_PATH.exists():
      try:
          return pd.read_csv(LOG_PATH, parse_dates=["ts"])
      except Exception:
          return pd.DataFrame(columns=["ts", "kg"])
  return pd.DataFrame(columns=["ts", "kg"])

def _save_log(df: pd.DataFrame) -> None:
  df.sort_values("ts", inplace=True)
  df.to_csv(LOG_PATH, index=False)

def record_footprint(kg_total: float) -> None:
  df = _load_log()
  new_row = pd.DataFrame([{"ts": pd.Timestamp.utcnow(), "kg": float(kg_total)}])
  df = pd.concat([df, new_row], ignore_index=True)
  _save_log(df)

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🌍 Climatrack")
st.markdown("### Calculate your carbon footprint with AI-powered analysis")

# Tabs (Risk before Settings)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Calculator", "📊 Analysis", "📈 Insights",
    "📰 News", "👥 Community", "⚠️ Risk", "⚙️ Settings"
])

with tab1:
  st.markdown("### 🎯 Your Carbon Footprint Dashboard")
  col1, col2, col3 = st.columns(3)
  with col1:
      st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌱 Daily Goal</div>
            <div class="metric-value" style="color: #2ecc71;">{kg_to_lbs(5.0):.1f}</div>
            <div>lb CO₂</div>
            <div class="status-badge status-success">Sustainable Target</div>
        </div>
        """, unsafe_allow_html=True)
  with col2:
      st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Global Average</div>
            <div class="metric-value" style="color: #3498db;">{kg_to_lbs(16.0):.1f}</div>
            <div>lb CO₂ per day</div>
            <div class="status-badge status-warning">Above Target</div>
        </div>
        """, unsafe_allow_html=True)
  with col3:
      st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Target 2050</div>
            <div class="metric-value" style="color: #e74c3c;">{kg_to_lbs(2.0):.1f}</div>
            <div>lb CO₂ per day</div>
            <div class="status-badge status-info">Climate Goal</div>
        </div>
        """, unsafe_allow_html=True)

  st.markdown("### 🤖 AI-Powered Analysis")
  user_input = st.text_area(
      "Describe your daily routine:",
      placeholder="Example: I drove 10 miles to work, used 5 kWh of electricity, and ate chicken for dinner...",
      height=120
  )
  goal_lbs = st.slider("Set your daily carbon footprint goal (lb CO₂)", 2.0, 44.0, kg_to_lbs(5.0), 0.5)
  goal_kg = goal_lbs / LBS_PER_KG

  if st.button("🚀 Calculate My Footprint", use_container_width=True):
      if user_input:
          with st.spinner("🤖 AI is analyzing your input..."):
              classification_prompt = f"""Is this text describing a daily routine or daily activities?

Text: {user_input}

Answer with ONLY: DAILY ROUTINE or NOT DAILY ROUTINE"""
              classification = llm_complete(classification_prompt, max_new_tokens=8, temp=0.0).strip()
              if "DAILY ROUTINE" not in classification.upper():
                  st.error("❌ I can't help you with that. Please describe your daily routine with specific activities.")
              else:
                  analysis_prompt = f"""Analyze this daily routine and explain step-by-step how each action affects the environment and climate:

Daily Routine: {user_input}

Provide:
1. Breakdown of each activity
2. Environmental impact (pos/neg) for each
3. Estimated footprint per activity
4. Suggestions to reduce impact
5. Overall assessment (clear bullets)"""
                  detailed = llm_complete(analysis_prompt, max_new_tokens=520, temp=0.6)
                  notice("🤖 AI Analysis Complete!", "success")
                  st.markdown("### 📊 Step-by-Step Environmental Impact Analysis")
                  st.markdown(detailed)

      distance_kg = len(re.findall(r'\d+\s*(?:miles?|km)', user_input.lower())) * 2
      electricity_kg = len(re.findall(r'\d+\s*(?:kwh|kilowatt)', user_input.lower())) * 1.5
      meat_kg = len(re.findall(r'\b(?:meat|chicken|beef|pork)\b', user_input.lower())) * 2.5
      total_kg = distance_kg + electricity_kg + meat_kg

      record_footprint(total_kg)

      total_lbs = kg_to_lbs(total_kg)
      if total_kg <= goal_kg:
          st.balloons()
          st.success(f"🎉 You're under your goal of {goal_lbs:.1f} lb CO₂!")
      else:
          notice(f"⚠️ You're {kg_to_lbs(total_kg - goal_kg):.1f} lb CO₂ above your goal.", "warning")

      c1, c2, c3 = st.columns(3)
      with c1: st.metric("🚗 Transport", f"{kg_to_lbs(distance_kg):.1f} lb CO₂")
      with c2: st.metric("⚡ Electricity", f"{kg_to_lbs(electricity_kg):.1f} lb CO₂")
      with c3: st.metric("🍖 Food", f"{kg_to_lbs(meat_kg):.1f} lb CO₂")

with tab2:
  st.markdown("---")
  st.markdown("#### 📊 Emission Categories")
  st.info("**Scope 1:** Direct emissions from vehicles and heating")
  st.info("**Scope 2:** Indirect emissions from purchased electricity, steam, or cooling")
  st.info("**Scope 3:** All other indirect emissions (supply chain, services, etc.)")

  st.markdown("---")
  st.markdown("#### ⚡ Region-Aware “What-If” Optimizer")
  location_setting = st.session_state.get("location", "United States")
  units_setting = UNITS_DEFAULT  

  zip_code = st.text_input("ZIP code (US) — optional, improves accuracy", value="")
  grid_kg, grid_label, is_fallback = get_grid_factor(zip_code, location_setting)
  grid_badge = f"**Grid factor:** {grid_kg:.3f} kg CO₂/kWh  ·  {grid_label}"
  notice(grid_badge, "warning" if is_fallback else "success")

  cA, cB = st.columns(2)
  with cA:
      kwh_day = st.number_input("Daily electricity use (kWh)", min_value=0.0, value=12.0, step=0.5)
      meals = st.number_input("High-impact meals per day", min_value=0.0, value=1.0, step=0.5)
      meal_now = st.selectbox("Current meal type", list(FOOD_KG_PER_SERVING.keys()), index=0)
  with cB:
      commute_miles = st.number_input("Commute miles per day", min_value=0.0, value=10.0, step=1.0)
      commute_mode = st.selectbox("Commute mode (current)", ["Gasoline car", "Hybrid", "EV", "Bus", "Rail", "Bike/Walk"], index=0)
      ev_eff = st.slider("EV efficiency (kWh/mi)", 0.15, 0.45, 0.30, 0.01)

  st.markdown("##### Try changes")
  o1, o2, o3 = st.columns(3)
  with o1:
      eff_reduction = st.slider("Reduce electricity use (%)", 0, 50, 20, 1)
  with o2:
      target_mode = st.selectbox("Switch commute to", ["(no change)", "Hybrid", "EV", "Bus", "Rail", "Bike/Walk"], index=0)
  with o3:
      meal_swap = st.selectbox("Swap meal to", ["(no change)", "Chicken", "Vegetarian"], index=0)

  baseline = compute_scenario(grid_kg, kwh_day, commute_miles, commute_mode, meals, meal_now, ev_kwh_per_mile=ev_eff)
  kwh_opt = kwh_day * (1 - eff_reduction / 100.0)
  mode_opt = commute_mode if target_mode == "(no change)" else target_mode
  meal_opt = meal_now if meal_swap == "(no change)" else meal_swap
  optimized = compute_scenario(grid_kg, kwh_opt, commute_miles, mode_opt, meals, meal_opt, ev_kwh_per_mile=ev_eff)

  base_total_disp = convert_mass_value(baseline["total"], units_setting)
  opt_total_disp = convert_mass_value(optimized["total"], units_setting)
  delta_disp = opt_total_disp - base_total_disp

  st.markdown("##### Results")
  m1, m2, m3 = st.columns(3)
  with m1: st.metric("Baseline total", f"{base_total_disp:.2f} {unit_suffix(units_setting)}")
  with m2: st.metric("Optimized total", f"{opt_total_disp:.2f} {unit_suffix(units_setting)}", delta=f"{delta_disp:.2f} {unit_suffix(units_setting)}")
  savings = baseline["total"] - optimized["total"]
  with m3: st.metric("Savings", f"{convert_mass_value(savings, units_setting):.2f} {unit_suffix(units_setting)}")

  categories = ["Electricity", "Transport", "Food"]
  base_vals = [convert_mass_value(baseline["electricity"], units_setting),
               convert_mass_value(baseline["transport"], units_setting),
               convert_mass_value(baseline["food"], units_setting)]
  opt_vals = [convert_mass_value(optimized["electricity"], units_setting),
              convert_mass_value(optimized["transport"], units_setting),
              convert_mass_value(optimized["food"], units_setting)]
  x = list(range(len(categories)))
  width = 0.38
  fig, ax = plt.subplots(figsize=(7, 4))
  ax.bar([i - width/2 for i in x], base_vals, width, label="Current")
  ax.bar([i + width/2 for i in x], opt_vals, width, label="Optimized", alpha=0.8)
  ax.set_xticks(x); ax.set_xticklabels(categories)
  ax.set_ylabel(unit_suffix(units_setting) + " / day")
  ax.set_title("Current vs Optimized")
  ax.legend(); ax.grid(True, axis="y", alpha=0.25)
  st.pyplot(fig, clear_figure=True)

  action_deltas = [
      ("Electricity efficiency", baseline['electricity'] - optimized['electricity']),
      (f"Commute: {commute_mode} → {mode_opt}", baseline['transport'] - optimized['transport']),
      (f"Meal: {meal_now} → {meal_opt}", baseline['food'] - optimized['food']),
  ]
  action_deltas.sort(key=lambda x: x[1], reverse=True)
  st.markdown("##### Top actions to hit your goal")
  for name, kg in action_deltas:
      if abs(kg) < 1e-6:
          continue
      st.write(f"- **{name}**: save **{convert_mass_value(kg, units_setting):.2f} {unit_suffix(units_setting)}/day**")

with tab3:
  st.markdown("### 📈 Your Carbon Journey")
  df_hist = _load_log()
  if df_hist.empty:
      st.info("No history yet. Run a calculation in the **Calculator** tab to start tracking.")
  else:
      df_hist_lbs = df_hist.copy()
      df_hist_lbs["lbs"] = df_hist_lbs["kg"] * LBS_PER_KG
      daily = (df_hist_lbs.set_index("ts").resample("D")["lbs"].mean().dropna())
      last_30 = daily.last("30D")

      if last_30.empty:
          st.info("No entries in the last 30 days yet—keep going!")
      else:
          fig, ax = plt.subplots(figsize=(10, 5))
          ax.plot(last_30.index, last_30.values, marker='o', linewidth=3)
          ax.set_ylabel("lb CO₂ per day"); ax.set_title("Your Carbon Footprint (last 30 days)")
          ax.grid(True, alpha=0.3)
          st.pyplot(fig)

      last_7 = daily.last("7D").mean() if not daily.empty else float("nan")
      prev_7 = daily.iloc[:-7].last("7D").mean() if len(daily) > 7 else float("nan")
      delta_pct = (100.0 * (last_7 - prev_7) / prev_7) if (pd.notna(last_7) and pd.notna(prev_7) and prev_7 != 0) else None

      c1, c2, c3 = st.columns(3)
      with c1: st.metric("7-day average", f"{(last_7 if pd.notna(last_7) else 0):.2f} lb CO₂")
      with c2: st.metric("Previous 7-day avg", f"{(prev_7 if pd.notna(prev_7) else 0):.2f} lb CO₂")
      with c3: st.metric("Change vs prev week", "—" if delta_pct is None else f"{delta_pct:+.1f} %", delta=None)

      st.markdown("#### Data")
      export = df_hist.copy()
      export["lbs"] = export["kg"] * LBS_PER_KG
      csv_bytes = export[["ts","lbs"]].rename(columns={"lbs":"lb_CO2"}).to_csv(index=False).encode("utf-8")
      st.download_button("⬇️ Export log (CSV, lb CO₂)", data=csv_bytes, file_name="footprint_log_lbs.csv", mime="text/csv")

      with st.expander("Maintenance"):
          colA, colB = st.columns(2)
          with colA:
              if st.button("↻ Recompute daily averages"): st.rerun()
          with colB:
              if st.button("🗑️ Clear all history", type="secondary"):
                  _save_log(pd.DataFrame(columns=["ts", "kg"]))
                  st.success("History cleared."); st.rerun()

with tab4:
  st.markdown("### 📰 Climate & Environment News")
  auto_count = st_autorefresh(interval=2 * 60 * 1000, key="news_auto_refresh") or 0

  c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
  with c1:
      news_query = st.text_input("Optional filter (e.g., wildfire, heatwave, policy)", key="news_query")
  with c2:
      news_window = st.selectbox("Time window", ["24h", "7d", "30d"], index=1, key="news_window")
  with c3:
      source_pref = st.selectbox("Source",
          ["All (aggregated)", "GDELT", "Google News", "NewsAPI", "The Guardian"], index=0, key="news_source")
  with c4:
      if st.button("🔄 Refresh now", use_container_width=True):
          st.session_state["news_force_refresh"] = st.session_state.get("news_force_refresh", 0) + 1
  force_refresh = st.session_state.get("news_force_refresh", 0)

  pref_map = {"All (aggregated)": "all", "GDELT": "gdelt", "Google News": "googlenews", "NewsAPI": "newsapi", "The Guardian": "guardian"}
  fresh_value = int(auto_count) + int(force_refresh)
  with st.spinner("Fetching climate news..."):
      news_items = get_climate_news(query=news_query or "", window=news_window, limit=24,
                                    source_pref=pref_map[source_pref], fresh=fresh_value)

  if not news_items:
      notice("No news found right now. Try a different time window or keyword.", "warning")
  else:
      st.caption(f"Last checked: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  ·  Auto-refresh: 2 min")
      for item in news_items:
          render_news_card(item)

with tab5:
  st.markdown("### 🔎 You’re Not Alone — Real Stories")
  issue_text = st.text_area(
      "Describe your climate-related problem (be specific):",
      key="community_issue",
      placeholder="Ex: Can't breathe properly during wildfire smoke in Queens; looking for masks/filters/indoor tips.",
      height=120
  )
  if st.button("🔍 Find similar stories", key="find_similar_btn", use_container_width=True):
      if not issue_text.strip():
          st.error("Write a sentence or two about your issue first.")
      else:
          with st.spinner("Planning search with AI, fetching Reddit, and filtering..."):
              plan = ai_build_search_plan(issue_text)
              queries = plan["queries"]
              subs = plan["subreddits"] or None
              time_window = plan["time_window"]
              must_terms = plan["must_terms"]
              posts = []
              for q in queries:
                  posts += reddit_search(q, limit=12, time_window=time_window, subs=subs)
              if len(posts) < 6:
                  for q in queries:
                      posts += reddit_search(q, limit=8, time_window=time_window, subs=None)
              filtered, seen = [], set()
              for p in posts:
                  if not p.get("id") or p["id"] in seen: continue
                  if ai_is_relevant(issue_text, p["title"], p["selftext"], must_terms):
                      filtered.append(p); seen.add(p["id"])
              key_terms = set(must_terms) if must_terms else set(re.findall(r"[a-z]{3,}", issue_text.lower()))
              def score_post(p):
                  base = (p["title"] + " " + p["selftext"]).lower()
                  overlap = sum(1 for w in key_terms if w in base)
                  return overlap + (p["score"] / 300.0)
              filtered.sort(key=score_post, reverse=True)
              if not filtered:
                  notice("No close matches found. Try adding the most important detail (e.g., location, AQI, smoke/pollen/heat).", "warning")
              else:
                  summary = ai_summarize_findings(issue_text, filtered[:8])
                  if summary:
                      st.success("### What others did that helped")
                      st.markdown(summary)
                  st.markdown("### How others described similar issues")
                  for p in filtered[:12]:
                      created_str = p["created"].strftime("%Y-%m-%d")
                      snippet = (p["selftext"] or "").strip()
                      if len(snippet) > 220: snippet = snippet[:220].rstrip() + "…"
                      st.markdown(
                          f"""
<div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #2ecc71;">
 <div style="display:flex; justify-content:space-between; align-items:center;">
   <div style="max-width:75%;">
     <a href="{p['url']}" target="_blank"><strong>{p['title']}</strong></a><br>
     <small style="color:#666;">r/{p['subreddit']} · u/{p['author']} · {created_str} · {p['score']} upvotes</small>
     <div style="margin-top:6px;color:#333;">{snippet}</div>
   </div>
   <div style="text-align:right;">
     <a href="{p['url']}" target="_blank" style="text-decoration:none;">Open thread →</a>
   </div>
 </div>
</div>
""",
                          unsafe_allow_html=True
                      )
  st.caption("Sources are public Reddit posts. We display links and short snippets only.")

# --------------------- RISK TAB ---------------------
with tab6:
  st.markdown("### ⚠️ Climate Risk Dashboard")
  st.caption("Free data: Open-Meteo Air Quality & Forecast. Composite risk score from AQI, heat index, and temperature anomaly.")

  cols = st.columns([3,2])
  with cols[0]:
    place = st.text_input("City or place", value="", placeholder="e.g., Queens, NY or 40.728,-73.794")
  with cols[1]:
    horizon = st.slider("Forecast horizon (hours)", 24, 96, 72, 1)

  if not place.strip():
      notice("Enter a city, ZIP, or `lat,lon` then press Enter.", "info")
  else:
      lat = lon = None
      m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", place)
      if m:
          lat, lon = float(m.group(1)), float(m.group(2))
          geo = {"name": place, "lat": lat, "lon": lon}
      else:
          geo = geocode_place(place)
          if geo:
              lat, lon = geo["lat"], geo["lon"]

      if not (lat and lon):
          notice("Couldn't geocode that place. Try a ZIP code or paste coordinates like `40.728,-73.794`.", "warning")
      else:
          with st.spinner("Fetching live air quality & weather…"):
              try:
                  df_aq = openmeteo_air_quality(lat, lon)
                  df_wx = openmeteo_weather(lat, lon)
                  baseline_max_c = openmeteo_recent_baseline(lat, lon)
              except Exception as e:
                  df_aq = df_wx = pd.DataFrame()
                  baseline_max_c = float("nan")
                  notice(f"Data fetch failed: {e}", "error")

          if df_aq.empty or df_wx.empty:
              notice("No forecast data returned for this location.", "warning")
          else:
              df = pd.merge_asof(df_aq.sort_values("ts"), df_wx.sort_values("ts"), on="ts")
              now_utc = pd.Timestamp.utcnow().tz_localize(None)
              df = df[df["ts"] >= now_utc].head(horizon)

              if "rh" in df.columns and df["rh"].notna().any():
                  df["hi_c"] = [heat_index_c(t, rh) for t, rh in zip(df["temp_c"].fillna(method="ffill"),
                                                                    df["rh"].fillna(method="ffill"))]
              else:
                  df["hi_c"] = df.get("apparent_c", df["temp_c"])

              df["aqi_score"] = df["aqi"].apply(score_from_aqi)
              df["hi_score"] = df["hi_c"].apply(score_from_heatindex_c)

              forecast_max_c = float(df["temp_c"].max()) if not df["temp_c"].isna().all() else float("nan")
              anom_c = (forecast_max_c - baseline_max_c) if pd.notna(baseline_max_c) else 0.0
              anom_score = score_from_temp_anomaly(anom_c)

              peak_aqi = int(df["aqi"].max())
              peak_hi_f = float(df["hi_c"].max() * 9/5 + 32)
              aqi_score = float(df["aqi_score"].max())
              hi_score = float(df["hi_score"].max())
              total_risk = composite_risk(aqi_score, hi_score, anom_score)

              m1, m2, m3, m4 = st.columns(4)
              with m1: st.metric("Composite Risk (0–100)", f"{total_risk:.1f}")
              with m2: st.metric("Peak AQI (next hrs)", f"{peak_aqi}")
              with m3: st.metric("Peak Heat Index", f"{peak_hi_f:.0f} °F")
              with m4: st.metric("Temp Anomaly vs 30-day max", f"{anom_c:+.1f} °C")

              fig1 = px.line(df, x="ts", y="aqi", title="Hourly AQI forecast", labels={"ts":"UTC time","aqi":"AQI"})
              fig1.update_layout(margin=dict(l=0,r=0,t=40,b=0))
              st.plotly_chart(fig1, use_container_width=True)

              fig2 = px.line(df, x="ts", y=[(df["hi_c"]*9/5+32)], title="Hourly Heat Index (°F)",
                             labels={"ts":"UTC time","value":"Heat Index (°F)"})
              fig2.update_layout(margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
              st.plotly_chart(fig2, use_container_width=True)

              try:
                  prompt = f"""
Location: {geo.get('name','(lat,lon)')} (lat {lat:.3f}, lon {lon:.3f})
Horizon: {horizon}h
Peak AQI: {peak_aqi}
Peak Heat Index (F): {peak_hi_f:.0f}
Temp anomaly vs baseline (C): {anom_c:+.1f}
Composite risk (0-100): {total_risk:.1f}

Explain the risk in clear bullets (what to watch for, simple precautions for sensitive groups vs general public).
"""
                  txt = llm_complete(prompt, max_new_tokens=220, temp=0.3)
              except Exception:
                  txt = (
                      f"- AQI peaks at {peak_aqi}; higher values mean worse air.\n"
                      f"- Heat Index reaches ~{peak_hi_f:.0f} °F; hydrate, limit exertion in the afternoon.\n"
                      f"- Temperature runs {anom_c:+.1f} °C vs recent max; expect {('warmer' if anom_c>0 else 'cooler')} conditions.\n"
                      f"- Composite risk {total_risk:.1f}/100: adjust outdoor time and ventilation accordingly."
                  )
              st.info(txt)

with tab7:
  st.markdown("### ⚙️ Settings & Preferences")
  c1, c2 = st.columns(2)
  with c1:
      st.markdown("#### 🌍 Location")
      location = st.selectbox("Select your region:", ["United States", "European Union", "Global Average"])
      st.markdown("#### 📊 Units")
      units = st.radio("Choose your preferred units:", ["Imperial (lbs CO₂)", "Metric (kg CO₂)"], index=0)
  with c2:
      st.markdown("#### 🔔 Notifications")
      daily_reminders = st.checkbox("Daily reminders", value=True)
      weekly_reports = st.checkbox("Weekly progress reports")
      achievements = st.checkbox("Achievement alerts", value=True)
  if st.button("💾 Save Settings", use_container_width=True):
      st.session_state["location"] = location
      st.session_state["units"] = UNITS_DEFAULT
      st.success("✅ Settings saved (display is locked to lb CO₂).")

# ---------------------------------------------------------
# Environment & Climate AI Assistant (BOTTOM SECTION)
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 Environment & Climate AI Assistant")
st.markdown("Ask me anything about environment, climate change, sustainability, and related topics!")

col1, col2 = st.columns([2, 1])
with col1:
  ai_question = st.text_area(
      "Ask your environmental question:",
      placeholder="Examples: How can I reduce my carbon footprint? What causes climate change? How do solar panels work?",
      height=100,
      key="env_ai_text"
  )
  if st.button("🌿 Ask Environment AI", use_container_width=True, key="env_ai_btn"):
      if ai_question.strip():
          with st.spinner("⏳ Environment AI is thinking..."):
              def ask_environment_ai(question: str) -> dict:
                try:
                    q = (question or "").strip()[:1500]
                    if not q:
                        return {"allowed": False, "reply": "Please enter an environmental question."}
                    classification_prompt = (
                        f"Question: {q}\n"
                        "Is this about environment, climate change, sustainability, energy, air/water, ecology, or related? "
                        "Answer ONLY YES or NO."
                    )
                    is_env = "YES" in llm_complete(classification_prompt, max_new_tokens=6, temp=0.0).upper()
                    if not is_env:
                        return {"allowed": False, "reply": "I can only answer environment/climate related questions here."}
                    answer_prompt = f"Question: {q}\nProvide a clear, practical answer in 4–7 sentences for a general audience."
                    answer = llm_complete(answer_prompt, max_new_tokens=320, temp=0.4).strip()
                    if not answer or len(answer) < 10:
                        return {"allowed": True, "reply": "I need a little more detail to give a useful answer."}
                    return {"allowed": True, "reply": answer}
                except Exception as e:
                    return {"allowed": False, "reply": f"Something went wrong: {e}"}
              result = ask_environment_ai(ai_question.strip())
          if result.get("allowed"):
              st.success("✅ Environment AI Response:")
              st.info(result.get("reply", ""))
          else:
              msg = result.get("reply", "I can't answer that.")
              st.error(msg if "can't answer" in msg.lower() else f"⚠️ {msg}")
      else:
          notice("Please enter a question about environment or climate change.", "warning")

with col2:
  st.markdown("""""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
  <p>🌍 Made with ❤️ for the planet | Powered by AI | Built with Streamlit</p>
  <p>Version 3.0 | Last updated: September 2024</p>
</div>
""", unsafe_allow_html=True)
