#!/usr/bin/env python3
"""
Climatrack Flask App - AI Climate Impact Calculator
Run with: python app_flask.py
"""
from flask import Flask, render_template, request, jsonify, session
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Import core logic (will create this next)
try:
    import core
except ImportError:
    print("Warning: core.py not found. Some features may not work.")
    core = None

@app.route("/")
def index():
    """Main page"""
    session.setdefault("location", "United States")
    session.setdefault("units", "Imperial (lbs CO₂)")
    return render_template("index.html",
        goal_lbs=5.0 * 2.20462262,  # ~11 lb
        avg_lbs=16.0 * 2.20462262,  # ~35.3 lb
        target_lbs=2.0 * 2.20462262,  # ~4.4 lb
        location=session.get("location", "United States"),
        units=session.get("units", "Imperial (lbs CO₂)"),
    )

@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    """Calculate carbon footprint"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        user_input = (data.get("user_input") or "").strip()
        goal_lbs = float(data.get("goal_lbs") or 11.0)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    
    result = core.run_calculator(user_input, goal_lbs)
    return jsonify({
        "distance_lbs": round(result.get("distance_kg", 0) * 2.20462262, 1),
        "electricity_lbs": round(result.get("electricity_kg", 0) * 2.20462262, 1),
        "food_lbs": round(result.get("meat_kg", 0) * 2.20462262, 1),
        "total_lbs": round(result.get("total_lbs", 0), 1),
        "goal_lbs": result.get("goal_lbs", goal_lbs),
        "under_goal": result.get("under_goal", False),
        "ai_analysis": result.get("ai_analysis"),
        "classification_error": result.get("classification_error"),
    })

@app.route("/api/news", methods=["POST"])
def api_news():
    """Fetch climate news"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        query = (data.get("query") or "").strip()
        window = data.get("window") or "7d"
    except Exception:
        query, window = "", "7d"
    
    items = core.get_climate_news(query=query, window=window, limit=24, source_pref="all", fresh=1)
    from datetime import datetime as dt
    out = []
    for it in items:
        pub = it.get("published")
        when = core._fmt_time(pub) if isinstance(pub, dt) else ""
        out.append({
            "title": it.get("title", ""),
            "url": it.get("url", ""),
            "source": it.get("source", ""),
            "summary": it.get("summary", ""),
            "when": when,
        })
    return jsonify({"items": out})

@app.route("/api/news/opinions", methods=["POST"])
def api_news_opinions():
    """Get AI agent opinions on news article"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        title = (data.get("title") or "").strip()
        summary = (data.get("summary") or "").strip()
        url = (data.get("url") or "").strip()
    except Exception:
        return jsonify({"error": "Invalid input", "opinions": []}), 400
    
    if not title and not summary:
        return jsonify({"error": "Provide at least title or summary", "opinions": []}), 400
    
    try:
        opinions = core.get_news_agent_opinions(title=title, summary=summary, url=url)
        return jsonify({"opinions": opinions})
    except Exception as e:
        return jsonify({"error": str(e), "opinions": []}), 500

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Region-aware optimizer analysis"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        location = session.get("location", "United States")
        zip_code = (data.get("zip_code") or "").strip()
        grid_kg, grid_label, _ = core.get_grid_factor(zip_code, location)
        kwh_day = float(data.get("kwh_day") or 12.0)
        meals = float(data.get("meals") or 1.0)
        meal_now = data.get("meal_now") or "Chicken"
        commute_miles = float(data.get("commute_miles") or 10.0)
        commute_mode = data.get("commute_mode") or "Gasoline car"
        ev_eff = float(data.get("ev_eff") or 0.30)
        eff_reduction = float(data.get("eff_reduction") or 20.0)
        target_mode = data.get("target_mode") or "(no change)"
        meal_swap = data.get("meal_swap") or "(no change)"
        
        baseline = core.compute_scenario(grid_kg, kwh_day, commute_miles, commute_mode, meals, meal_now, ev_kwh_per_mile=ev_eff)
        kwh_opt = kwh_day * (1 - eff_reduction / 100.0)
        mode_opt = commute_mode if target_mode == "(no change)" else target_mode
        meal_opt = meal_now if meal_swap == "(no change)" else meal_swap
        optimized = core.compute_scenario(grid_kg, kwh_opt, commute_miles, mode_opt, meals, meal_opt, ev_kwh_per_mile=ev_eff)
        
        base_total_lbs = core.kg_to_lbs(baseline["total"])
        opt_total_lbs = core.kg_to_lbs(optimized["total"])
        savings_lbs = base_total_lbs - opt_total_lbs
        
        action_deltas = [
            ("Electricity efficiency", baseline['electricity'] - optimized['electricity']),
            (f"Commute: {commute_mode} → {mode_opt}", baseline['transport'] - optimized['transport']),
            (f"Meal: {meal_now} → {meal_opt}", baseline['food'] - optimized['food']),
        ]
        action_deltas.sort(key=lambda x: x[1], reverse=True)
        top_actions = [{"name": name, "savings": core.kg_to_lbs(kg)} for name, kg in action_deltas if abs(kg) > 1e-6]
        
        return jsonify({
            "baseline_total": base_total_lbs,
            "optimized_total": opt_total_lbs,
            "savings": savings_lbs,
            "top_actions": top_actions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/insights", methods=["GET"])
def api_insights():
    """Get progress tracking insights"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        df_hist = core.load_footprint_log()
        if df_hist.empty:
            return jsonify({"has_data": False})
        df_hist_lbs = df_hist.copy()
        df_hist_lbs["lbs"] = df_hist_lbs["kg"] * core.LBS_PER_KG
        import pandas as pd
        daily = (df_hist_lbs.set_index("ts").resample("D")["lbs"].mean().dropna())
        last_7 = daily.last("7D").mean() if not daily.empty else float("nan")
        prev_7 = daily.iloc[:-7].last("7D").mean() if len(daily) > 7 else float("nan")
        delta_pct = (100.0 * (last_7 - prev_7) / prev_7) if (pd.notna(last_7) and pd.notna(prev_7) and prev_7 != 0) else None
        return jsonify({
            "has_data": True,
            "last_7_avg": float(last_7) if pd.notna(last_7) else 0.0,
            "prev_7_avg": float(prev_7) if pd.notna(prev_7) else 0.0,
            "change_pct": float(delta_pct) if delta_pct is not None else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/community", methods=["POST"])
def api_community():
    """Find similar Reddit stories"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        issue_text = (data.get("issue_text") or "").strip()
        if not issue_text:
            return jsonify({"error": "Write a sentence or two about your issue first.", "posts": []}), 400
        
        # Deep analysis: understand the problem and get targeted search plan
        analysis = core.ai_deep_analyze_problem(issue_text)
        queries = analysis["queries"]
        subs = analysis["subreddits"] or None
        time_window = analysis.get("time_window", "year")
        must_terms = analysis.get("must_terms", [])
        
        # Run AI solution and Reddit fetch in parallel
        ai_solution = [None]
        def build_solution():
            ai_solution[0] = core.ai_generate_problem_solution(issue_text, analysis)
        
        solution_thread = threading.Thread(target=build_solution)
        solution_thread.start()
        
        # Fetch Reddit in parallel (all queries at once)
        posts = []
        with ThreadPoolExecutor(max_workers=min(6, len(queries) + 2)) as ex:
            futs = [ex.submit(core.reddit_search, q, 18, time_window, subs) for q in queries]
            for fut in as_completed(futs):
                try:
                    posts.extend(fut.result())
                except Exception:
                    pass
        if len(posts) < 6:
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(core.reddit_search, q, 12, time_window, None) for q in queries[:3]]
                for fut in as_completed(futs):
                    try:
                        posts.extend(fut.result())
                    except Exception:
                        pass
        # Dedupe by id
        seen_ids = set()
        unique_posts = []
        for p in posts:
            if p.get("id") and p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                unique_posts.append(p)
        posts = unique_posts
        
        key_terms = set(must_terms) if must_terms else set(re.findall(r"[a-z]{3,}", issue_text.lower()))
        def score_post(p):
            base = (p["title"] + " " + (p.get("selftext") or "")).lower()
            overlap = sum(1 for w in key_terms if w in base)
            return overlap + (p.get("score", 0) / 400.0)
        def term_count(p):
            base = (p["title"] + " " + (p.get("selftext") or "")).lower()
            return sum(1 for t in must_terms if t.lower() in base) if must_terms else 0
        
        # Only run LLM on a limited set: pre-sort by term overlap, cap candidates
        strict_candidates = sorted([p for p in posts if term_count(p) >= 2], key=lambda p: (term_count(p), score_post(p)), reverse=True)[:20]
        strict_ids = set()
        strictly_related = []
        for p in strict_candidates:
            if p["id"] in strict_ids:
                continue
            if core.ai_is_strongly_related(issue_text, analysis, p["title"], p.get("selftext") or "", min_terms=2):
                strictly_related.append(p)
                strict_ids.add(p["id"])
        
        somewhat_candidates = sorted([p for p in posts if p["id"] not in strict_ids and term_count(p) >= 1], key=lambda p: (term_count(p), score_post(p)), reverse=True)[:25]
        somewhat_related = []
        for p in somewhat_candidates:
            if p["id"] in strict_ids:
                continue
            if core.ai_is_somewhat_related(issue_text, analysis, p["title"], p.get("selftext") or ""):
                somewhat_related.append(p)
                strict_ids.add(p["id"])
        
        # Last-resort: if still empty, take posts with 1+ must_term (no LLM)
        if len(strictly_related) + len(somewhat_related) == 0 and posts:
            for p in sorted(posts, key=lambda p: (term_count(p), score_post(p)), reverse=True):
                if p["id"] in strict_ids:
                    continue
                if term_count(p) >= 1:
                    somewhat_related.append(p)
                    strict_ids.add(p["id"])
                    if len(somewhat_related) >= 8:
                        break
        if len(strictly_related) + len(somewhat_related) == 0 and posts:
            for p in sorted(posts, key=lambda p: (p.get("score", 0), score_post(p)), reverse=True)[:6]:
                somewhat_related.append(p)
                strict_ids.add(p["id"])
        
        strictly_related.sort(key=score_post, reverse=True)
        somewhat_related.sort(key=score_post, reverse=True)
        filtered = strictly_related + somewhat_related
        filtered = filtered[:12]
        
        solution_thread.join(timeout=5)
        ai_solution_final = ai_solution[0] if ai_solution[0] else ""
        
        summary = None
        if filtered:
            summary = core.ai_summarize_findings(issue_text, filtered[:8], skip_comments=True)
        
        out_posts = []
        n_strict = len(strictly_related)
        for i, p in enumerate(filtered[:12]):
            snippet = (p["selftext"] or "").strip()
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "…"
            tier = "strict" if i < n_strict else "related"
            out_posts.append({
                "title": p["title"],
                "url": p["url"],
                "subreddit": p["subreddit"],
                "author": p["author"],
                "created": p["created"].strftime("%Y-%m-%d") if hasattr(p["created"], "strftime") else str(p["created"]),
                "score": p["score"],
                "snippet": snippet,
                "tier": tier,
            })
        
        return jsonify({"solution": ai_solution_final, "summary": summary, "posts": out_posts})
    except Exception as e:
        return jsonify({"error": str(e), "posts": []}), 500

@app.route("/api/risk", methods=["POST"])
def api_risk():
    """Get climate risk forecast"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        place = (data.get("place") or "").strip()
        horizon = int(data.get("horizon") or 72)
        
        if not place:
            return jsonify({"error": "Enter a city, ZIP, or coordinates"}), 400
        
        lat = lon = None
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", place)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            geo = {"name": place, "lat": lat, "lon": lon}
        else:
            geo = core.geocode_place(place)
            if geo:
                lat, lon = geo["lat"], geo["lon"]
        
        if not (lat and lon):
            return jsonify({"error": "Couldn't geocode that place. Try a ZIP code or coordinates like 40.728,-73.794"}), 400
        
        df_aq = core.openmeteo_air_quality(lat, lon)
        df_wx = core.openmeteo_weather(lat, lon)
        baseline_max_c = core.openmeteo_recent_baseline(lat, lon)
        
        if df_aq.empty or df_wx.empty:
            return jsonify({"error": "No forecast data returned for this location"}), 500
        
        import pandas as pd
        df = pd.merge_asof(df_aq.sort_values("ts"), df_wx.sort_values("ts"), on="ts")
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        df = df[df["ts"] >= now_utc].head(horizon)
        
        if "rh" in df.columns and df["rh"].notna().any():
            df["hi_c"] = [core.heat_index_c(t, rh) for t, rh in zip(df["temp_c"].fillna(method="ffill"), df["rh"].fillna(method="ffill"))]
        else:
            df["hi_c"] = df.get("apparent_c", df["temp_c"])
        
        df["aqi_score"] = df["aqi"].apply(core.score_from_aqi)
        df["hi_score"] = df["hi_c"].apply(core.score_from_heatindex_c)
        
        forecast_max_c = float(df["temp_c"].max()) if not df["temp_c"].isna().all() else float("nan")
        anom_c = (forecast_max_c - baseline_max_c) if pd.notna(baseline_max_c) else 0.0
        anom_score = core.score_from_temp_anomaly(anom_c)
        
        peak_aqi = int(df["aqi"].max())
        peak_hi_f = float(df["hi_c"].max() * 9/5 + 32)
        aqi_score = float(df["aqi_score"].max())
        hi_score = float(df["hi_score"].max())
        total_risk = core.composite_risk(aqi_score, hi_score, anom_score)
        
        try:
            prompt = f"""Location: {geo.get('name','(lat,lon)')} (lat {lat:.3f}, lon {lon:.3f})
Horizon: {horizon}h
Peak AQI: {peak_aqi}
Peak Heat Index (F): {peak_hi_f:.0f}
Temp anomaly vs baseline (C): {anom_c:+.1f}
Composite risk (0-100): {total_risk:.1f}

Explain the risk in clear bullets (what to watch for, simple precautions for sensitive groups vs general public)."""
            risk_explanation = core.llm_complete(prompt, max_new_tokens=220, temp=0.3)
        except Exception:
            risk_explanation = (
                f"- AQI peaks at {peak_aqi}; higher values mean worse air.\n"
                f"- Heat Index reaches ~{peak_hi_f:.0f} °F; hydrate, limit exertion in the afternoon.\n"
                f"- Temperature runs {anom_c:+.1f} °C vs recent max; expect {('warmer' if anom_c>0 else 'cooler')} conditions.\n"
                f"- Composite risk {total_risk:.1f}/100: adjust outdoor time and ventilation accordingly."
            )
        
        return jsonify({
            "composite_risk": total_risk,
            "peak_aqi": peak_aqi,
            "peak_hi_f": peak_hi_f,
            "temp_anomaly": anom_c,
            "horizon": horizon,
            "risk_explanation": risk_explanation,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Save user settings"""
    try:
        data = request.json if request.is_json else request.form
        session["location"] = data.get("location", "United States")
        session["units"] = data.get("units", "Imperial (lbs CO₂)")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai-assistant", methods=["POST"])
def api_ai_assistant():
    """Environment & Climate AI Assistant"""
    if not core:
        return jsonify({"error": "Core module not available"}), 500
    try:
        data = request.json if request.is_json else request.form
        question = (data.get("question") or "").strip()[:1500]
        
        if not question:
            return jsonify({"allowed": False, "reply": "Please enter a climate or environment-related question."}), 400
        
        # Only accept climate and environment related questions (strict)
        classification_prompt = (
            "You are a strict filter. Only accept questions that are clearly and directly about:\n"
            "climate change, environment, sustainability, ecology, renewable energy, pollution, "
            "conservation, biodiversity, carbon footprint, recycling, green living, air/water quality, "
            "wildlife, ecosystems, or environmental policy.\n\n"
            "Reject: general knowledge, math, cooking, sports, entertainment, health (unless air quality/pollution), "
            "relationships, tech (unless green tech/energy), history, etc.\n\n"
            f"Question: \"{question}\"\n\n"
            "Is this question ONLY about climate or environment as defined above? Answer with exactly one word: YES or NO."
        )
        raw = core.llm_complete(classification_prompt, max_new_tokens=8, temp=0.0).upper().strip()
        is_env = raw.startswith("YES") or raw == "YES"
        if not is_env:
            return jsonify({
                "allowed": False,
                "reply": "Ask Climi only answers climate and environment questions. Please ask about things like climate change, sustainability, carbon footprint, recycling, green energy, or nature conservation."
            })
        
        # Generate answer
        answer_prompt = f"Question: {question}\nProvide a clear, practical answer in 4–7 sentences for a general audience."
        answer = core.llm_complete(answer_prompt, max_new_tokens=320, temp=0.4).strip()
        
        if not answer or len(answer) < 10:
            return jsonify({"allowed": True, "reply": "I need a little more detail to give a useful answer.", "articles": []})
        
        # Web search for related articles
        articles = core.search_web_articles(question, max_results=5)
        
        return jsonify({"allowed": True, "reply": answer, "articles": articles})
    except Exception as e:
        return jsonify({"allowed": False, "reply": f"Something went wrong: {e}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    app.run(host="127.0.0.1", port=port, debug=True)
