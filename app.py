# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timezone
from sklearn.pipeline import Pipeline

from features_lib import compute_single_pair_features, FEATURE_VERSION

# =========================
# 🔐 Load secrets securely
# =========================
if "FOOTBALL_DATA_API_KEY" in st.secrets and st.secrets["FOOTBALL_DATA_API_KEY"]:
    os.environ["FOOTBALL_DATA_API_KEY"] = st.secrets["FOOTBALL_DATA_API_KEY"]

if "FD_MIN_INTERVAL_SEC" in st.secrets and st.secrets["FD_MIN_INTERVAL_SEC"]:
    os.environ["FD_MIN_INTERVAL_SEC"] = str(st.secrets["FD_MIN_INTERVAL_SEC"])

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    st.error("❌ API key not found! Please set FOOTBALL_DATA_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

# =========================
# ⚽ Config
# =========================
COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1"
}

# =========================
# 📥 Fetch upcoming matches
# =========================
@st.cache_data(ttl=300)
def fetch_upcoming_matches(league_code: str):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=SCHEDULED"
    headers = {"X-Auth-Token": API_KEY}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    matches = []
    for m in data.get("matches", []):
        matches.append({
            "utcDate": m["utcDate"],
            "homeTeam": m["homeTeam"]["name"],
            "awayTeam": m["awayTeam"]["name"],
        })
    return pd.DataFrame(matches)

# =========================
# 📦 Load trained model
# =========================
def load_model(model_path):
    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):  # legacy format
        pipeline = artifact["pipeline"]
        feat_cols = artifact["feature_cols"]
        feature_version = artifact.get("feature_version", "v1")
    elif isinstance(artifact, Pipeline):  # modern format
        pipeline = artifact
        feat_cols = (
            list(pipeline.feature_names_in_)
            if hasattr(pipeline, "feature_names_in_")
            else []
        )
        feature_version = "v1"
    else:
        raise ValueError("❌ Unknown model format")

    return pipeline, feat_cols, feature_version

# =========================
# 🎛️ Streamlit UI
# =========================
st.title("⚽ Football Match Predictor")

league = st.selectbox("Select League", list(COMPETITIONS.keys()), format_func=lambda x: COMPETITIONS[x])

matches_df = fetch_upcoming_matches(league)
if matches_df.empty:
    st.warning("⚠️ No upcoming matches found.")
    st.stop()

match_strs = [
    f"{row.homeTeam} vs {row.awayTeam} ({row.utcDate})"
    for row in matches_df.itertuples()
]
match_choice = st.selectbox("Choose Match", match_strs)

# Parse chosen match
chosen = matches_df.iloc[match_strs.index(match_choice)]
home_team = chosen["homeTeam"]
away_team = chosen["awayTeam"]
ref_time = datetime.fromisoformat(chosen["utcDate"].replace("Z", "+00:00"))

model_file = st.text_input("Model file path", "best_grid_model_meta.joblib")
if not os.path.exists(model_file):
    st.error(f"❌ Model file not found: {model_file}")
    st.stop()

if st.button("🔮 Predict"):
    # Load data (historic matches CSV must exist locally)
    if not os.path.exists("matches_data.csv"):
        st.error("❌ matches_data.csv not found. Please provide historical data.")
        st.stop()

    matches_hist = pd.read_csv("matches_data.csv")
    pipeline, feat_cols, feature_version = load_model(model_file)

    if feature_version != FEATURE_VERSION:
        st.warning(f"⚠️ Model feature version ({feature_version}) differs from current ({FEATURE_VERSION}).")

    # Compute features
    X, meta = compute_single_pair_features(
        matches=matches_hist,
        competition=league,
        home_team_input=home_team,
        away_team_input=away_team,
        ref_datetime=ref_time
    )

    if feat_cols:
        X = X.reindex(columns=feat_cols, fill_value=0)

    proba = pipeline.predict_proba(X)[0]
    classes_model = list(pipeline.classes_)

    prob_map = {cls: float(p) for cls, p in zip(classes_model, proba)}
    p_home = prob_map.get("H", 0.0)
    p_draw = prob_map.get("D", 0.0)
    p_away = prob_map.get("A", 0.0)

    st.subheader(f"🔮 Prediction for {home_team} vs {away_team}")
    st.write(f"**Match Time (UTC):** {ref_time}")
    st.write(f"- 🏠 Home Win: **{p_home:.1%}**")
    st.write(f"- 🤝 Draw: **{p_draw:.1%}**")
    st.write(f"- 🛫 Away Win: **{p_away:.1%}**")

    outcome, conf = max(
        [("Home Win", p_home), ("Draw", p_draw), ("Away Win", p_away)],
        key=lambda x: x[1]
    )
    st.success(f"**Most likely outcome: {outcome} ({conf:.1%})**")
