# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime

from features_lib import compute_single_pair_features, FEATURE_VERSION

# ... (جزء تحميل مفتاح API وإعدادات الدوري يبقى كما هو)
if "FOOTBALL_DATA_API_KEY" in st.secrets and st.secrets["FOOTBALL_DATA_API_KEY"]:
    os.environ["FOOTBALL_DATA_API_KEY"] = st.secrets["FOOTBALL_DATA_API_KEY"]
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    st.error("❌ API key not found! Please set FOOTBALL_DATA_API_KEY in Streamlit secrets.")
    st.stop()

COMPETITIONS = {"PL": "Premier League", "PD": "La Liga", "SA": "Serie A", "BL1": "Bundesliga", "FL1": "Ligue 1"}

@st.cache_data(ttl=300)
def fetch_upcoming_matches(league_code: str):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=SCHEDULED"
    headers = {"X-Auth-Token": API_KEY}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    matches = [{"utcDate": m["utcDate"], "homeTeam": m["homeTeam"]["name"], "awayTeam": m["awayTeam"]["name"]} for m in data.get("matches", [])]
    return pd.DataFrame(matches)

# --- تم تبسيط وتصحيح هذه الدالة ---
@st.cache_resource
def load_model(model_path):
    """
    Loads any scikit-learn compatible model object directly from a joblib file.
    """
    try:
        model = joblib.load(model_path)
        st.success(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}")
        raise e

st.title("⚽ Football Match Predictor")

league = st.selectbox("Select League", list(COMPETITIONS.keys()), format_func=lambda x: COMPETITIONS[x])

matches_df = fetch_upcoming_matches(league)
if matches_df.empty:
    st.warning("⚠️ No upcoming matches found for this league.")
    st.stop()

match_strs = [f"{row.homeTeam} vs {row.awayTeam} ({row.utcDate})" for row in matches_df.itertuples()]
match_choice = st.selectbox("Choose Match", match_strs)

chosen = matches_df.iloc[match_strs.index(match_choice)]
home_team, away_team = chosen["homeTeam"], chosen["awayTeam"]
ref_time = datetime.fromisoformat(chosen["utcDate"].replace("Z", "+00:00"))

model_file = st.text_input("Model file path", "ensemble_model_v3_PL.joblib")
data_file = st.text_input("Historical data file path", "matches_data.csv")

if st.button("🔮 Predict"):
    if not os.path.exists(model_file):
        st.error(f"❌ Model file not found: {model_file}"); st.stop()
    if not os.path.exists(data_file):
        st.error(f"❌ Data file not found: {data_file}"); st.stop()

    matches_hist = pd.read_csv(data_file)
    pipeline = load_model(model_file)
    
    X, meta = compute_single_pair_features(
        matches=matches_hist,
        competition=league,
        home_team_input=home_team,
        away_team_input=away_team,
        ref_datetime=ref_time
    )

    proba = pipeline.predict_proba(X)[0]
    classes_model = list(pipeline.classes_)

    prob_map = {cls: float(p) for cls, p in zip(classes_model, proba)}
    p_home, p_draw, p_away = prob_map.get("H", 0.0), prob_map.get("D", 0.0), prob_map.get("A", 0.0)

    st.subheader(f"🔮 Prediction for {home_team} vs {away_team}")
    col1, col2, col3 = st.columns(3)
    col1.metric("🏠 Home Win", f"{p_home:.1%}")
    col2.metric("🤝 Draw", f"{p_draw:.1%}")
    col3.metric("🛫 Away Win", f"{p_away:.1%}")

    outcome, conf = max([("Home Win", p_home), ("Draw", p_draw), ("Away Win", p_away)], key=lambda x: x[1])
    st.success(f"**Most likely outcome: {outcome} ({conf:.1%})**")

