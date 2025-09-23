# app.py
# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime, timezone
from importlib import import_module

from features_lib import compute_single_pair_features, FEATURE_VERSION

# skops للتحميل الآمن
try:
    from skops.io import load as skops_load
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽")

# تحميل API key
if "FOOTBALL_DATA_API_KEY" in st.secrets and st.secrets["FOOTBALL_DATA_API_KEY"]:
    os.environ["FOOTBALL_DATA_API_KEY"] = st.secrets["FOOTBALL_DATA_API_KEY"]
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if not API_KEY:
    st.error("❌ API key not found! Please set FOOTBALL_DATA_API_KEY in Streamlit secrets or environment.")
    st.stop()

COMPETITIONS = {
    "PL": "Premier League", "PD": "La Liga", "SA": "Serie A", "BL1": "Bundesliga", "FL1": "Ligue 1",
}

@st.cache_data(ttl=300)
def fetch_upcoming_matches(league_code: str) -> pd.DataFrame:
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"status": "SCHEDULED"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for m in data.get("matches", []):
        rows.append({
            "utcDate": m.get("utcDate"),
            "homeTeam": (m.get("homeTeam") or {}).get("name"),
            "awayTeam": (m.get("awayTeam") or {}).get("name"),
            "matchday": m.get("matchday"),
        })
    return pd.DataFrame(rows)

def ensure_sklearn_unpickle_shims():
    # Hotfix: حقن _RemainderColsList عند تحميل joblib قديم
    try:
        ct_mod = import_module("sklearn.compose._column_transformer")
        if not hasattr(ct_mod, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            ct_mod._RemainderColsList = _RemainderColsList
    except Exception:
        pass

@st.cache_resource
def load_model(model_path: str):
    try:
        if SKOPS_AVAILABLE and model_path.endswith(".skops"):
            return skops_load(model_path, trusted=True)
        ensure_sklearn_unpickle_shims()  # fallback للـ joblib
        return joblib.load(model_path)
    except Exception as e:
        msg = str(e)
        if "_RemainderColsList" in msg:
            st.error(
                "Failed to load model due to scikit-learn version mismatch. "
                "Use a .skops model, or pin scikit-learn to the training version, "
                "or retrain with the provided script (which avoids ColumnTransformer)."
            )
        else:
            st.error(f"Failed to load model: {e}")
        raise

st.title("⚽ Football Match Predictor")

colA, colB = st.columns([1, 2])
with colA:
    league = st.selectbox("Select League", list(COMPETITIONS.keys()), format_func=lambda x: COMPETITIONS[x])
with colB:
    st.info(f"Feature version expected by code: {FEATURE_VERSION}")

matches_df = fetch_upcoming_matches(league)
if matches_df.empty:
    st.warning("⚠️ No upcoming matches found for this league.")
    st.stop()

def fmt_match(row):
    try:
        dt = datetime.fromisoformat(row.utcDate.replace("Z", "+00:00"))
        dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        dt_str = row.utcDate
    return f"{row.homeTeam} vs {row.awayTeam} — {dt_str}"

match_strs = [fmt_match(r) for r in matches_df.itertuples()]
match_choice = st.selectbox("Choose Match", match_strs)
chosen = matches_df.iloc[match_strs.index(match_choice)]
home_team, away_team = chosen["homeTeam"], chosen["awayTeam"]
ref_time = datetime.fromisoformat(chosen["utcDate"].replace("Z", "+00:00")).astimezone(timezone.utc)

st.divider()
st.subheader("Model and Data")
default_model = f"ensemble_model_v3_{league}.skops" if SKOPS_AVAILABLE else f"ensemble_model_v3_{league}.joblib"
model_file = st.text_input("Model file path", default_model)
data_file = st.text_input("Historical data file path", "matches_data.csv")

if st.button("🔮 Predict"):
    import os
    if not os.path.exists(model_file):
        st.error(f"❌ Model file not found: {model_file}"); st.stop()
    if not os.path.exists(data_file):
        st.error(f"❌ Data file not found: {data_file}"); st.stop()

    matches_hist = pd.read_csv(data_file)
    pipeline = load_model(model_file)

    # تحذير توافق نسخة الميزات
    model_feature_version = getattr(pipeline, "feature_version_", "Unknown")
    if model_feature_version != "Unknown" and model_feature_version != FEATURE_VERSION:
        st.warning(f"Model feature version ({model_feature_version}) differs from code ({FEATURE_VERSION}). Consider retraining.")

    # حساب ميزات المباراة المختارة
    X, meta = compute_single_pair_features(
        matches=matches_hist,
        competition=league,
        home_team_input=home_team,
        away_team_input=away_team,
        ref_datetime=ref_time
    )

    # إعادة ترتيب الأعمدة بحسب ما حفظه النموذج
    expected_cols = getattr(pipeline, "feature_names_expected_", None)
    if expected_cols is not None:
        X = X.reindex(columns=list(expected_cols), fill_value=0)
    elif hasattr(pipeline, "feature_names_in_"):
        X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)

    # تنبؤ واحتمالات
    proba = pipeline.predict_proba(X)[0]
    classes_model = list(pipeline.classes_)
    prob_map = {cls: float(p) for cls, p in zip(classes_model, proba)}
    p_home, p_draw, p_away = prob_map.get("H", 0.0), prob_map.get("D", 0.0), prob_map.get("A", 0.0)

    st.subheader(f"🔮 Prediction for {home_team} vs {away_team}")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏠 Home Win", f"{p_home:.1%}")
    c2.metric("🤝 Draw", f"{p_draw:.1%}")
    c3.metric("🛫 Away Win", f"{p_away:.1%}")

    outcome, conf = max([("Home Win", p_home), ("Draw", p_draw), ("Away Win", p_away)], key=lambda x: x[1])
    st.success(f"Most likely outcome: {outcome} ({conf:.1%})")

    with st.expander("Details"):
        st.write({
            "home_team_input": home_team,
            "home_team_resolved": meta.get("home_team_resolved"),
            "away_team_input": away_team,
            "away_team_resolved": meta.get("away_team_resolved"),
            "feature_version_in_model": model_feature_version,
            "classes_": classes_model,
        })
