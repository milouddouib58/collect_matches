# app.py
# -*- coding: utf-8 -*-
import os
import sys
import streamlit as st
import pandas as pd
import requests
import joblib
from datetime import datetime, timezone
from importlib import import_module

from features_lib import compute_single_pair_features, FEATURE_VERSION

# حاول استخدام skops أولاً (للتحميل الآمن عبر الإصدارات)
try:
    from skops.io import load as skops_load
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="centered")

# ============== أدوات مساعدة ==============
def ensure_sklearn_unpickle_shims():
    # Hotfix: نماذج joblib القديمة قد تحتاج _RemainderColsList
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
        if SKOPS_AVAILABLE and model_path.lower().endswith(".skops"):
            return skops_load(model_path, trusted=True)
        # fallback للـ joblib
        ensure_sklearn_unpickle_shims()
        return joblib.load(model_path)
    except Exception as e:
        msg = str(e)
        if "_RemainderColsList" in msg:
            st.error(
                "Failed to load model due to scikit-learn version mismatch. "
                "Use a .skops model, or pin scikit-learn to the training version, "
                "or retrain with the provided training script (no ColumnTransformer)."
            )
        else:
            st.error(f"Failed to load model: {e}")
        raise

@st.cache_data(ttl=300)
def fetch_upcoming_matches(league_code: str, api_key: str) -> pd.DataFrame:
    try:
        url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
        headers = {"X-Auth-Token": api_key}
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
    except Exception as e:
        st.warning(f"Failed to fetch matches: {e}")
        return pd.DataFrame([])

def fmt_match(row):
    try:
        dt = datetime.fromisoformat(row.utcDate.replace("Z", "+00:00"))
        dt_str = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        dt_str = row.utcDate
    return f"{row.homeTeam} vs {row.awayTeam} — {dt_str}"

def render_diag():
    with st.sidebar.expander("Diagnostics", expanded=False):
        import platform
        ver = {}
        try:
            import sklearn; ver["scikit-learn"] = sklearn.__version__
        except Exception:
            ver["scikit-learn"] = "N/A"
        try:
            import numpy as np; ver["numpy"] = np.__version__
        except Exception:
            ver["numpy"] = "N/A"
        try:
            import pandas as pd; ver["pandas"] = pd.__version__
        except Exception:
            ver["pandas"] = "N/A"
        ver["joblib"] = getattr(joblib, "__version__", "N/A")
        ver["skops"] = "available" if SKOPS_AVAILABLE else "not installed"
        ver["python"] = sys.version.split()[0]
        ver["platform"] = platform.platform()
        st.json(ver)
        colA, colB = st.columns(2)
        if colA.button("Clear data cache"):
            st.cache_data.clear()
            st.success("Data cache cleared.")
        if colB.button("Clear resource cache"):
            st.cache_resource.clear()
            st.success("Resource cache cleared.")

def map_proba_to_HDA(classes_model, proba):
    """
    يحوّل أي تمثيل للفئات إلى خريطة H/D/A -> probability.
    يدعم:
      - نصوص H/D/A مباشرة (أو مرادفات: Home/Draw/Away, 1/X/2)
      - أرقام {0,1,2} كما ينتج LabelEncoder أبجديًا (A=0, D=1, H=2)
    """
    labels = list(classes_model)

    # حالة نصوص
    if all(isinstance(c, str) for c in labels):
        norm = {}
        for cls, p in zip(labels, proba):
            s = cls.strip().lower()
            if s in ("h", "home", "home win", "1"): norm["H"] = float(p)
            elif s in ("d", "draw", "x"): norm["D"] = float(p)
            elif s in ("a", "away", "away win", "2"): norm["A"] = float(p)
        # إذا كانت بالضبط H/D/A ولكن بحروف مختلفة
        if not norm and set([c.upper() for c in labels]) >= {"H","D","A"}:
            for cls, p in zip(labels, proba):
                norm[cls.upper()] = float(p)
        return {"H": norm.get("H",0.0), "D": norm.get("D",0.0), "A": norm.get("A",0.0)}

    # حالة أرقام {0,1,2} => A=0, D=1, H=2
    try:
        if set(int(x) for x in labels) == {0,1,2}:
            idxA = labels.index(0)
            idxD = labels.index(1)
            idxH = labels.index(2)
            return {"H": float(proba[idxH]), "D": float(proba[idxD]), "A": float(proba[idxA])}
    except Exception:
        pass

    # fallback
    return {"H": 0.0, "D": 0.0, "A": 0.0}

# ============== واجهة المستخدم ==============
st.title("⚽ Football Match Predictor")
render_diag()

COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
}

st.info(f"Feature version expected by code: {FEATURE_VERSION}")

# اختيار وضع الإدخال
mode = st.radio("Input mode", ["Live API", "Manual entry"], index=0, horizontal=True)

# إدخال مسارات النموذج والبيانات
default_model_name = f"ensemble_model_v3_PL.skops" if SKOPS_AVAILABLE else f"ensemble_model_v3_PL.joblib"
model_file = st.text_input("Model file path", default_model_name)
data_file = st.text_input("Historical data CSV", "matches_data.csv")

league = None
home_team = None
away_team = None
ref_time = None

if mode == "Live API":
    # تحميل API KEY من secrets أو environment (لا نوقف الصفحة إن لم يوجد)
    if "FOOTBALL_DATA_API_KEY" in st.secrets and st.secrets["FOOTBALL_DATA_API_KEY"]:
        os.environ["FOOTBALL_DATA_API_KEY"] = st.secrets["FOOTBALL_DATA_API_KEY"]
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")

    c1, c2 = st.columns([1, 1])
    with c1:
        league = st.selectbox("League", list(COMPETITIONS.keys()), format_func=lambda x: COMPETITIONS[x], index=0)
    with c2:
        st.caption("Tip: Switch to Manual entry if you don't have an API key.")

    matches_df = pd.DataFrame([])
    if api_key:
        with st.spinner("Fetching scheduled matches..."):
            matches_df = fetch_upcoming_matches(league, api_key)
    else:
        st.warning("No API key found. Set FOOTBALL_DATA_API_KEY in secrets or environment, or use Manual entry mode.")

    if not matches_df.empty:
        match_strs = [fmt_match(r) for r in matches_df.itertuples()]
        match_choice = st.selectbox("Choose Match", match_strs)
        chosen = matches_df.iloc[match_strs.index(match_choice)]
        home_team, away_team = chosen["homeTeam"], chosen["awayTeam"]
        ref_time = datetime.fromisoformat(chosen["utcDate"].replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        st.info("No scheduled matches fetched. You can switch to Manual entry mode below.")
else:
    # Manual entry
    league = st.selectbox("League", list(COMPETITIONS.keys()), format_func=lambda x: COMPETITIONS[x], index=0, key="league_manual")
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.text_input("Home team", value="Arsenal")
    with c2:
        away_team = st.text_input("Away team", value="Chelsea")
    ref_time = st.datetime_input("Reference datetime (UTC)", value=datetime.now(timezone.utc))

st.divider()
run = st.button("🔮 Predict")

# ============== تنفيذ التنبؤ ==============
if run:
    # تحقق من الملفات
    if not os.path.exists(model_file):
        st.error(f"❌ Model file not found: {model_file}")
        st.stop()
    if not os.path.exists(data_file):
        st.error(f"❌ Data file not found: {data_file}")
        st.stop()
    if not (home_team and away_team and league and ref_time):
        st.error("Please ensure League, Home team, Away team, and Reference time are set.")
        st.stop()

    # تحميل البيانات والنموذج
    with st.spinner("Loading data and model..."):
        try:
            matches_hist = pd.read_csv(data_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        try:
            pipeline = load_model(model_file)
        except Exception:
            st.stop()

    # تحذير توافق نسخة الميزات
    model_feature_version = getattr(pipeline, "feature_version_", "Unknown")
    if model_feature_version != "Unknown" and model_feature_version != FEATURE_VERSION:
        st.warning(f"Model feature version ({model_feature_version}) differs from code ({FEATURE_VERSION}). Consider retraining.")

    # استخراج الميزات
    with st.spinner("Computing features..."):
        try:
            X, meta = compute_single_pair_features(
                matches=matches_hist,
                competition=league,
                home_team_input=home_team,
                away_team_input=away_team,
                ref_datetime=ref_time
            )
        except Exception as e:
            st.error(f"Feature computation failed: {e}")
            st.stop()

    # إعادة ترتيب الأعمدة كما يتوقعها النموذج
    expected_cols = getattr(pipeline, "feature_names_expected_", None)
    if expected_cols is not None:
        X = X.reindex(columns=list(expected_cols), fill_value=0)
    elif hasattr(pipeline, "feature_names_in_"):
        X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)

    # تنبؤ واحتمالات
    with st.spinner("Predicting..."):
        try:
            proba = pipeline.predict_proba(X)[0]
            classes_model = list(getattr(pipeline, "classes_", []))
            prob_map = map_proba_to_HDA(classes_model, proba)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    p_home, p_draw, p_away = prob_map.get("H", 0.0), prob_map.get("D", 0.0), prob_map.get("A", 0.0)

    st.subheader(f"🔮 Prediction for {home_team} vs {away_team}")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏠 Home Win", f"{p_home:.1%}")
    c2.metric("🤝 Draw", f"{p_draw:.1%}")
    c3.metric("🛫 Away Win", f"{p_away:.1%}")

    outcome, conf = max(
        [("Home Win", p_home), ("Draw", p_draw), ("Away Win", p_away)],
        key=lambda x: x[1]
    )
    st.success(f"Most likely outcome: {outcome} ({conf:.1%})")

    with st.expander("Details"):
        st.write({
            "home_team_input": home_team,
            "home_team_resolved": meta.get("home_team_resolved"),
            "away_team_input": away_team,
            "away_team_resolved": meta.get("away_team_resolved"),
            "feature_version_in_model": model_feature_version,
            "classes_": list(getattr(pipeline, "classes_", [])),
        })
