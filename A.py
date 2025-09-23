# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

st.set_page_config(page_title="Football Predictor", layout="centered")

# ---------- Utils ----------
def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    alias = {
        'date': ['date','match_date'],
        'competition': ['competition','league','tournament'],
        'home_team_name': ['home_team_name','home_team','home'],
        'away_team_name': ['away_team_name','away_team','away'],
        'home_team_id': ['home_team_id','home_id','home_code'],
        'away_team_id': ['away_team_id','away_id','away_code'],
        'home_team_elo': ['home_team_elo','home_elo'],
        'away_team_elo': ['away_team_elo','away_elo'],
        'elo_diff': ['elo_diff'],
        'home_form_points': ['home_form_points','home_last5_points','home_form_pts'],
        'away_form_points': ['away_form_points','away_last5_points','away_form_pts'],
        'home_form_gs': ['home_form_gs','home_last5_goals_scored'],
        'home_form_gc': ['home_form_gc','home_last5_goals_conceded'],
        'away_form_gs': ['away_form_gs','away_last5_goals_scored'],
        'away_form_gc': ['away_form_gc','away_last5_goals_conceded'],
        'home_team_league_pos': ['home_team_league_pos','home_pos','home_rank'],
        'away_team_league_pos': ['away_team_league_pos','away_pos','away_rank'],
    }
    for target, cands in alias.items():
        if target not in df.columns:
            for c in cands:
                if c in df.columns:
                    df[target] = df[c]
                    break
    if 'home_team_id' not in df.columns and 'home_team_name' in df.columns:
        df['home_team_id'] = df['home_team_name'].astype(str)
    if 'away_team_id' not in df.columns and 'away_team_name' in df.columns:
        df['away_team_id'] = df['away_team_name'].astype(str)
    if 'elo_diff' not in df.columns:
        if 'home_team_elo' in df.columns and 'away_team_elo' in df.columns:
            df['elo_diff'] = pd.to_numeric(df['home_team_elo'], errors='coerce') - pd.to_numeric(df['away_team_elo'], errors='coerce')
    if 'competition' not in df.columns:
        df['competition'] = 'Unknown'
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(date.today()).strftime("%Y-%m-%d")
    return df

def combine_probs(p_draw: np.ndarray, p_home_given_no_draw: np.ndarray) -> np.ndarray:
    p_home = (1.0 - p_draw) * p_home_given_no_draw
    p_away = (1.0 - p_draw) * (1.0 - p_home_given_no_draw)
    P = np.vstack([p_home, p_draw, p_away]).T
    P = np.clip(P, 1e-12, 1.0)
    return P / P.sum(axis=1, keepdims=True)

def compute_closeness_score(X: pd.DataFrame) -> np.ndarray:
    # يعتمد على مخرجات FeatureBuilder
    return 0.5*X['closeness_elo'].values + 0.25*X['closeness_points'].values + 0.25*X['closeness_gd'].values

class BinTS:
    def __init__(self, T): self.T = T
    def transform(self, p):
        p = np.clip(p, 1e-12, 1-1e-12)
        z = np.log(p/(1-p))
        return 1.0 / (1.0 + np.exp(-z / self.T))

@st.cache_resource
def load_artifact(path_or_file):
    if hasattr(path_or_file, "read"):  # Uploaded file
        return joblib.load(path_or_file)
    return joblib.load(path_or_file)

def predict_from_df(art, df_raw: pd.DataFrame) -> pd.DataFrame:
    # 1) normalize + fill optional columns
    df = normalize_schema(df_raw)
    for opt in ['home_team_league_pos','away_team_league_pos','competition','date','home_team_id','away_team_id']:
        if opt not in df.columns: df[opt] = np.nan

    # 2) features
    fb = art['feature_builder']
    tp = art['team_priors']
    X = pd.concat([fb.transform(df), tp.transform(df)], axis=1)

    # 3) stage models + calibration
    m1, T1 = art['stage1_draw_model'], art['stage1_temp_T']
    m2, T2 = art['stage2_ha_model'], art['stage2_temp_T']
    tau, gamma = art['decision']['tau_draw'], art['decision']['gamma_close']
    label_map = art['label_map']

    p_draw = BinTS(T1).transform(m1.predict_proba(X)[:,1])
    p_home_cond = BinTS(T2).transform(m2.predict_proba(X)[:,1])
    P = combine_probs(p_draw, p_home_cond)

    s = compute_closeness_score(X)
    draw_mask = (p_draw >= tau) & (s >= gamma)
    y_pred = np.where(draw_mask, 1, np.where(p_home_cond >= 0.5, 0, 2))

    out = pd.DataFrame({
        'pred': [label_map[i] for i in y_pred],
        'prob_home': P[:,0],
        'prob_draw': P[:,1],
        'prob_away': P[:,2],
        'p_draw_stage1': p_draw,
        'p_home_given_no_draw': p_home_cond,
        'closeness': s
    })
    # احتفظ ببعض أعمدة الإدخال المفيدة
    keep_cols = [c for c in ['date','competition','home_team_name','away_team_name','home_team_id','away_team_id',
                             'home_team_elo','away_team_elo','home_form_points','away_form_points'] if c in df.columns]
    return pd.concat([df[keep_cols].reset_index(drop=True), out], axis=1)

# ---------- UI ----------
st.title("⚽ Football Outcome Predictor (Inference)")

# تحميل النموذج
st.markdown("حمّل ملف النموذج المدرب (.joblib) أو استخدم مسار على الخادم.")
colA, colB = st.columns([2,1])
with colA:
    model_file = st.file_uploader("Upload artifact (.joblib)", type=["joblib"])
with colB:
    default_path = st.text_input("Path on server", "football_model_two_stage.joblib")

if st.button("Load Model"):
    try:
        art = load_artifact(model_file if model_file else default_path)
        st.success("Model loaded ✅")
        st.session_state["art"] = art
        # عرض معلومات النسخ للمساعدة على ضبط requirements
        meta = art.get('meta', {})
        st.info(f"xgboost={meta.get('xgboost_version','?')} | best_iter_stage1={meta.get('best_iter_stage1')} | best_iter_stage2={meta.get('best_iter_stage2')}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if "art" in st.session_state:
    st.header("Single Match Prediction")
    with st.form("single"):
        c1, c2 = st.columns(2)
        with c1:
            date_str = st.text_input("Date (YYYY-MM-DD)", value=str(date.today()))
            competition = st.text_input("Competition", value="League")
            home_name = st.text_input("Home Team (name/id)", value="Home")
            home_elo = st.number_input("Home ELO", 1200, 2300, 1600)
            home_form_pts = st.number_input("Home form points (last 5)", 0, 15, 8)
            home_gs = st.number_input("Home goals scored (last 5)", 0, 20, 7)
            home_gc = st.number_input("Home goals conceded (last 5)", 0, 20, 4)
            home_pos = st.number_input("Home league pos (optional)", 1, 30, 10)
        with c2:
            away_name = st.text_input("Away Team (name/id)", value="Away")
            away_elo = st.number_input("Away ELO", 1200, 2300, 1580)
            away_form_pts = st.number_input("Away form points (last 5)", 0, 15, 7)
            away_gs = st.number_input("Away goals scored (last 5)", 0, 20, 6)
            away_gc = st.number_input("Away goals conceded (last 5)", 0, 20, 5)
            away_pos = st.number_input("Away league pos (optional)", 1, 30, 12)
        submitted = st.form_submit_button("Predict")
    if submitted:
        art = st.session_state["art"]
        row = {
            "date": date_str,
            "competition": competition,
            "home_team_id": home_name, "away_team_id": away_name,
            "home_team_name": home_name, "away_team_name": away_name,
            "home_team_elo": home_elo, "away_team_elo": away_elo,
            "home_form_points": home_form_pts, "home_form_gs": home_gs, "home_form_gc": home_gc,
            "away_form_points": away_form_pts, "away_form_gs": away_gs, "away_form_gc": away_gc,
            "home_team_league_pos": home_pos, "away_team_league_pos": away_pos
        }
        df_pred = predict_from_df(art, pd.DataFrame([row]))
        st.subheader(f"Prediction: {df_pred.loc[0,'pred']}")
        st.write({
            "Home Win": f"{df_pred.loc[0,'prob_home']:.2%}",
            "Draw": f"{df_pred.loc[0,'prob_draw']:.2%}",
            "Away Win": f"{df_pred.loc[0,'prob_away']:.2%}"
        })
        with st.expander("Details"):
            st.dataframe(df_pred)

    st.header("Batch Prediction (CSV)")
    st.markdown("ارفع CSV يحتوي على الأعمدة القياسية أو المرادفات (سيتم التطبيع تلقائيًا).")
    csv_file = st.file_uploader("Upload matches CSV", type=["csv"], key="csvu")
    if csv_file is not None:
        try:
            df_in = pd.read_csv(csv_file)
            df_out = predict_from_df(st.session_state["art"], df_in)
            st.success(f"Predicted {len(df_out)} matches ✅")
            st.dataframe(df_out.head(20))
            st.download_button("Download predictions CSV", df_out.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed to predict: {e}")
