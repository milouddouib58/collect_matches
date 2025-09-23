# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import joblib
import subprocess
import sys
import os
from features_lib import _resolve_team_name, compute_h2h_for_home, list_feature_columns, parse_dates, compute_single_pair_features
from predict_match import map_proba_to_HDA

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Football Predictor ⚽",
    page_icon="⚽",
    layout="wide"
)

# --- الدوال المساعدة ---

def run_script(command, api_key=None):
    """
    يشغل سكربت بايثون ويعرض المخرجات في الواجهة بشكل حي.
    """
    env = os.environ.copy()
    if api_key:
        env['FOOTBALL_DATA_API_KEY'] = api_key
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        env=env
    )
    
    output_box = st.empty()
    output_text = ""
    for line in iter(process.stdout.readline, ''):
        output_text += line
        output_box.code(output_text, language='bash')
        
    process.wait()
    st.success("اكتملت العملية بنجاح!")

@st.cache_data(ttl=3600)
def fetch_upcoming_matches(league_code, api_key):
    """
    تجلب المباريات القادمة والمقامة حاليًا.
    """
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {"status": "SCHEDULED,LIVE,IN_PLAY"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json().get("matches", [])
    except requests.exceptions.RequestException as e:
        st.error(f"خطأ في جلب المباريات: {e}")
        return []

@st.cache_resource
def load_model(path):
    """
    تحميل النموذج المدرب مع التخزين المؤقت.
    """
    return joblib.load(path)

@st.cache_data
def load_data(path):
    """
    تحميل البيانات التاريخية مع التخزين المؤقت.
    """
    return parse_dates(pd.read_csv(path))

# --- الواجهة الرئيسية ---
st.title("⚽ نظام التنبؤ بنتائج المباريات")
st.sidebar.header("الإعدادات")
api_key = st.sidebar.text_input("🔑 أدخل مفتاح football-data.org API", type="password")

if api_key:
    st.session_state['api_key'] = api_key
    st.sidebar.success("تم حفظ مفتاح API!")

# --- إنشاء التبويبات ---
tab1, tab2, tab3 = st.tabs(["📊 جمع البيانات", "🧠 تدريب النموذج", "🔮 التنبؤ"])

# --- محتوى تبويب جمع البيانات ---
with tab1:
    st.header("الخطوة 1: جمع وتحديث بيانات المباريات")
    league_to_collect = st.text_input("أدخل رمز الدوري (e.g., PL, BL1, PD, SA)", "PL", key="collect_league")

    if st.button("🚀 بدء جمع البيانات"):
        if 'api_key' in st.session_state and st.session_state['api_key']:
            st.info("جاري جمع البيانات... قد تستغرق هذه العملية بعض الوقت.")
            command = [sys.executable, "collect_matches.py", "--league", league_to_collect, "--current-only"]
            run_script(command, st.session_state['api_key'])
        else:
            st.warning("الرجاء إدخال مفتاح API في الشريط الجانبي أولاً.")

# --- محتوى تبويب تدريب النموذج ---
with tab2:
    st.header("الخطوة 2: هندسة الميزات وتدريب النموذج")
    st.error(
        """
        **تحذير:** عملية التدريب تستهلك موارد عالية.
        من المرجح أن تفشل على منصة Streamlit Cloud المجانية بسبب انتهاء المهلة.
        **الحل الموصى به:** قم بتشغيل هذه الخطوة على جهاز الكمبيوتر الخاص بك.
        """
    )
    
    league_to_train = st.text_input("أدخل رمز الدوري للتدريب", "PL", key="train_league")

    st.subheader("2.1: هندسة الميزات")
    if st.button("🛠️ إنشاء ملف الميزات"):
        features_file = f"features_{league_to_train}.csv"
        if not os.path.exists("matches_data.csv"):
            st.error("ملف `matches_data.csv` غير موجود. الرجاء جمعه أولاً.")
        else:
            command = [sys.executable, "engineer_features.py", "--league", league_to_train, "--output", features_file]
            run_script(command)

    st.subheader("2.2: تدريب النموذج")
    if st.button("🏃‍♂️ بدء تدريب النموذج"):
        model_file = f"ensemble_model_v3_{league_to_train}.joblib"
        features_file = f"features_{league_to_train}.csv"
        if not os.path.exists(features_file):
            st.error(f"ملف الميزات {features_file} غير موجود. الرجاء إنشاؤه أولاً.")
        else:
            command = [sys.executable, "train_model_advanced.py", "--features-file", features_file, "--league", league_to_train, "--model-out", model_file]
            run_script(command)

# --- محتوى تبويب التنبؤ ---
with tab3:
    st.header("الخطوة 3: التنبؤ بنتائج المباريات")
    
    leagues = {
        "الدوري الإنجليزي": "PL",
        "الدوري الإسباني": "PD",
        "الدوري الإيطالي": "SA",
        "الدوري الألماني": "BL1",
        "دوري أبطال أوروبا": "CL"
    }
    selected_league_name = st.selectbox("اختر الدوري", list(leagues.keys()))
    league_code = leagues[selected_league_name]

    model_path = f"ensemble_model_v3_{league_code}.joblib"
    data_path = "matches_data.csv"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        st.warning(f"ملف النموذج `{model_path}` أو ملف البيانات `{data_path}` غير موجود. الرجاء تدريب النموذج وجمع البيانات أولاً.")
    else:
        if 'api_key' not in st.session_state or not st.session_state['api_key']:
            st.warning("الرجاء إدخال مفتاح API في الشريط الجانبي لجلب المباريات القادمة.")
        else:
            model = load_model(model_path)
            matches_history = load_data(data_path)
            
            matches = fetch_upcoming_matches(league_code, st.session_state['api_key'])
            if not matches:
                st.info("لا توجد مباريات قادمة أو قائمة حاليًا في هذا الدوري.")
            else:
                match_options = {
                    f"{m['homeTeam']['shortName']} vs {m['awayTeam']['shortName']} ({pd.to_datetime(m['utcDate']).strftime('%d-%b %H:%M')})": m
                    for m in matches
                }
                selected_match_str = st.selectbox("اختر المباراة", list(match_options.keys()))

                if st.button("🎯 تنبؤ بالنتيجة"):
                    selected_match = match_options[selected_match_str]
                    home_team_name = selected_match['homeTeam']['shortName']
                    away_team_name = selected_match['awayTeam']['shortName']
                    
                    with st.spinner("...جاري التحليل"):
                        X, meta = compute_single_pair_features(
                            matches=matches_history,
                            competition=league_code,
                            home_team_input=home_team_name,
                            away_team_input=away_team_name,
                            ref_datetime=pd.to_datetime(selected_match['utcDate'])
                        )
                        
                        expected_cols = getattr(model, "feature_names_expected_", list_feature_columns())
                        X = X.reindex(columns=expected_cols, fill_value=0)
                        
                        proba = model.predict_proba(X)[0]
                        encoder = getattr(model, "label_encoder_", None)
                        classes_model = encoder.classes_ if encoder else list(getattr(model, "classes_", ['A', 'D', 'H']))
                        prob_map = map_proba_to_HDA(classes_model, proba)

                        st.subheader(f"📊 نتيجة التنبؤ لمباراة: {meta['home_team_resolved']} vs {meta['away_team_resolved']}")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("فوز المضيف (H)", f"{prob_map['H']:.1%}")
                        col2.metric("تعادل (D)", f"{prob_map['D']:.1%}")
                        col3.metric("فوز الضيف (A)", f"{prob_map['A']:.1%}")
