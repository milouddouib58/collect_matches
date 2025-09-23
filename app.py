# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
from features_lib import _resolve_team_name, compute_h2h_for_home, list_feature_columns, parse_dates
from predict_match import map_proba_to_HDA

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="متنبئ المباريات ⚽",
    page_icon="🔮",
    layout="centered"
)

# --- تحميل الموارد باستخدام الكاش لتسريع الأداء ---
# يتم تحميل هذه الموارد مرة واحدة فقط عند بدء تشغيل التطبيق
@st.cache_resource
def load_model(path="ensemble_model_v3_PL.joblib"):
    return joblib.load(path)

@st.cache_data
def load_data(path="matches_data.csv"):
    return parse_dates(pd.read_csv(path))

@st.cache_data
def load_features(path="features_PL.csv"):
    # نأخذ أحدث سطر لكل فريق من ملف الميزات
    df = pd.read_csv(path)
    df_latest = df.sort_values('date').drop_duplicates(subset='home_team', keep='last')
    return df_latest.set_index('home_team')

# --- تحميل كل شيء ---
MODEL = load_model()
MATCHES_HISTORY = load_data()
FEATURES_DF = load_features()
LEAGUE_CODE = "PL"

# --- الواجهة الرسومية ---
st.title("🔮 التنبؤ بنتائج المباريات")
st.write("تطبيق للتنبؤ بنتائج مباريات الدوري الإنجليزي الممتاز (PL) باستخدام نموذج تعلم الآلة.")

st.markdown("---")

# --- مدخلات المستخدم ---
home_team_input = st.text_input("الفريق المضيف (Home Team)", placeholder="e.g., Man City")
away_team_input = st.text_input("الفريق الضيف (Away Team)", placeholder="e.g., Arsenal")

if st.button("🎯 تنبؤ بالنتيجة"):
    if home_team_input and away_team_input:
        with st.spinner("...جاري التحليل"):
            try:
                # --- منطق التنبؤ ---
                team_pool = list(FEATURES_DF.index)
                home_team = _resolve_team_name(home_team_input, team_pool)
                away_team = _resolve_team_name(away_team_input, team_pool)

                # استخراج الميزات المحسوبة مسبقًا
                home_features = FEATURES_DF.loc[home_team]
                away_features = FEATURES_DF.loc[away_team]

                feature_vector = {}
                # بناء متجه الميزات
                for col in home_features.index:
                    if col.startswith('h_'):
                        feature_vector[col] = home_features[col]
                    if col.startswith('a_'):
                        corresponding_h_col = 'h_' + col[2:]
                        feature_vector[col] = away_features.get(corresponding_h_col)
                
                # حساب المواجهات المباشرة (H2H)
                h2h_series = pd.Series({"competition": LEAGUE_CODE, "home_team": home_team, "away_team": away_team, "date": pd.Timestamp.now(tz='utc')})
                pts, gf, ga = compute_h2h_for_home(h2h_series, MATCHES_HISTORY, k=3)
                feature_vector["h2h_home_pts3"], feature_vector["h2h_home_avg_gf3"], feature_vector["h2h_home_avg_ga3"] = pts, gf, ga

                # تجهيز البيانات للنموذج
                X = pd.DataFrame([feature_vector])
                expected_cols = getattr(MODEL, "feature_names_expected_", list_feature_columns())
                X = X.reindex(columns=expected_cols, fill_value=0)

                # التنبؤ
                proba = MODEL.predict_proba(X)[0]
                encoder = getattr(MODEL, "label_encoder_", None)
                classes_model = encoder.classes_ if encoder else list(getattr(MODEL, "classes_", ['A', 'D', 'H']))
                prob_map = map_proba_to_HDA(classes_model, proba)

                # عرض النتائج
                st.subheader(f"📊 نتيجة التنبؤ لمباراة: {home_team} vs {away_team}")
                col1, col2, col3 = st.columns(3)
                col1.metric("فوز المضيف (H)", f"{prob_map['H']:.1%}")
                col2.metric("تعادل (D)", f"{prob_map['D']:.1%}")
                col3.metric("فوز الضيف (A)", f"{prob_map['A']:.1%}")

            except KeyError as e:
                st.error(f"لم يتم العثور على الفريق: {e}. الرجاء التأكد من كتابة الاسم بشكل صحيح.")
            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {e}")
    else:
        st.warning("الرجاء إدخال اسم الفريقين.")
