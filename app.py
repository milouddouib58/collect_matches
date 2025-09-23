import streamlit as st
import requests
import joblib
import pandas as pd

# ==============================
# API Key (من secrets)
# ==============================
API_KEY = st.secrets["API_TOKEN"]
API_URL = "https://api.football-data.org/v4/matches"

HEADERS = {"X-Auth-Token": API_KEY}

# ==============================
# جلب المباريات القادمة
# ==============================
def get_upcoming_matches(league_code="PL"):
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches?status=SCHEDULED"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        st.error(f"❌ API error: {response.status_code}")
        return []
    data = response.json()
    matches = data.get("matches", [])
    return matches

# ==============================
# تحميل الموديل
# ==============================
def load_model(model_path):
    return joblib.load(model_path)

# ==============================
# واجهة Streamlit
# ==============================
st.title("⚽ Football Match Predictor")

# اختيار الدوري
leagues = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1"
}
league_name = st.selectbox("📌 اختر الدوري", list(leagues.keys()))
league_code = leagues[league_name]

# جلب المباريات القادمة
matches = get_upcoming_matches(league_code)

if matches:
    # تحويل لقائمة لعرضها بشكل مرتب
    match_options = [
        f"{m['homeTeam']['name']} vs {m['awayTeam']['name']} ({m['utcDate'][:10]})"
        for m in matches
    ]
    selected_match = st.selectbox("🎯 اختر مباراة", match_options)

    # استخراج بيانات الفريقين
    match_index = match_options.index(selected_match)
    home_team = matches[match_index]["homeTeam"]["name"]
    away_team = matches[match_index]["awayTeam"]["name"]

    st.write(f"✅ اخترت المباراة: **{home_team} vs {away_team}**")

    # اختيار الموديل
    model_file = st.selectbox(
        "🤖 اختر الموديل",
        ["model_PL_advanced.joblib", "best_grid_model_logreg_estimator.joblib"]
    )

    if st.button("🚀 تنبؤ النتيجة"):
        model = load_model(model_file)

        # ⚠️ هنا لازم يكون عندك نفس طريقة تجهيز الداتا اللي استعملتها أثناء التدريب
        # مثال بسيط (لازم تعدل حسب الـ features الحقيقية عندك):
        features = pd.DataFrame([{
            "home_team": home_team,
            "away_team": away_team
        }])

        try:
            prediction = model.predict(features)[0]
            probas = model.predict_proba(features)[0]

            st.subheader("🔮 النتيجة المتوقعة")
            st.write(f"👉 {home_team} vs {away_team}: **{prediction}**")
            st.write("📊 الاحتمالات:")
            st.write(f"- فوز {home_team}: {probas[0]:.2f}")
            st.write(f"- تعادل: {probas[1]:.2f}")
            st.write(f"- فوز {away_team}: {probas[2]:.2f}")
        except Exception as e:
            st.error(f"⚠️ Error in prediction: {e}")

else:
    st.warning("🚫 لا توجد مباريات قادمة لهذا الدوري.")
