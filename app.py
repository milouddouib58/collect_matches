# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import joblib
import subprocess
import sys
import os
from features_lib import (
    _resolve_team_name,
    compute_h2h_for_home,
    list_feature_columns,
    parse_dates,
    compute_single_pair_features,
)
from predict_match import map_proba_to_HDA

# ──────────────────────────────────────────────
# إعدادات الصفحة
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Football Predictor ⚽",
    page_icon="⚽",
    layout="wide",
)

# ← إصلاح #5: دعم RTL للنص العربي
st.markdown(
    """
    <style>
    /* اتجاه النص من اليمين لليسار */
    .stMarkdown, .stText, .stMetric, .stSelectbox label,
    .stTextInput label, .stHeader, .stSubheader {
        direction: rtl;
        text-align: right;
    }
    /* تنسيق بطاقات النتائج */
    div[data-testid="stMetric"] {
        background: #f0f2f6;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        font-size: 1.1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# الدوال المساعدة
# ──────────────────────────────────────────────

def run_script(command, api_key=None):
    """
    يشغل سكربت بايثون ويعرض المخرجات بشكل حي.
    ← إصلاح #1: يتحقق من returncode قبل إعلان النجاح.
    """
    env = os.environ.copy()
    if api_key:
        env["FOOTBALL_DATA_API_KEY"] = api_key

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=env,
    )

    output_box = st.empty()
    output_text = ""
    for line in iter(process.stdout.readline, ""):
        output_text += line
        output_box.code(output_text, language="bash")

    process.wait()

    # ← إصلاح #1: فحص كود الخروج
    if process.returncode == 0:
        st.success("✅ اكتملت العملية بنجاح!")
    else:
        st.error(
            f"❌ فشلت العملية (كود الخروج: {process.returncode}). "
            "راجع المخرجات أعلاه لمعرفة السبب."
        )
    return process.returncode


@st.cache_data(ttl=3600, show_spinner="جاري جلب المباريات...")
def fetch_upcoming_matches(league_code, api_key):
    """
    تجلب المباريات القادمة والمقامة حاليًا.
    """
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": api_key}
    # ← ملاحظة: الـ API يقبل قيم مفصولة بفاصلة
    params = {"status": "SCHEDULED,LIVE,IN_PLAY"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json().get("matches", [])
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 403:
            st.error("🔒 مفتاح API غير صالح أو لا يملك صلاحية لهذا الدوري.")
        elif resp.status_code == 429:
            st.error("⏳ تجاوزت الحد المسموح من الطلبات. انتظر دقيقة وحاول مجددًا.")
        else:
            st.error(f"خطأ HTTP {resp.status_code}: {e}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"خطأ في الاتصال: {e}")
        return []


@st.cache_resource(show_spinner="جاري تحميل النموذج...")
def load_model(path):
    """تحميل النموذج المدرب."""
    return joblib.load(path)


@st.cache_data(show_spinner="جاري تحميل البيانات التاريخية...")
def load_data(path):
    """تحميل البيانات التاريخية."""
    return parse_dates(pd.read_csv(path))


def build_match_label(m, idx):
    """
    ← إصلاح #3: يبني تسمية فريدة لكل مباراة
    باستخدام الفهرس لتجنب التكرار.
    """
    home = m["homeTeam"]["shortName"]
    away = m["awayTeam"]["shortName"]
    dt = pd.to_datetime(m["utcDate"]).strftime("%d-%b %H:%M")
    status = m.get("status", "")
    status_emoji = "🔴" if status in ("LIVE", "IN_PLAY") else "📅"
    return f"{status_emoji} {home} vs {away}  ({dt})"


# ──────────────────────────────────────────────
# الواجهة الرئيسية
# ──────────────────────────────────────────────
st.title("⚽ نظام التنبؤ بنتائج المباريات")

# --- الشريط الجانبي ---
st.sidebar.header("⚙️ الإعدادات")
api_key = st.sidebar.text_input(
    "🔑 أدخل مفتاح football-data.org API", type="password"
)
if api_key:
    st.session_state["api_key"] = api_key
    st.sidebar.success("✅ تم حفظ مفتاح API!")

# ── معلومات مفيدة ──
with st.sidebar.expander("📖 رموز الدوريات المدعومة"):
    st.markdown(
        """
        | الرمز | الدوري |
        |-------|--------|
        | `PL`  | الإنجليزي الممتاز |
        | `PD`  | الإسباني |
        | `SA`  | الإيطالي |
        | `BL1` | الألماني |
        | `CL`  | دوري الأبطال |
        """
    )

# ──────────────────────────────────────────────
# التبويبات
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 جمع البيانات", "🧠 تدريب النموذج", "🔮 التنبؤ"]
)

# ╔══════════════════════════════════════════════╗
# ║  تبويب 1: جمع البيانات                       ║
# ╚══════════════════════════════════════════════╝
with tab1:
    st.header("الخطوة 1: جمع وتحديث بيانات المباريات")
    league_to_collect = st.text_input(
        "أدخل رمز الدوري (e.g., PL, BL1, PD, SA)",
        "PL",
        key="collect_league",
    )

    if st.button("🚀 بدء جمع البيانات"):
        if not st.session_state.get("api_key"):
            st.warning("⚠️ الرجاء إدخال مفتاح API في الشريط الجانبي أولاً.")
            st.stop()  # ← إصلاح #4

        st.info("جاري جمع البيانات... قد تستغرق هذه العملية بعض الوقت.")
        command = [
            sys.executable,
            "collect_matches.py",
            "--league",
            league_to_collect,
            "--current-only",
        ]
        run_script(command, st.session_state["api_key"])

# ╔══════════════════════════════════════════════╗
# ║  تبويب 2: تدريب النموذج                      ║
# ╚══════════════════════════════════════════════╝
with tab2:
    st.header("الخطوة 2: هندسة الميزات وتدريب النموذج")
    st.error(
        """
        **⚠️ تحذير:** عملية التدريب تستهلك موارد عالية.
        من المرجح أن تفشل على Streamlit Cloud المجانية بسبب انتهاء المهلة.
        **الحل الموصى به:** شغّل هذه الخطوة على جهازك المحلي:
        ```bash
        python engineer_features.py --league PL --output features_PL.csv
        python train_model_advanced.py --features-file features_PL.csv --league PL
        ```
        """
    )

    league_to_train = st.text_input(
        "أدخل رمز الدوري للتدريب", "PL", key="train_league"
    )

    st.subheader("2.1: هندسة الميزات")
    if st.button("🛠️ إنشاء ملف الميزات"):
        if not os.path.exists("matches_data.csv"):
            st.error("ملف `matches_data.csv` غير موجود. الرجاء جمعه أولاً (الخطوة 1).")
            st.stop()  # ← إصلاح #4

        features_file = f"features_{league_to_train}.csv"
        command = [
            sys.executable,
            "engineer_features.py",
            "--league",
            league_to_train,
            "--output",
            features_file,
        ]
        run_script(command)

    st.subheader("2.2: تدريب النموذج")
    if st.button("🏃‍♂️ بدء تدريب النموذج"):
        features_file = f"features_{league_to_train}.csv"
        if not os.path.exists(features_file):
            st.error(
                f"ملف الميزات `{features_file}` غير موجود. أنشئه أولاً (الخطوة 2.1)."
            )
            st.stop()  # ← إصلاح #4

        model_file = f"ensemble_model_v3_{league_to_train}.joblib"
        command = [
            sys.executable,
            "train_model_advanced.py",
            "--features-file",
            features_file,
            "--league",
            league_to_train,
            "--model-out",
            model_file,
        ]
        run_script(command)

# ╔══════════════════════════════════════════════╗
# ║  تبويب 3: التنبؤ                              ║
# ╚══════════════════════════════════════════════╝
with tab3:
    st.header("الخطوة 3: التنبؤ بنتائج المباريات")

    leagues = {
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿 الدوري الإنجليزي": "PL",
        "🇪🇸 الدوري الإسباني": "PD",
        "🇮🇹 الدوري الإيطالي": "SA",
        "🇩🇪 الدوري الألماني": "BL1",
        "🏆 دوري أبطال أوروبا": "CL",
    }
    selected_league_name = st.selectbox("اختر الدوري", list(leagues.keys()))
    league_code = leagues[selected_league_name]

    model_path = f"ensemble_model_v3_{league_code}.joblib"
    data_path = "matches_data.csv"

    # ── التحقق من الملفات ──
    missing = []
    if not os.path.exists(model_path):
        missing.append(f"النموذج `{model_path}`")
    if not os.path.exists(data_path):
        missing.append(f"البيانات `{data_path}`")

    if missing:
        st.warning(
            f"الملفات التالية مفقودة: {' و '.join(missing)}.\n\n"
            "الرجاء جمع البيانات وتدريب النموذج أولاً."
        )
        st.stop()  # ← إصلاح #4

    if not st.session_state.get("api_key"):
        st.warning("الرجاء إدخال مفتاح API في الشريط الجانبي لجلب المباريات القادمة.")
        st.stop()  # ← إصلاح #4

    # ── تحميل النموذج والبيانات ──
    model = load_model(model_path)
    matches_history = load_data(data_path)

    # ── جلب المباريات القادمة ──
    matches = fetch_upcoming_matches(league_code, st.session_state["api_key"])

    if not matches:
        st.info("لا توجد مباريات قادمة أو قائمة حاليًا في هذا الدوري.")
        st.stop()

    # ← إصلاح #3: استخدام قائمة مرقمة بدلاً من dict قد يكرر المفاتيح
    match_labels = [build_match_label(m, i) for i, m in enumerate(matches)]
    selected_idx = st.selectbox(
        "اختر المباراة",
        range(len(matches)),
        format_func=lambda i: match_labels[i],
    )

    if st.button("🎯 تنبؤ بالنتيجة"):
        selected_match = matches[selected_idx]
        home_team_name = selected_match["homeTeam"]["shortName"]
        away_team_name = selected_match["awayTeam"]["shortName"]

        # ← إصلاح #2: try/except حول كل منطق التنبؤ
        try:
            with st.spinner("جاري التحليل والتنبؤ..."):
                # ── حساب الميزات ──
                X, meta = compute_single_pair_features(
                    matches=matches_history,
                    competition=league_code,
                    home_team_input=home_team_name,
                    away_team_input=away_team_name,
                    ref_datetime=pd.to_datetime(selected_match["utcDate"]),
                )

                # ── محاذاة الأعمدة مع ما يتوقعه النموذج ──
                expected_cols = getattr(
                    model, "feature_names_expected_", list_feature_columns()
                )
                X = X.reindex(columns=expected_cols, fill_value=0)

                # ── التنبؤ ──
                proba = model.predict_proba(X)[0]
                encoder = getattr(model, "label_encoder_", None)
                classes_model = (
                    encoder.classes_
                    if encoder
                    else list(getattr(model, "classes_", ["A", "D", "H"]))
                )
                prob_map = map_proba_to_HDA(classes_model, proba)

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء التنبؤ: {e}")
            st.exception(e)  # يعرض التفاصيل الكاملة في expander
            st.stop()

        # ── عرض النتائج ──
        st.subheader(
            f"📊 نتيجة التنبؤ: {meta['home_team_resolved']} vs {meta['away_team_resolved']}"
        )

        # تحديد الفائز المتوقع
        winner_key = max(prob_map, key=prob_map.get)
        winner_label = {
            "H": f"🏠 فوز {meta['home_team_resolved']}",
            "D": "🤝 تعادل",
            "A": f"✈️ فوز {meta['away_team_resolved']}",
        }

        st.info(f"**التوقع الأرجح:** {winner_label[winner_key]}  ({prob_map[winner_key]:.1%})")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "🏠 فوز المضيف (H)",
            f"{prob_map['H']:.1%}",
            delta="✅ الأرجح" if winner_key == "H" else None,
        )
        col2.metric(
            "🤝 تعادل (D)",
            f"{prob_map['D']:.1%}",
            delta="✅ الأرجح" if winner_key == "D" else None,
        )
        col3.metric(
            "✈️ فوز الضيف (A)",
            f"{prob_map['A']:.1%}",
            delta="✅ الأرجح" if winner_key == "A" else None,
        )

        # ← إصلاح #6: عرض شريط بصري للاحتمالات
        st.markdown("---")
        st.markdown("**توزيع الاحتمالات:**")
        prob_df = pd.DataFrame(
            {
                "النتيجة": ["فوز المضيف", "تعادل", "فوز الضيف"],
                "الاحتمال": [prob_map["H"], prob_map["D"], prob_map["A"]],
            }
        )
        st.bar_chart(prob_df.set_index("النتيجة"), height=250)
