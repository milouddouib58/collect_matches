import pandas as pd
from features_lib import engineer_match_features
from train_model_advanced import run_training

# --- الخطوة 1: إعداد البيانات ---
input_matches_file = "matches_data.csv"
try:
    matches = pd.read_csv(input_matches_file)
    print(f"👍 تم تحميل {input_matches_file} بنجاح.")
except FileNotFoundError:
    print(f"❌ خطأ: تأكد من رفع ملف '{input_matches_file}' إلى جلسة Colab.")

# --- الخطوة 2: هندسة الميزات ---
league_code = "PL"
features_output_file = f"features_v3_{league_code}.csv"
print("\n⚙️  بدء هندسة الميزات (قد تستغرق هذه الخطوة بضع دقائق)...")
features_df = engineer_match_features(matches, competition=league_code)
features_df.to_csv(features_output_file, index=False)
print(f"✅ تم حفظ الميزات في '{features_output_file}'")

# --- الخطوة 3: تدريب النموذج ---
model_output_file = f"ensemble_model_v3_{league_code}.joblib"
print("\n🧠 بدء تدريب النموذج (هذه الخطوة قد تكون طويلة جدًا، تحل بالصبر)...")
run_training(features_file=features_output_file, league=league_code, model_out=model_output_file)

print("\n🎉🎉🎉 انتهى المشروع! 🎉🎉🎉")
print(f"النموذج النهائي جاهز في: '{model_output_file}'")
