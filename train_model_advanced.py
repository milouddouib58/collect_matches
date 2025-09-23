# train_model_advanced.py
# -*- coding: utf-8 -*-
import argparse
import random
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import joblib
from operator import itemgetter

# تثبيت البذرة للتكرارية
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier # <--- تم تفعيل الشبكات العصبونية

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from features_lib import list_feature_columns, FEATURE_VERSION

def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    split_index = int(round(n * (1 - test_size)))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    return train_df, test_df

def make_preprocessor(feature_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(transformers=[("num", numeric_transformer, feature_cols)])

def build_models_and_grids(preproc):
    models = {}
    # LogisticRegression
    models["logreg"] = (
        Pipeline([("preproc", preproc), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__solver": ["lbfgs"]}
    )
    # RandomForest
    models["rf"] = (
        Pipeline([("preproc", preproc), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [10, 20], "clf__min_samples_leaf": [5, 10]}
    )
    # --- تفعيل الشبكة العصبونية MLP ---
    models["mlp"] = (
        Pipeline([("preproc", preproc), ("clf", MLPClassifier(max_iter=500, random_state=SEED))]),
        {"clf__hidden_layer_sizes": [(64, 32), (128,)], "clf__alpha": [1e-4, 1e-3], "clf__learning_rate_init": [1e-3]}
    )
    if XGB_AVAILABLE:
        models["xgboost"] = (
            Pipeline([("preproc", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED))]),
            # --- توسيع شبكة البحث لـ XGBoost ---
            {"clf__n_estimators": [100, 300, 500], "clf__max_depth": [3, 5], "clf__learning_rate": [0.01, 0.1], "clf__subsample": [0.7, 1.0]}
        )
    if LGBM_AVAILABLE:
        models["lightgbm"] = (
            Pipeline([("preproc", preproc), ("clf", LGBMClassifier(random_state=SEED))]),
            {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.01, 0.1], "clf__num_leaves": [31, 63]}
        )
    return models

def run_training(features_file, league, model_out, cv_splits=3, scoring="neg_log_loss"):
    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()
    df = df.dropna(subset=feat_cols, how='any') # تجاهل أي صفوف بها قيم ميزات فارغة

    encoder = LabelEncoder()
    df["target_encoded"] = encoder.fit_transform(df["target"])
    
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)
    X_train, y_train_encoded = train_df[feat_cols], train_df["target_encoded"]
    X_test, y_test = test_df[feat_cols], test_df["target"]

    preproc = make_preprocessor(feat_cols)
    models_and_grids = build_models_and_grids(preproc)
    
    best_estimators = {}
    all_results = {}
    
    print(f"🚀 Starting GridSearch for {len(models_and_grids)} models...")
    for name, (pipe, grid) in models_and_grids.items():
        print("\n" + "="*50 + f"\n🔍 Searching for best {name}...")
        gs = GridSearchCV(estimator=pipe, param_grid=grid, scoring=scoring, cv=TimeSeriesSplit(n_splits=cv_splits), n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train_encoded)
        
        best_estimators[name] = gs.best_estimator_
        all_results[name] = {"best_score": gs.best_score_, "best_params": gs.best_params_}
        print(f"✅ Best {name}: score={gs.best_score_:.4f}, params={gs.best_params_}")

    # --- بناء نموذج الدمج (Ensemble) ---
    print("\n" + "="*50 + "\n🧠 Building Ensemble Model...")
    
    # اختيار أفضل 3 نماذج بناءً على نتيجة التقييم
    top_models = sorted(all_results.items(), key=lambda item: item[1]['best_score'], reverse=True)[:3]
    
    ensemble_estimators = [(name, best_estimators[name]) for name, data in top_models]
    
    print("Combining models:", [name for name, est in ensemble_estimators])
    
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    voting_clf.fit(X_train, y_train_encoded)

    # تقييم النموذج المدمج
    test_proba = voting_clf.predict_proba(X_test)
    test_loss = log_loss(y_test, test_proba)
    print(f"\n🏆 Ensemble Model Test LogLoss: {test_loss:.4f}")

    # حفظ النموذج المدمج النهائي
    joblib.dump(voting_clf, model_out)
    print(f"\n✅ Final Ensemble model saved to: {model_out}")
    
    # حفظ بيانات وصفية للتشغيل
    metadata = {
        "best_individual_models": all_results,
        "ensemble_components": [name for name, data in top_models],
        "ensemble_test_logloss": test_loss,
        "feature_version": FEATURE_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(model_out.replace(".joblib", "_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Input features CSV file.")
    parser.add_argument("--league", required=True, help="League code (e.g., PL).")
    parser.add_argument("--model-out", default="model.joblib", help="Path to save the final model.")
    args = parser.parse_args()
    run_training(args.features, args.league, args.model_out)
