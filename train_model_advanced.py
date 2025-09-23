# train_model_advanced.py
# -*- coding: utf-8 -*-
import argparse
import random
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# نماذج إضافية إذا متوفرة
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_index = int(round(len(df_sorted) * (1 - test_size)))
    return df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]

def make_preprocessor(feature_cols):
    return ColumnTransformer(transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), feature_cols)
    ])

def build_models_and_grids(preproc):
    models = {}
    models["logreg"] = (
        Pipeline([("preproc", preproc), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.05, 0.1, 1.0]}
    )
    models["rf"] = (
        Pipeline([("preproc", preproc), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [10, 20]}
    )
    models["mlp"] = (
        Pipeline([("preproc", preproc), ("clf", MLPClassifier(max_iter=500, random_state=SEED))]),
        {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [1e-3, 1e-4]}
    )
    if XGB_AVAILABLE:
        models["xgboost"] = (
            Pipeline([("preproc", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5], "clf__learning_rate": [0.03, 0.1]}
        )
    if LGBM_AVAILABLE:
        models["lightgbm"] = (
            Pipeline([("preproc", preproc), ("clf", LGBMClassifier(random_state=SEED))]),
            {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.03, 0.1]}
        )
    return models

def run_training(features_file: str, league: str, model_out: str, cv_splits=3, scoring="neg_log_loss"):
    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()

    # التدريب على الفئات النصية مباشرة ("H","D","A")
    train_df, test_df = temporal_train_test_split(df)
    X_train, y_train = train_df[feat_cols], train_df["target"]
    X_test, y_test = test_df[feat_cols], test_df["target"]

    preproc = make_preprocessor(feat_cols)
    models_and_grids = build_models_and_grids(preproc)

    best_estimators = {}
    all_results = {}

    print(f"🚀 Starting GridSearch for {len(models_and_grids)} models...")
    for name, (pipe, grid) in models_and_grids.items():
        print(f"\n🔍 Searching for best {name}...")
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=TimeSeriesSplit(n_splits=cv_splits),
            n_jobs=-1,
            verbose=1
        )
        gs.fit(X_train, y_train)
        best_estimators[name] = gs.best_estimator_
        all_results[name] = {"best_score": gs.best_score_, "best_params": gs.best_params_}
        print(f"✅ Best {name}: score={gs.best_score_:.5f}")

    print("\n🧠 Building Ensemble Model...")
    top_models = sorted(all_results.items(), key=lambda item: item[1]['best_score'], reverse=True)[:3]
    ensemble_estimators = [(name, best_estimators[name]) for name, _ in top_models]
    print("Combining models:", [name for name, _ in ensemble_estimators])

    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    voting_clf.fit(X_train, y_train)

    test_proba = voting_clf.predict_proba(X_test)
    test_loss = log_loss(y_test, test_proba, labels=voting_clf.classes_)
    print(f"\n🏆 Ensemble Model Test LogLoss: {test_loss:.5f}")

    # تخزين نسخة الميزات داخل الكائن
    setattr(voting_clf, "feature_version_", FEATURE_VERSION)

    joblib.dump(voting_clf, model_out)
    print(f"\n✅ Final Ensemble model saved to: {model_out}")

    metadata = {
        "league": league,
        "best_individual_models": all_results,
        "ensemble_components": [name for name, _ in top_models],
        "ensemble_test_logloss": float(test_loss),
        "classes_": list(voting_clf.classes_),
        "feature_version": FEATURE_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(model_out.replace(".joblib", "_metadata.json"), 'w', encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Train ensemble model for match outcome prediction.")
    parser.add_argument("--features-file", type=str, required=True, help="Path to engineered features CSV")
    parser.add_argument("--league", type=str, default="PL", help="League code (PL, PD, SA, BL1, FL1...)")
    parser.add_argument("--model-out", type=str, default="ensemble_model_v3_PL.joblib", help="Output path for the model")
    parser.add_argument("--cv-splits", type=int, default=3)
    args = parser.parse_args()

    run_training(
        features_file=args.features_file,
        league=args.league,
        model_out=args.model_out,
        cv_splits=args.cv_splits,
        scoring="neg_log_loss"
    )

if __name__ == "__main__":
    main()
