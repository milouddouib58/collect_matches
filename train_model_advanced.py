# train_model_advanced.py
# -*- coding: utf-8 -*-
import argparse
import sys
import os
import random
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import joblib

# تثبيت البذرة للتكرارية
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# حاول استيراد XGBoost و LightGBM إن أمكن
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# Import your features list from project
from features_lib import list_feature_columns, FEATURE_VERSION

# تم حذف جزء TensorFlow لتبسيط الحل والتركيز على نماذج sklearn
# import tensorflow as tf ...


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    split_index = int(round(n * (1 - test_size)))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    return train_df, test_df


def evaluate_on_test(pipeline, X_test, y_test_raw, encoder):
    proba = pipeline.predict_proba(X_test)
    preds_idx = np.argmax(proba, axis=1)
    preds = encoder.inverse_transform(preds_idx)
    acc = accuracy_score(y_test_raw, preds)
    loss = log_loss(y_test_raw, proba, labels=encoder.classes_)
    cm = confusion_matrix(y_test_raw, preds, labels=encoder.classes_)
    report = classification_report(y_test_raw, preds, output_dict=True, zero_division=0)
    return {"accuracy": float(acc), "log_loss": float(loss), "confusion_matrix": cm.tolist(), "report": report}


def make_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(transformers=[("num", numeric_transformer, list_feature_columns())])


def build_models_and_grids():
    """Return dict of {name: (estimator_pipeline, param_grid)}"""
    preproc = make_preprocessor()
    models = {}

    # LogisticRegression
    pipe_log = Pipeline([
        ("preproc", preproc),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED))
    ])
    grid_log = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    models["logreg"] = (pipe_log, grid_log)

    # RandomForest
    pipe_rf = Pipeline([
        ("preproc", preproc),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    grid_rf = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 5],
        "clf__class_weight": [None, "balanced"]
    }
    models["rf"] = (pipe_rf, grid_rf)

    if XGB_AVAILABLE:
        pipe_xgb = Pipeline([
            ("preproc", preproc),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED, verbosity=0))
        ])
        grid_xgb = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.01, 0.1],
            "clf__subsample": [0.7, 1.0]
        }
        models["xgboost"] = (pipe_xgb, grid_xgb)
    else:
        print("⚠️ xgboost not installed — skipping XGBoost model (install via pip install xgboost)")

    if LGBM_AVAILABLE:
        pipe_lgb = Pipeline([
            ("preproc", preproc),
            ("clf", LGBMClassifier(random_state=SEED))
        ])
        grid_lgb = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [-1, 16],
            "clf__learning_rate": [0.01, 0.1],
            "clf__num_leaves": [31, 63]
        }
        models["lightgbm"] = (pipe_lgb, grid_lgb)
    else:
        print("⚠️ lightgbm not installed — skipping LightGBM model (install via pip install lightgbm)")

    return models


def run_grid_search_for_model(name, pipeline, param_grid, X_train, y_train, cv, scoring="neg_log_loss", n_jobs=-1, verbose=1):
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        verbose=verbose
    )
    print(f"\n>>> Running GridSearchCV for {name} ...")
    gs.fit(X_train, y_train)
    print(f">>> Done {name}: best_score={gs.best_score_:.4f}, best_params={gs.best_params_}")
    return gs


def run_training(features_file, league, model_out, cv_splits=3, scoring="neg_log_loss"):
    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()
    if not set(feat_cols).issubset(set(df.columns)):
        raise ValueError(f"Some feature columns missing in {features_file}. Expected: {feat_cols}")

    encoder = LabelEncoder()
    df["target_encoded"] = encoder.fit_transform(df["target"])

    train_df, test_df = temporal_train_test_split(df, test_size=0.2)
    
    # ✅ *** FIX 1: Use the encoded target for training ***
    X_train, y_train = train_df[feat_cols], train_df["target_encoded"]
    X_test, y_test = test_df[feat_cols], test_df["target"]

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    models = build_models_and_grids()

    all_results = {}
    best_overall_score = -np.inf
    best_artifact = None
    best_model_name = None

    for name, (pipe, grid) in models.items():
        print("\n" + "=" * 80)
        print(f"Grid search for model: {name}")
        try:
            gs = run_grid_search_for_model(name, pipe, grid, X_train, y_train, cv=tscv, scoring=scoring)
        except Exception as e:
            print(f"!!! GridSearch failed for {name}: {e}")
            continue

        test_metrics = evaluate_on_test(gs.best_estimator_, X_test, y_test, encoder)
        all_results[name] = {
            "best_params": gs.best_params_,
            "cv_best_score": float(gs.best_score_),
            "test_metrics": test_metrics,
            "cv_results": gs.cv_results_
        }
        print(f"Test metrics for {name}: acc={test_metrics['accuracy']:.4f}, logloss={test_metrics['log_loss']:.4f}")

        if float(gs.best_score_) > best_overall_score:
            best_overall_score = float(gs.best_score_)
            best_artifact = gs.best_estimator_
            best_model_name = name

    # ✅ *** FIX 2: Save the best pipeline directly ***
    if best_artifact is not None:
        print(f"\n🏆 Best model found: {best_model_name} (CV score: {best_overall_score:.4f})")
        
        # Save important metadata as attributes of the pipeline object itself
        best_artifact.feature_version_ = FEATURE_VERSION
        best_artifact.feature_cols_ = feat_cols
        best_artifact.labels_ = list(encoder.classes_)
        best_artifact.trained_at_utc_ = datetime.now(timezone.utc).isoformat()
        best_artifact.training_league_ = league

        joblib.dump(best_artifact, model_out)
        print(f"\n✅ Best pipeline saved directly to: {model_out}")

        # Optionally save full metadata to a separate JSON for analysis
        results_path = model_out.replace(".joblib", "_full_results.json")
        for model_name, data in all_results.items():
            if 'cv_results' in data:
                for k, v in data['cv_results'].items():
                    if isinstance(v, np.ndarray):
                        data['cv_results'][k] = v.tolist()
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"ℹ️ Full training metadata/results saved to: {results_path}")

    else:
        print("\n❌ No best model could be determined. Nothing was saved.")

    return best_artifact


def run_main(features_file="features_PL.csv", league="PL", model_out="best_model.joblib"):
    return run_training(features_file, league, model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="features_PL.csv", help="Input features CSV file.")
    parser.add_argument("--league", type=str, default="PL", help="League code (e.g., PL, PD).")
    parser.add_argument("--model-out", type=str, default="model.joblib", help="Path to save the final model pipeline.")
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of splits for TimeSeriesSplit CV.")
    args, unknown = parser.parse_known_args()
    run_training(args.features, args.league, args.model_out, cv_splits=args.cv_splits)

