# train_grid_search.py
# -*- coding: utf-8 -*-
"""
شغّل:
# في Colab قد تحتاج لتثبيت الحزم التالية أولًا:
# !pip install xgboost lightgbm

python train_grid_search.py --features features_PL.csv --league PL --model-out best_grid_model.joblib
أو في Colab:
from train_grid_search import run_main
run_main("features_PL.csv", "PL", "best_grid_model.joblib")
"""

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
# from sklearn.neural_network import MLPClassifier # Temporarily removed due to import error

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(SEED)


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    split_index = int(round(n * (1 - test_size)))
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    return train_df, test_df


def evaluate_on_test(pipeline, X_test, y_test_raw, encoder):
    # pipeline may be (preprocessor, keras_model) tuple for fallback; here we only evaluate sklearn pipelines
    if isinstance(pipeline, tuple):
        preproc, keras_model = pipeline
        X_test_proc = preproc.transform(X_test)
        proba = keras_model.predict(X_test_proc)
    else:
        proba = pipeline.predict_proba(X_test)

    preds_idx = np.argmax(proba, axis=1)
    preds = encoder.inverse_transform(preds_idx)
    acc = accuracy_score(y_test_raw, preds)
    loss = log_loss(y_test_raw, proba, labels=encoder.classes_)
    cm = confusion_matrix(y_test_raw, preds, labels=encoder.classes_)
    report = classification_report(y_test_raw, preds, output_dict=True, zero_division=0)
    return {"accuracy": float(acc), "log_loss": float(loss), "confusion_matrix": cm.tolist(), "report": report}


def make_preprocessor():
    # Simple numeric-only preprocessor (expand if you have categorical features)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # apply to all feature columns
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
        "clf__penalty": ["l2"],  # 'l1' requires solver change; keep l2 for stability
        "clf__solver": ["lbfgs"]
    }
    models["logreg"] = (pipe_log, grid_log)

    # RandomForest
    pipe_rf = Pipeline([
        ("preproc", preproc),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))
    ])
    grid_rf = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__class_weight": [None, "balanced", "balanced_subsample"]
    }
    models["rf"] = (pipe_rf, grid_rf)

    # MLP (sklearn) - Temporarily removed due to import error
    # pipe_mlp = Pipeline([
    #     ("preproc", preproc),
    #     ("clf", MLPClassifier(max_iter=500, random_state=SEED))
    # ])
    # grid_mlp = {
    #     "clf__hidden_layer_sizes": [(64, 32), (128, 64), (64,)],
    #     "clf__alpha": [1e-4, 1e-3, 1e-2],
    #     "clf__learning_rate_init": [1e-3, 5e-4]
    # }
    # models["mlp"] = (pipe_mlp, grid_mlp)

    # XGBoost (if available)
    if XGB_AVAILABLE:
        pipe_xgb = Pipeline([
            ("preproc", preproc),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=SEED, verbosity=0))
        ])
        grid_xgb = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 6, 10],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__subsample": [0.7, 1.0]
        }
        models["xgboost"] = (pipe_xgb, grid_xgb)
    else:
        print("⚠️ xgboost not installed — skipping XGBoost model (install via pip install xgboost)")

    # LightGBM (if available)
    if LGBM_AVAILABLE:
        pipe_lgb = Pipeline([
            ("preproc", preproc),
            ("clf", LGBMClassifier(random_state=SEED))
        ])
        grid_lgb = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [-1, 8, 16],
            "clf__learning_rate": [0.01, 0.1],
            "clf__num_leaves": [31, 63]
        }
        models["lightgbm"] = (pipe_lgb, grid_lgb)
    else:
        print("⚠️ lightgbm not installed — skipping LightGBM model (install via pip install lightgbm)")

    return models


def run_grid_search_for_model(name, pipeline, param_grid, X_train, y_train, cv, scoring="neg_log_loss", n_jobs=-1, verbose=2):
    """
    Run GridSearchCV and return fitted GridSearchCV object.
    """
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=n_jobs,
        verbose=verbose
    )
    print(f"\n>>> Running GridSearchCV for {name} with {len(param_grid.keys())} hyperparameter groups ...")
    gs.fit(X_train, y_train)
    print(f">>> Done {name}: best_score={gs.best_score_:.4f}, best_params={gs.best_params_}")
    return gs


def run_training(features_file, league, model_out, cv_splits=3, scoring="neg_log_loss"):
    # read
    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()
    if not set(feat_cols).issubset(set(df.columns)):
        raise ValueError(f"Some feature columns missing in {features_file}. Expected: {feat_cols}")

    # encode labels
    encoder = LabelEncoder()
    df["target_encoded"] = encoder.fit_transform(df["target"])

    # temporal split
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)
    X_train, y_train = train_df[feat_cols], train_df["target"]
    X_test, y_test = test_df[feat_cols], test_df["target"]

    # cross-validation strategy: TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    models = build_models_and_grids()

    all_results = {}
    best_overall_score = -np.inf  # because scoring is neg_log_loss, higher (closer to 0) is better
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

        # evaluate best estimator on test
        test_metrics = evaluate_on_test(gs.best_estimator_, X_test, y_test, encoder)
        all_results[name] = {
            "best_params": gs.best_params_,
            "cv_best_score": float(gs.best_score_),
            "test_metrics": test_metrics,
            "cv_results": gs.cv_results_
        }
        print(f"Test metrics for {name}: acc={test_metrics['accuracy']:.4f}, logloss={test_metrics['log_loss']:.4f}")

        # choose best by cv score (higher is better because neg_log_loss)
        if float(gs.best_score_) > best_overall_score:
            best_overall_score = float(gs.best_score_)
            best_artifact = gs.best_estimator_
            best_model_name = name

    # Save results and best model
    artifact = {
        "feature_cols": feat_cols,
        "labels": list(encoder.classes_),
        "competition": league,
        "feature_version": FEATURE_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model_name": best_model_name,
        "best_cv_score_neg_logloss": float(best_overall_score),
        "all_results": all_results
    }

    # Save the pipeline/model
    joblib.dump(artifact, model_out)
    # save the actual best estimator separately for inference
    if best_artifact is not None:
        best_model_path = model_out.replace(".joblib", f"_{best_model_name}_estimator.joblib")
        joblib.dump(best_artifact, best_model_path)
        print(f"\n✅ Best estimator saved to: {best_model_path}")

    print(f"\n✅ All metadata/results saved to: {model_out}")
    return artifact


def run_main(features_file="features_PL.csv", league="PL", model_out="best_grid_model.joblib"):
    return run_training(features_file, league, model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="features_PL.csv")
    parser.add_argument("--league", type=str, default="PL")
    parser.add_argument("--model-out", type=str, default="best_grid_model.joblib")
    parser.add_argument("--cv-splits", type=int, default=3)
    # ignore unknown args from Colab
    args, unknown = parser.parse_known_args()
    run_training(args.features, args.league, args.model_out, cv_splits=args.cv_splits)