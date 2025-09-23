# train_model.py
# -*- coding: utf-8 -*-
import argparse
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from features_lib import list_feature_columns, RESULT_LABELS, FEATURE_VERSION


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    split = int(round(n * (1 - test_size)))
    return df_sorted.iloc[:split], df_sorted.iloc[split:]


def build_pipelines():
    feat_steps_base = [("imputer", SimpleImputer(strategy="median"))]

    pipe_log = Pipeline(steps=[
        *feat_steps_base,
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    pipe_rf = Pipeline(steps=[
        *feat_steps_base,
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])
    return {"logreg": pipe_log, "rf": pipe_rf}


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)
    preds = model.predict(X)
    y_true = y

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "log_loss": float(log_loss(y_true, proba, labels=model.named_steps["clf"].classes_))
    }
    
    cm = confusion_matrix(y_true, preds, labels=model.named_steps["clf"].classes_).tolist()
    metrics["confusion_matrix"] = {
        "labels_order": list(map(str, model.named_steps["clf"].classes_)),
        "matrix": cm
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="تدريب نموذج توقع نتائج المباريات.")
    parser.add_argument("--features", type=str, default="features_PL.csv", help="ملف ميزات CSV")
    parser.add_argument("--league", type=str, default="PL", help="رمز الدوري للتدريب (للتوثيق فقط)")
    parser.add_argument("--model-out", type=str, default=None, help="مسار حفظ النموذج .joblib")
    parser.add_argument("--metrics-out", type=str, default=None, help="مسار ملف التقييم .json")
    args = parser.parse_args()

    model_path = args.model_out or f"model_{args.league}_{FEATURE_VERSION}.joblib"
    metrics_path = args.metrics_out or f"metrics_{args.league}_{FEATURE_VERSION}.json"

    df = pd.read_csv(args.features)
    feat_cols = list_feature_columns()

    # تأكد من الأعمدة
    for c in feat_cols:
        if c not in df.columns:
            raise ValueError(f"العمود المفقود في الميزات: {c}")

    X = df[feat_cols].copy()
    y = df["target"].copy()

    # تقسيم زمني
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)
    X_train, y_train = train_df[feat_cols], train_df["target"]
    X_test, y_test = test_df[feat_cols], test_df["target"]

    pipelines = build_pipelines()
    results = {}
    best_name, best_model, best_logloss = None, None, np.inf

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        train_metrics = evaluate(pipe, X_train, y_train)
        test_metrics = evaluate(pipe, X_test, y_test)
        results[name] = {"train": train_metrics, "test": test_metrics}

        if test_metrics["log_loss"] < best_logloss:
            best_logloss = test_metrics["log_loss"]
            best_name = name
            best_model = pipe

    # حفظ النموذج مع بيانات التعريف
    artifact = {
        "pipeline": best_model,
        "feature_cols": feat_cols,
        "labels": RESULT_LABELS,
        "competition": args.league,
        "feature_version": FEATURE_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_models_metrics": results,
        "chosen_model": best_name
    }
    joblib.dump(artifact, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "league": args.league,
            "feature_version": FEATURE_VERSION,
            "chosen_model": best_name,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"- تم حفظ النموذج: {model_path}")
    print(f"- تم حفظ تقرير القياس: {metrics_path}")
    print(f"- النموذج المختار: {best_name} | LogLoss={best_logloss:.4f}")


if __name__ == "__main__":
    main()

