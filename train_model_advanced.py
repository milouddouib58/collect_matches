# train_model_advanced.py
# -*- coding: utf-8 -*-
import argparse
import random
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import platform

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder # Import LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# نماذج إضافية إذا متوفرة
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

# skops للحفظ الآمن بين الإصدارات
try:
    from skops.io import dump as skops_dump
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False


from features_lib import list_feature_columns, FEATURE_VERSION

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_index = int(round(len(df_sorted) * (1 - test_size)))
    return df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]

def make_preprocessor():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def build_models_and_grids(preproc, inner_threads: int = 1):
    models = {}
    models["logreg"] = (
        Pipeline([("preproc", preproc),
                  ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"clf__C": [0.05, 0.1, 1.0]}
    )
    models["rf"] = (
        Pipeline([("preproc", preproc),
                  ("clf", RandomForestClassifier(random_state=SEED, n_jobs=inner_threads))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [10, 20]}
    )
    models["mlp"] = (
        Pipeline([("preproc", preproc),
                  ("clf", MLPClassifier(max_iter=500, random_state=SEED))]),
        {"clf__hidden_layer_sizes": [(64, 32)], "clf__alpha": [1e-3, 1e-4]}
    )
    if XGB_AVAILABLE:
        models["xgboost"] = (
            Pipeline([("preproc", preproc),
                      ("clf", XGBClassifier(use_label_encoder=False,
                                            eval_metric="mlogloss",
                                            random_state=SEED,
                                            n_jobs=inner_threads))]),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.03, 0.1]}
        )
    if LGBM_AVAILABLE:
        models["lightgbm"] = (
            Pipeline([("preproc", preproc),
                      ("clf", LGBMClassifier(random_state=SEED, n_jobs=inner_threads))]),
            {"clf__n_estimators": [200, 400],
             "clf__learning_rate": [0.03, 0.1]}
        )
    return models

def run_training(features_file: str,
                 league: str,
                 model_out: str,
                 cv_splits=3,
                 scoring="neg_log_loss",
                 n_jobs: int = None,
                 inner_threads: int = 1,
                 parallel_backend: str = None):
    """
    n_jobs: عدد عمليات GridSearchCV (يفضّل 1 على ويندوز لتفادي أخطاء loky)
    inner_threads: عدد خيوط النماذج الداخلية (RF/XGB/LGBM)
    parallel_backend: 'threading' أو 'loky' (threading آمن على ويندوز)
    """
    # إعدادات افتراضية آمنة على ويندوز
    is_windows = (platform.system().lower() == "windows")
    if n_jobs is None:
        n_jobs = 1 if is_windows else -1
    if parallel_backend is None:
        parallel_backend = "threading" if is_windows else "loky"


    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()

    # استخدم reindex لضمان وجود كل الأعمدة (المفقودة تُملأ NaN ويعالجها الـ Imputer)
    train_df, test_df = temporal_train_test_split(df)
    X_train, y_train = train_df.reindex(columns=feat_cols), train_df["target"]
    X_test, y_test = test_df.reindex(columns=feat_cols), test_df["target"]

    # Encode the target variable
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)


    preproc = make_preprocessor()
    models_and_grids = build_models_and_grids(preproc, inner_threads=inner_threads)

    best_estimators = {}
    all_results = {}

    print(f"🚀 Starting GridSearch for {len(models_and_grids)} models...")
    print(f"- OS: {platform.system()} | Python: {sys.version.split()[0]}")
    print(f"- GridSearch n_jobs: {n_jobs} | backend: {parallel_backend} | inner_threads: {inner_threads}")

    # استخدم backend آمن
    with joblib.parallel_backend(parallel_backend):
        for name, (pipe, grid) in models_and_grids.items():
            print(f"\n🔍 Searching for best {name}...")
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring=scoring,
                cv=TimeSeriesSplit(n_splits=cv_splits),
                n_jobs=n_jobs,            # آمن على ويندوز: 1
                verbose=1,
                error_score="raise"
            )
            gs.fit(X_train, y_train_encoded) # Use encoded target for training
            best_estimators[name] = gs.best_estimator_
            all_results[name] = {
                "best_score": float(gs.best_score_), # Convert to float
                "best_params": gs.best_params_
                }
            print(f"✅ Best {name}: score={gs.best_score_:.5f}")

    print("\n🧠 Building Ensemble Model...")
    top_models = sorted(all_results.items(), key=lambda item: item[1]['best_score'], reverse=True)[:3]
    ensemble_estimators = [(name, best_estimators[name]) for name, _ in top_models]
    print("Combining models:", [name for name, _ in ensemble_estimators])

    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    voting_clf.fit(X_train, y_train_encoded) # Use encoded target for training

    test_proba = voting_clf.predict_proba(X_test)
    test_loss = log_loss(y_test_encoded, test_proba, labels=voting_clf.classes_) # Use encoded target for loss calculation
    print(f"\n🏆 Ensemble Model Test LogLoss: {test_loss:.5f}")

    # خصائص للموديل لتوافق التطبيق
    import sklearn
    setattr(voting_clf, "feature_version_", FEATURE_VERSION)
    setattr(voting_clf, "sklearn_version_", sklearn.__version__)
    setattr(voting_clf, "python_version_", sys.version)
    setattr(voting_clf, "feature_names_expected_", feat_cols)
    # Store the encoder as well
    setattr(voting_clf, "label_encoder_", encoder)

    # حفظ .joblib
    joblib.dump(voting_clf, model_out)
    print(f"✅ Saved joblib: {model_out}")

    # حفظ .skops (مستحسن للنقل بين البيئات)
    if SKOPS_AVAILABLE:
        skops_path = model_out.replace(".joblib", ".skops")
        skops_dump(voting_clf, skops_path, metadata={
            "feature_version": FEATURE_VERSION,
            "sklearn_version": sklearn.__version__,
            "python_version": sys.version,
            "league": league,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"✅ Saved skops: {skops_path}")
    else:
        print("⚠️ skops not available. Skipping .skops save.")


    metadata = {
        "league": league,
        "best_individual_models": all_results,
        "ensemble_components": [name for name, _ in top_models],
        "ensemble_test_logloss": float(test_loss),
        "classes_": list(voting_clf.classes_),
        "feature_version": FEATURE_VERSION,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features_count": len(feat_cols),
        "gridsearch_n_jobs": n_jobs,
        "parallel_backend": parallel_backend,
        "inner_threads": inner_threads
    }
    with open(model_out.replace(".joblib", "_metadata.json"), 'w', encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Train ensemble model for match outcome prediction.")
    parser.add_argument("--features-file", type=str, default=None, help="Path to engineered features CSV (default: features_{league}.csv)")
    parser.add_argument("--league", type=str, default="PL", help="League code")
    parser.add_argument("--model-out", type=str, default="ensemble_model_v3_PL.joblib", help="Output path (.joblib); .skops will also be saved if available")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=None, help="GridSearch parallel jobs (use 1 on Windows)")
    parser.add_argument("--inner-threads", type=int, default=1, help="Threads for inner models (RF/XGB/LGBM)")
    parser.add_argument("--parallel-backend", type=str, choices=["threading", "loky"], default=None, help="Joblib backend (threading recommended on Windows)")
    return parser


def main(argv=None):
    parser = get_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    features_file = args.features_file or f"features_{args.league}.csv"
    if not Path(features_file).exists():
        raise FileNotFoundError(f"Features file not found: {features_file}. Generate it with engineer_features.py first.")

    run_training(
        features_file=features_file,
        league=args.league,
        model_out=args.model_out,
        cv_splits=args.cv_splits,
        scoring="neg_log_loss",
        n_jobs=args.n_jobs,
        inner_threads=args.inner_threads,
        parallel_backend=args.parallel_backend
    )

if __name__ == "__main__":
    main()
