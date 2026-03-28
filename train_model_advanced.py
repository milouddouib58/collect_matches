# train_model_advanced.py
# -*- coding: utf-8 -*-
"""
تدريب نموذج متقدم مع:
- معايرة احتمالات (CalibratedClassifierCV)
- تقرير أهمية الميزات
- دعم CatBoost
"""

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    log_loss,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV  # ← جديد
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,  # ← جديد كبديل
)
from sklearn.neural_network import MLPClassifier

# نماذج إضافية
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

try:
    from catboost import CatBoostClassifier  # ← جديد
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from skops.io import dump as skops_dump
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False

from features_lib import list_feature_columns, FEATURE_VERSION

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def temporal_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2
):
    """تقسيم زمني: أقدم 80% للتدريب، أحدث 20% للاختبار."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_index = int(round(len(df_sorted) * (1 - test_size)))
    return df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]


def make_preprocessor():
    """معالج أولي: ملء القيم المفقودة + تطبيع."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_models_and_grids(preproc, inner_threads: int = 1):
    """بناء النماذج وشبكات البحث."""
    models = {}

    # 1. Logistic Regression
    models["logreg"] = (
        Pipeline([
            ("preproc", preproc),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=SEED,
                multi_class="multinomial",  # ← أفضل لـ 3 فئات
            )),
        ]),
        {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]},
    )

    # 2. Random Forest
    models["rf"] = (
        Pipeline([
            ("preproc", preproc),
            ("clf", RandomForestClassifier(
                random_state=SEED,
                n_jobs=inner_threads,
            )),
        ]),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [8, 12, 20],
            "clf__min_samples_leaf": [5, 10],  # ← جديد: تقليل الـ overfitting
        },
    )

    # 3. MLP
    models["mlp"] = (
        Pipeline([
            ("preproc", preproc),
            ("clf", MLPClassifier(
                max_iter=500,
                random_state=SEED,
                early_stopping=True,  # ← جديد: إيقاف مبكر
                validation_fraction=0.15,
            )),
        ]),
        {
            "clf__hidden_layer_sizes": [(128, 64), (64, 32)],
            "clf__alpha": [1e-3, 1e-4],
        },
    )

    # 4. XGBoost
    if XGB_AVAILABLE:
        models["xgboost"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=SEED,
                    n_jobs=inner_threads,
                )),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.8, 1.0],  # ← جديد
            },
        )

    # 5. LightGBM
    if LGBM_AVAILABLE:
        models["lightgbm"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", LGBMClassifier(
                    random_state=SEED,
                    n_jobs=inner_threads,
                    verbose=-1,  # ← إسكات التحذيرات
                )),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__learning_rate": [0.03, 0.1],
                "clf__num_leaves": [31, 63],  # ← جديد
            },
        )

    # 6. CatBoost ← جديد بالكامل
    if CATBOOST_AVAILABLE:
        models["catboost"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", CatBoostClassifier(
                    random_state=SEED,
                    verbose=0,
                    thread_count=inner_threads,
                )),
            ]),
            {
                "clf__iterations": [300, 500],
                "clf__depth": [4, 6],
                "clf__learning_rate": [0.03, 0.1],
            },
        )

    # 7. Gradient Boosting (بديل إذا لم يتوفر XGB/LGBM)
    if not XGB_AVAILABLE and not LGBM_AVAILABLE:
        models["gboost"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", GradientBoostingClassifier(
                    random_state=SEED,
                )),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.1],
            },
        )

    return models


def print_feature_importance(model, feat_cols: list, top_n: int = 15):
    """
    ← جديد: طباعة أهم الميزات من النموذج.
    """
    print(f"\n📊 أهم {top_n} ميزة:")
    print("-" * 45)

    importances = None

    # محاولة استخراج الأهمية من النماذج المختلفة
    for name, estimator in (
        model.named_estimators_.items()
        if hasattr(model, "named_estimators_")
        else []
    ):
        clf = estimator
        # الوصول للمصنف داخل Pipeline
        if hasattr(clf, "named_steps"):
            clf = clf.named_steps.get("clf", clf)

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            print(f"  (من نموذج: {name})")
            break
        elif hasattr(clf, "coef_"):
            importances = np.mean(np.abs(clf.coef_), axis=0)
            print(f"  (من نموذج: {name})")
            break

    if importances is not None and len(importances) == len(feat_cols):
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        for i, row in imp_df.head(top_n).iterrows():
            bar = "█" * int(row["importance"] / imp_df["importance"].max() * 20)
            print(f"  {row['feature']:.<35s} {row['importance']:.4f} {bar}")
    else:
        print("  (لم يتم استخراج أهمية الميزات)")


def run_training(
    features_file: str,
    league: str,
    model_out: str,
    cv_splits=3,
    scoring="neg_log_loss",
    n_jobs: int = None,
    inner_threads: int = 1,
    parallel_backend: str = None,
    calibrate: bool = True,  # ← جديد
):
    """التدريب الرئيسي مع معايرة الاحتمالات."""

    is_windows = platform.system().lower() == "windows"
    if n_jobs is None:
        n_jobs = 1 if is_windows else -1
    if parallel_backend is None:
        parallel_backend = "threading" if is_windows else "loky"

    # ── تحميل البيانات ──
    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()

    train_df, test_df = temporal_train_test_split(df)
    X_train = train_df.reindex(columns=feat_cols)
    y_train = train_df["target"]
    X_test = test_df.reindex(columns=feat_cols)
    y_test = test_df["target"]

    print(f"📋 حجم التدريب: {len(X_train)} | حجم الاختبار: {len(X_test)}")
    print(f"📋 توزيع الهدف (تدريب): {dict(y_train.value_counts())}")
    print(f"📋 توزيع الهدف (اختبار): {dict(y_test.value_counts())}")

    # ── ترميز الهدف ──
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    print(f"📋 ترتيب الفئات: {list(encoder.classes_)}")

    # ── بناء النماذج ──
    preproc = make_preprocessor()
    models_and_grids = build_models_and_grids(preproc, inner_threads)

    best_estimators = {}
    all_results = {}

    print(f"\n🚀 بدء GridSearch لـ {len(models_and_grids)} نموذج...")
    print(f"  OS: {platform.system()} | Python: {sys.version.split()[0]}")
    print(f"  n_jobs: {n_jobs} | backend: {parallel_backend}")

    with joblib.parallel_backend(parallel_backend):
        for name, (pipe, grid) in models_and_grids.items():
            print(f"\n🔍 بحث: {name}...")
            try:
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring=scoring,
                    cv=TimeSeriesSplit(n_splits=cv_splits),
                    n_jobs=n_jobs,
                    verbose=1,
                    error_score="raise",
                )
                gs.fit(X_train, y_train_enc)
                best_estimators[name] = gs.best_estimator_
                all_results[name] = {
                    "best_score": float(gs.best_score_),
                    "best_params": {
                        k: (str(v) if not isinstance(v, (int, float)) else v)
                        for k, v in gs.best_params_.items()
                    },
                }
                print(f"  ✅ {name}: score={gs.best_score_:.5f}")
            except Exception as e:
                print(f"  ❌ {name} فشل: {e}")
                continue

    if not best_estimators:
        raise RuntimeError("لم ينجح أي نموذج في التدريب!")

    # ── بناء Ensemble ──
    print("\n🧠 بناء النموذج المجمّع (Ensemble)...")
    top_models = sorted(
        all_results.items(),
        key=lambda item: item[1]["best_score"],
        reverse=True,
    )[:3]
    ensemble_estimators = [
        (name, best_estimators[name]) for name, _ in top_models
    ]
    print(f"  النماذج المختارة: {[n for n, _ in ensemble_estimators]}")

    voting_clf = VotingClassifier(
        estimators=ensemble_estimators, voting="soft"
    )
    voting_clf.fit(X_train, y_train_enc)

    # ── معايرة الاحتمالات ← جديد ──
    if calibrate:
        print("\n📐 معايرة الاحتمالات (Calibration)...")
        # تقسيم بيانات المعايرة من نهاية التدريب
        cal_size = min(300, len(X_train) // 5)
        X_cal = X_train.iloc[-cal_size:]
        y_cal = y_train_enc[-cal_size:]
        X_train_pre = X_train.iloc[:-cal_size]
        y_train_pre = y_train_enc[:-cal_size]

        # إعادة تدريب بدون بيانات المعايرة
        voting_clf_pre = VotingClassifier(
            estimators=ensemble_estimators, voting="soft"
        )
        voting_clf_pre.fit(X_train_pre, y_train_pre)

        calibrated_clf = CalibratedClassifierCV(
            voting_clf_pre,
            method="isotonic",
            cv="prefit",  # النموذج مدرب مسبقاً
        )
        calibrated_clf.fit(X_cal, y_cal)
        final_model = calibrated_clf
        print("  ✅ تمت المعايرة بنجاح")
    else:
        final_model = voting_clf

    # ── التقييم ──
    test_proba = final_model.predict_proba(X_test)
    test_pred = final_model.predict(X_test)
    test_loss = log_loss(y_test_enc, test_proba)

    print(f"\n{'='*50}")
    print(f"🏆 Test LogLoss: {test_loss:.5f}")
    print(f"{'='*50}")

    # تقرير التصنيف
    print("\n📊 تقرير التصنيف:")
    print(
        classification_report(
            y_test_enc,
            test_pred,
            target_names=encoder.classes_,
        )
    )

    # أهمية الميزات
    print_feature_importance(voting_clf, feat_cols)

    # ── حفظ النموذج ──
    import sklearn

    # إضافة خصائص للتوافق
    setattr(final_model, "feature_version_", FEATURE_VERSION)
    setattr(final_model, "sklearn_version_", sklearn.__version__)
    setattr(final_model, "python_version_", sys.version)
    setattr(final_model, "feature_names_expected_", feat_cols)
    setattr(final_model, "label_encoder_", encoder)
    setattr(final_model, "is_calibrated_", calibrate)

    joblib.dump(final_model, model_out)
    print(f"\n✅ تم الحفظ: {model_out}")

    # ── حفظ skops ──
    if SKOPS_AVAILABLE:
        skops_path = model_out.replace(".joblib", ".skops")
        skops_dump(
            final_model,
            skops_path,
            metadata={
                "feature_version": FEATURE_VERSION,
                "league": league,
                "calibrated": str(calibrate),
            },
        )
        print(f"✅ تم الحفظ: {skops_path}")

    # ── حفظ البيانات الوصفية ──
    metadata = {
        "league": league,
        "best_individual_models": all_results,
        "ensemble_components": [n for n, _ in top_models],
        "ensemble_test_logloss": float(test_loss),
        "calibrated": calibrate,
        "classes": list(encoder.classes_),
        "feature_version": FEATURE_VERSION,
        "features_count": len(feat_cols),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = model_out.replace(".joblib", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ تم الحفظ: {meta_path}")


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str, default=None)
    parser.add_argument("--league", type=str, default="PL")
    parser.add_argument(
        "--model-out",
        type=str,
        default="ensemble_model_v3_PL.joblib",
    )
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--inner-threads", type=int, default=1)
    parser.add_argument(
        "--parallel-backend",
        type=str,
        choices=["threading", "loky"],
        default=None,
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="تعطيل معايرة الاحتمالات",
    )
    return parser


def main(argv=None):
    parser = get_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"تجاهل args غير معروفة: {unknown}")

    features_file = args.features_file or f"features_{args.league}.csv"
    if not Path(features_file).exists():
        raise FileNotFoundError(
            f"ملف الميزات غير موجود: {features_file}"
        )

    run_training(
        features_file=features_file,
        league=args.league,
        model_out=args.model_out,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
        inner_threads=args.inner_threads,
        parallel_backend=args.parallel_backend,
        calibrate=not args.no_calibrate,
    )


if __name__ == "__main__":
    main()
