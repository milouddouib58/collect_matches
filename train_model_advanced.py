# train_model_advanced.py
# -*- coding: utf-8 -*-
"""
تدريب نموذج متقدم مع:
- معايرة احتمالات (CalibratedClassifierCV)
- تقرير أهمية الميزات
- دعم CatBoost
- توافق مع scikit-learn 1.2 → 1.6+
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
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    log_loss,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier

# ── نماذج إضافية (اختيارية) ──
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
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from skops.io import dump as skops_dump
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False

from features_lib import list_feature_columns, FEATURE_VERSION

# ── ثوابت ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── التحقق من إصدار scikit-learn ──
SKLEARN_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
print(f"📋 scikit-learn version: {sklearn.__version__}")


# =====================================================================
#  دوال مساعدة
# =====================================================================


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


def _make_logistic_regression():
    """
    إنشاء LogisticRegression بطريقة متوافقة مع كل الإصدارات.

    scikit-learn < 1.6:  multi_class موجود
    scikit-learn ≥ 1.6:  multi_class محذوف (multinomial تلقائي)
    """
    base_params = {
        "max_iter": 2000,
        "random_state": SEED,
        "solver": "lbfgs",
    }

    # إضافة multi_class فقط إذا الإصدار يدعمه
    if SKLEARN_VERSION < (1, 6):
        base_params["multi_class"] = "multinomial"

    return LogisticRegression(**base_params)


def _make_random_forest(inner_threads: int = 1):
    """إنشاء RandomForestClassifier."""
    return RandomForestClassifier(
        random_state=SEED,
        n_jobs=inner_threads,
    )


def _make_mlp():
    """إنشاء MLPClassifier."""
    return MLPClassifier(
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
    )


def _make_xgboost(inner_threads: int = 1):
    """إنشاء XGBClassifier بطريقة متوافقة."""
    params = {
        "eval_metric": "mlogloss",
        "random_state": SEED,
        "n_jobs": inner_threads,
    }

    # use_label_encoder حُذف في إصدارات حديثة من xgboost
    try:
        clf = XGBClassifier(use_label_encoder=False, **params)
    except TypeError:
        clf = XGBClassifier(**params)

    return clf


def _make_lightgbm(inner_threads: int = 1):
    """إنشاء LGBMClassifier."""
    return LGBMClassifier(
        random_state=SEED,
        n_jobs=inner_threads,
        verbose=-1,
    )


def _make_catboost(inner_threads: int = 1):
    """إنشاء CatBoostClassifier."""
    return CatBoostClassifier(
        random_state=SEED,
        verbose=0,
        thread_count=inner_threads,
    )


def _make_gradient_boosting():
    """إنشاء GradientBoostingClassifier."""
    return GradientBoostingClassifier(
        random_state=SEED,
    )


# =====================================================================
#  بناء النماذج
# =====================================================================


def build_models_and_grids(preproc, inner_threads: int = 1):
    """
    بناء النماذج وشبكات البحث.

    النماذج:
    ├── 1. Logistic Regression (دائماً)
    ├── 2. Random Forest (دائماً)
    ├── 3. MLP Neural Network (دائماً)
    ├── 4. XGBoost (إذا متوفر)
    ├── 5. LightGBM (إذا متوفر)
    ├── 6. CatBoost (إذا متوفر)
    └── 7. Gradient Boosting (بديل إذا لم يتوفر 4 أو 5)
    """
    models = {}

    # ── 1. Logistic Regression ──
    print("  📦 إضافة: LogisticRegression")
    models["logreg"] = (
        Pipeline([
            ("preproc", preproc),
            ("clf", _make_logistic_regression()),
        ]),
        {"clf__C": [0.01, 0.05, 0.1, 0.5, 1.0]},
    )

    # ── 2. Random Forest ──
    print("  📦 إضافة: RandomForest")
    models["rf"] = (
        Pipeline([
            ("preproc", preproc),
            ("clf", _make_random_forest(inner_threads)),
        ]),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [8, 12, 20],
            "clf__min_samples_leaf": [5, 10],
        },
    )

    # ── 3. MLP ──
    print("  📦 إضافة: MLPClassifier")
    models["mlp"] = (
        Pipeline([
            ("preproc", preproc),
            ("clf", _make_mlp()),
        ]),
        {
            "clf__hidden_layer_sizes": [(128, 64), (64, 32)],
            "clf__alpha": [1e-3, 1e-4],
        },
    )

    # ── 4. XGBoost ──
    if XGB_AVAILABLE:
        print("  📦 إضافة: XGBoost ✅")
        models["xgboost"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", _make_xgboost(inner_threads)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
        )
    else:
        print("  📦 XGBoost: ❌ غير متوفر")

    # ── 5. LightGBM ──
    if LGBM_AVAILABLE:
        print("  📦 إضافة: LightGBM ✅")
        models["lightgbm"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", _make_lightgbm(inner_threads)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__learning_rate": [0.03, 0.1],
                "clf__num_leaves": [31, 63],
            },
        )
    else:
        print("  📦 LightGBM: ❌ غير متوفر")

    # ── 6. CatBoost ──
    if CATBOOST_AVAILABLE:
        print("  📦 إضافة: CatBoost ✅")
        models["catboost"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", _make_catboost(inner_threads)),
            ]),
            {
                "clf__iterations": [300, 500],
                "clf__depth": [4, 6],
                "clf__learning_rate": [0.03, 0.1],
            },
        )
    else:
        print("  📦 CatBoost: ❌ غير متوفر")

    # ── 7. Gradient Boosting (بديل) ──
    if not XGB_AVAILABLE and not LGBM_AVAILABLE:
        print("  📦 إضافة: GradientBoosting (بديل)")
        models["gboost"] = (
            Pipeline([
                ("preproc", preproc),
                ("clf", _make_gradient_boosting()),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.1],
            },
        )

    print(f"  📊 إجمالي النماذج: {len(models)}")
    return models


# =====================================================================
#  أهمية الميزات
# =====================================================================


def print_feature_importance(model, feat_cols: list, top_n: int = 15):
    """
    طباعة أهم الميزات من النموذج المجمّع.
    """
    print(f"\n📊 أهم {top_n} ميزة:")
    print("-" * 50)

    importances = None
    source_name = ""

    # محاولة استخراج الأهمية من نماذج VotingClassifier
    estimators_dict = {}
    if hasattr(model, "named_estimators_"):
        estimators_dict = model.named_estimators_
    elif hasattr(model, "estimators_"):
        # في بعض الأحيان يكون list بدل dict
        for i, est in enumerate(model.estimators_):
            estimators_dict[f"model_{i}"] = est

    for name, estimator in estimators_dict.items():
        clf = estimator
        # الوصول للمصنف داخل Pipeline
        if hasattr(clf, "named_steps"):
            clf = clf.named_steps.get("clf", clf)

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            source_name = name
            break
        elif hasattr(clf, "coef_"):
            importances = np.mean(np.abs(clf.coef_), axis=0)
            source_name = name
            break

    if importances is not None and len(importances) == len(feat_cols):
        print(f"  (المصدر: {source_name})")
        imp_df = pd.DataFrame({
            "feature": feat_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        max_imp = imp_df["importance"].max()
        if max_imp == 0:
            max_imp = 1  # تجنب القسمة على صفر

        for _, row in imp_df.head(top_n).iterrows():
            bar_len = int(row["importance"] / max_imp * 25)
            bar = "█" * bar_len
            print(f"  {row['feature']:.<40s} {row['importance']:.4f} {bar}")
    else:
        print("  (لم يتم استخراج أهمية الميزات)")


# =====================================================================
#  التدريب الرئيسي
# =====================================================================


def run_training(
    features_file: str,
    league: str,
    model_out: str,
    cv_splits: int = 3,
    scoring: str = "neg_log_loss",
    n_jobs: int = None,
    inner_threads: int = 1,
    parallel_backend: str = None,
    calibrate: bool = True,
):
    """
    التدريب الرئيسي مع معايرة الاحتمالات.

    الخطوات:
    ┌─────────────────────────────────────┐
    │ 1. تحميل وتقسيم البيانات            │
    │ 2. ترميز المتغير الهدف              │
    │ 3. بناء النماذج                      │
    │ 4. GridSearch لكل نموذج             │
    │ 5. اختيار أفضل 3 نماذج              │
    │ 6. بناء VotingClassifier            │
    │ 7. معايرة الاحتمالات (اختياري)       │
    │ 8. التقييم                           │
    │ 9. حفظ النموذج + البيانات الوصفية    │
    └─────────────────────────────────────┘
    """

    # ── إعدادات التوازي ──
    is_windows = platform.system().lower() == "windows"
    if n_jobs is None:
        n_jobs = 1 if is_windows else -1
    if parallel_backend is None:
        parallel_backend = "threading" if is_windows else "loky"

    # ══════════════════════════════════════
    #  1. تحميل وتقسيم البيانات
    # ══════════════════════════════════════

    print("\n" + "=" * 55)
    print("  🚀 بدء تدريب النموذج")
    print("=" * 55)

    df = pd.read_csv(features_file).dropna(subset=["target"])
    feat_cols = list_feature_columns()

    train_df, test_df = temporal_train_test_split(df)
    X_train = train_df.reindex(columns=feat_cols)
    y_train = train_df["target"]
    X_test = test_df.reindex(columns=feat_cols)
    y_test = test_df["target"]

    print(f"\n📋 حجم التدريب: {len(X_train)} | حجم الاختبار: {len(X_test)}")
    print(f"📋 توزيع الهدف (تدريب): {dict(y_train.value_counts())}")
    print(f"📋 توزيع الهدف (اختبار): {dict(y_test.value_counts())}")

    # ══════════════════════════════════════
    #  2. ترميز المتغير الهدف
    # ══════════════════════════════════════

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    print(f"📋 ترتيب الفئات: {list(encoder.classes_)}")

    # ══════════════════════════════════════
    #  3. بناء النماذج
    # ══════════════════════════════════════

    print(f"\n📦 بناء النماذج...")
    preproc = make_preprocessor()
    models_and_grids = build_models_and_grids(preproc, inner_threads)

    # ══════════════════════════════════════
    #  4. GridSearch
    # ══════════════════════════════════════

    best_estimators = {}
    all_results = {}

    print(f"\n🔍 بدء GridSearch...")
    print(f"  OS: {platform.system()} | Python: {sys.version.split()[0]}")
    print(f"  sklearn: {sklearn.__version__}")
    print(f"  n_jobs: {n_jobs} | backend: {parallel_backend}")
    print(f"  inner_threads: {inner_threads}")

    with joblib.parallel_backend(parallel_backend):
        for name, (pipe, grid) in models_and_grids.items():
            print(f"\n{'─'*45}")
            print(f"🔍 بحث: {name}...")

            try:
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring=scoring,
                    cv=TimeSeriesSplit(n_splits=cv_splits),
                    n_jobs=n_jobs,
                    verbose=0,
                    error_score="raise",
                )
                gs.fit(X_train, y_train_enc)

                best_estimators[name] = gs.best_estimator_
                all_results[name] = {
                    "best_score": float(gs.best_score_),
                    "best_params": {
                        k: (
                            str(v)
                            if not isinstance(v, (int, float, bool))
                            else v
                        )
                        for k, v in gs.best_params_.items()
                    },
                }
                print(f"  ✅ {name}: score={gs.best_score_:.5f}")
                print(f"     params: {gs.best_params_}")

            except Exception as e:
                print(f"  ❌ {name} فشل: {e}")
                all_results[name] = {
                    "best_score": float("-inf"),
                    "error": str(e),
                }
                continue

    # التحقق من وجود نماذج ناجحة
    if not best_estimators:
        print("\n❌ لم ينجح أي نموذج في التدريب!")
        print("   الأسباب المحتملة:")
        print("   - بيانات غير كافية")
        print("   - ميزات تحتوي على قيم غير صالحة")
        raise RuntimeError("لم ينجح أي نموذج في التدريب")

    # ══════════════════════════════════════
    #  5. اختيار أفضل النماذج
    # ══════════════════════════════════════

    print(f"\n{'='*45}")
    print("🏆 ترتيب النماذج:")

    successful_results = {
        k: v for k, v in all_results.items()
        if k in best_estimators
    }

    sorted_models = sorted(
        successful_results.items(),
        key=lambda item: item[1]["best_score"],
        reverse=True,
    )

    for rank, (name, info) in enumerate(sorted_models, 1):
        medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else "  "
        selected = " ← مختار" if rank <= 3 else ""
        print(f"  {medal} {rank}. {name}: {info['best_score']:.5f}{selected}")

    # ══════════════════════════════════════
    #  6. بناء Ensemble
    # ══════════════════════════════════════

    # اختيار أفضل 3 (أو أقل إذا لم يتوفر 3)
    n_ensemble = min(3, len(sorted_models))
    top_models = sorted_models[:n_ensemble]
    ensemble_estimators = [
        (name, best_estimators[name]) for name, _ in top_models
    ]

    print(f"\n🧠 بناء Ensemble من {n_ensemble} نموذج...")
    print(f"  النماذج: {[n for n, _ in ensemble_estimators]}")

    voting_clf = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft",
    )
    voting_clf.fit(X_train, y_train_enc)

    # ══════════════════════════════════════
    #  7. معايرة الاحتمالات
    # ══════════════════════════════════════

    if calibrate and len(X_train) >= 100:
        print(f"\n📐 معايرة الاحتمالات...")

        # تقسيم بيانات المعايرة من نهاية التدريب
        cal_size = min(300, len(X_train) // 5)
        cal_size = max(cal_size, 30)  # حد أدنى

        X_train_pre = X_train.iloc[:-cal_size]
        y_train_pre = y_train_enc[:-cal_size]
        X_cal = X_train.iloc[-cal_size:]
        y_cal = y_train_enc[-cal_size:]

        print(f"  بيانات التدريب: {len(X_train_pre)}")
        print(f"  بيانات المعايرة: {len(X_cal)}")

        try:
            # إعادة تدريب بدون بيانات المعايرة
            voting_clf_pre = VotingClassifier(
                estimators=ensemble_estimators,
                voting="soft",
            )
            voting_clf_pre.fit(X_train_pre, y_train_pre)

            calibrated_clf = CalibratedClassifierCV(
                voting_clf_pre,
                method="isotonic",
                cv="prefit",
            )
            calibrated_clf.fit(X_cal, y_cal)
            final_model = calibrated_clf
            print("  ✅ تمت المعايرة بنجاح")

        except Exception as e:
            print(f"  ⚠️ فشلت المعايرة: {e}")
            print("  ℹ️ سيتم استخدام النموذج بدون معايرة")
            final_model = voting_clf
            calibrate = False

    elif calibrate and len(X_train) < 100:
        print(f"\n⚠️ بيانات غير كافية للمعايرة ({len(X_train)} < 100)")
        print("  ℹ️ سيتم استخدام النموذج بدون معايرة")
        final_model = voting_clf
        calibrate = False
    else:
        final_model = voting_clf

    # ══════════════════════════════════════
    #  8. التقييم
    # ══════════════════════════════════════

    print(f"\n{'='*55}")
    print("  📊 التقييم على بيانات الاختبار")
    print(f"{'='*55}")

    test_proba = final_model.predict_proba(X_test)
    test_pred = final_model.predict(X_test)
    test_loss = log_loss(y_test_enc, test_proba)

    print(f"\n🏆 Test LogLoss: {test_loss:.5f}")

    # مقارنة مع baseline (توزيع متساوي)
    baseline_loss = log_loss(
        y_test_enc,
        np.full_like(test_proba, 1.0 / 3.0),
    )
    improvement = (baseline_loss - test_loss) / baseline_loss * 100
    print(f"📊 Baseline LogLoss (uniform): {baseline_loss:.5f}")
    print(f"📈 تحسّن: {improvement:.1f}%")

    # تقرير التصنيف
    print(f"\n📊 تقرير التصنيف:")
    print(
        classification_report(
            y_test_enc,
            test_pred,
            target_names=list(encoder.classes_),
            zero_division=0,
        )
    )

    # أهمية الميزات (من النموذج غير المعاير)
    print_feature_importance(voting_clf, feat_cols)

    # ══════════════════════════════════════
    #  9. حفظ النموذج
    # ══════════════════════════════════════

    print(f"\n{'='*45}")
    print("  💾 حفظ النموذج")
    print(f"{'='*45}")

    # إضافة خصائص للتوافق
    setattr(final_model, "feature_version_", FEATURE_VERSION)
    setattr(final_model, "sklearn_version_", sklearn.__version__)
    setattr(final_model, "python_version_", sys.version)
    setattr(final_model, "feature_names_expected_", feat_cols)
    setattr(final_model, "label_encoder_", encoder)
    setattr(final_model, "is_calibrated_", calibrate)
    setattr(final_model, "classes_", encoder.classes_)

    # حفظ joblib
    joblib.dump(final_model, model_out)
    file_size = Path(model_out).stat().st_size / 1024
    print(f"  ✅ {model_out} ({file_size:.0f} KB)")

    # حفظ skops
    if SKOPS_AVAILABLE:
        skops_path = model_out.replace(".joblib", ".skops")
        try:
            skops_dump(
                final_model,
                skops_path,
                metadata={
                    "feature_version": FEATURE_VERSION,
                    "league": league,
                    "calibrated": str(calibrate),
                    "sklearn_version": sklearn.__version__,
                },
            )
            print(f"  ✅ {skops_path}")
        except Exception as e:
            print(f"  ⚠️ فشل حفظ skops: {e}")

    # حفظ البيانات الوصفية
    metadata = {
        "league": league,
        "feature_version": FEATURE_VERSION,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibrated": calibrate,
        "classes": list(encoder.classes_),
        "features_count": len(feat_cols),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_logloss": round(float(test_loss), 5),
        "baseline_logloss": round(float(baseline_loss), 5),
        "improvement_pct": round(float(improvement), 1),
        "ensemble_components": [n for n, _ in top_models],
        "individual_models": {
            k: {
                "score": round(v["best_score"], 5),
                "params": v.get("best_params", {}),
            }
            for k, v in all_results.items()
            if "error" not in v
        },
        "failed_models": {
            k: v.get("error", "unknown")
            for k, v in all_results.items()
            if "error" in v
        },
        "gridsearch_settings": {
            "cv_splits": cv_splits,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "parallel_backend": parallel_backend,
        },
    }

    meta_path = model_out.replace(".joblib", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ {meta_path}")

    # ── ملخص نهائي ──
    print(f"\n{'='*55}")
    print(f"  🎉 اكتمل التدريب بنجاح!")
    print(f"{'='*55}")
    print(f"  النموذج: {model_out}")
    print(f"  الدوري: {league}")
    print(f"  نسخة الميزات: {FEATURE_VERSION}")
    print(f"  عدد الميزات: {len(feat_cols)}")
    print(f"  معاير: {'نعم' if calibrate else 'لا'}")
    print(f"  Test LogLoss: {test_loss:.5f}")
    print(f"  التحسّن عن Baseline: {improvement:.1f}%")
    print(f"{'='*55}")


# =====================================================================
#  نقطة الدخول
# =====================================================================


def get_arg_parser():
    """إنشاء معالج وسائط سطر الأوامر."""
    parser = argparse.ArgumentParser(
        description="تدريب نموذج متقدم للتنبؤ بنتائج المباريات",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python train_model_advanced.py --league PL
  python train_model_advanced.py --features-file features_PL.csv --league PL
  python train_model_advanced.py --league PL --no-calibrate
  python train_model_advanced.py --league PL --n-jobs 1 --parallel-backend threading
        """,
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="مسار ملف الميزات (الافتراضي: features_{league}.csv)",
    )
    parser.add_argument(
        "--league",
        type=str,
        default="PL",
        help="رمز الدوري (الافتراضي: PL)",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="مسار حفظ النموذج (الافتراضي: ensemble_model_v3_{league}.joblib)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="عدد طيات التحقق المتقاطع (الافتراضي: 3)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="عدد العمليات المتوازية (الافتراضي: 1 على Windows، -1 غيره)",
    )
    parser.add_argument(
        "--inner-threads",
        type=int,
        default=1,
        help="عدد خيوط النماذج الداخلية (الافتراضي: 1)",
    )
    parser.add_argument(
        "--parallel-backend",
        type=str,
        choices=["threading", "loky"],
        default=None,
        help="محرك التوازي (الافتراضي: threading على Windows)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="تعطيل معايرة الاحتمالات",
    )
    return parser


def main(argv=None):
    """الدالة الرئيسية."""
    parser = get_arg_parser()
    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print(f"⚠️ تجاهل وسائط غير معروفة: {unknown}")

    # تحديد المسارات الافتراضية
    features_file = args.features_file or f"features_{args.league}.csv"
    model_out = args.model_out or f"ensemble_model_v3_{args.league}.joblib"

    # التحقق من وجود الملف
    if not Path(features_file).exists():
        print(f"❌ ملف الميزات غير موجود: {features_file}")
        print(f"   الحل: شغّل أولاً:")
        print(f"   python engineer_features.py --league {args.league}")
        sys.exit(1)

    # بدء التدريب
    try:
        run_training(
            features_file=features_file,
            league=args.league,
            model_out=model_out,
            cv_splits=args.cv_splits,
            scoring="neg_log_loss",
            n_jobs=args.n_jobs,
            inner_threads=args.inner_threads,
            parallel_backend=args.parallel_backend,
            calibrate=not args.no_calibrate,
        )
    except KeyboardInterrupt:
        print("\n⛔ تم الإلغاء بواسطة المستخدم")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ خطأ أثناء التدريب: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
