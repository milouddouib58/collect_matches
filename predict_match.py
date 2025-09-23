# predict_match.py
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime, timezone
import pandas as pd
import joblib
import importlib
import os

from features_lib import compute_single_pair_features, FEATURE_VERSION, list_feature_columns

# skops (اختياري) للتحميل الآمن بين الإصدارات
try:
    from skops.io import load as skops_load
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False

def ensure_sklearn_unpickle_shims():
    # Hotfix لتحميل joblib قديم فيه _RemainderColsList
    try:
        ct_mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(ct_mod, "_RemainderColsList"):
            class _RemainderColsList(list): pass
            ct_mod._RemainderColsList = _RemainderColsList
    except Exception:
        pass

def load_model_any(path: str):
    if SKOPS_AVAILABLE and path.lower().endswith(".skops"):
        return skops_load(path, trusted=True)
    ensure_sklearn_unpickle_shims()
    return joblib.load(path)

def map_proba_to_HDA(classes_model, proba):
    """
    يحوّل أي تمثيل للفئات إلى خريطة H/D/A -> probability
    يدعم:
      - نصوص H/D/A مباشرة
      - نصوص مرادفات: Home/Home Win/1, Draw/X, Away/Away Win/2
      - أرقام {0,1,2} كما ينتج LabelEncoder بترتيب أبجدي (A=0, D=1, H=2)
    """
    labels = list(classes_model)
    out = {}

    # حالة نصوص
    if all(isinstance(c, str) for c in labels):
        norm_map = {}
        for cls, p in zip(labels, proba):
            s = cls.strip().lower()
            if s in ("h", "home", "home win", "1"):
                norm_map["H"] = float(p)
            elif s in ("d", "draw", "x"):
                norm_map["D"] = float(p)
            elif s in ("a", "away", "away win", "2"):
                norm_map["A"] = float(p)
            else:
                # إذا كانت قيم غريبة، تجاهلها
                pass
        # إذا كانت بالضبط H/D/A كنصوص كبيرة/صغيرة
        if not norm_map and set([c.upper() for c in labels]) >= {"H", "D", "A"}:
            for cls, p in zip(labels, proba):
                norm_map[cls.upper()] = float(p)
        return {
            "H": norm_map.get("H", 0.0),
            "D": norm_map.get("D", 0.0),
            "A": norm_map.get("A", 0.0),
        }

    # حالة أرقام {0,1,2} (LabelEncoder أبجدي: A=0, D=1, H=2)
    try:
        label_set = set(int(x) for x in labels)
        if label_set == {0, 1, 2}:
            idx_A = labels.index(0)
            idx_D = labels.index(1)
            idx_H = labels.index(2)
            return {
                "H": float(proba[idx_H]),
                "D": float(proba[idx_D]),
                "A": float(proba[idx_A]),
            }
    except Exception:
        pass

    # fallback: خصّص أعلى احتمال للنتيجة الأقرب اسمًا (قلّما نصل هنا)
    # لكن لتجنّب 0.0 للجميع نطبّق softmax للتجزئة (احتمالات proba جاهزة أصلاً، لذا نعيدها كما هي بأي ترتيب)
    # سنعيد 0.0 لأننا لا نعرف الترتيب، لكن نحتفظ بالأمان:
    return {"H": 0.0, "D": 0.0, "A": 0.0}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict match outcome H/D/A probabilities.")
    parser.add_argument("--league", type=str, default="PL")
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)  # .skops مفضل
    parser.add_argument("--data", type=str, default="matches_data.csv")
    parser.add_argument("--when", type=str, default=None)  # ISO datetime UTC
    args, _ = parser.parse_known_args(argv)

    # مرجع الزمن
    ref_dt = datetime.now(timezone.utc) if not args.when else datetime.fromisoformat(args.when).astimezone(timezone.utc)

    # تحميل البيانات والنموذج
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data CSV not found: {args.data}")
    matches = pd.read_csv(args.data)

    pipeline = load_model_any(args.model)

    # تحذير اختلاف نسخة الميزات
    from features_lib import FEATURE_VERSION
    model_feature_version = getattr(pipeline, "feature_version_", "Unknown")
    if model_feature_version != "Unknown" and model_feature_version != FEATURE_VERSION:
        print(f"! Warning: Feature version differs: model={model_feature_version}, code={FEATURE_VERSION}")

    # استخراج ميزات المباراة
    X, meta = compute_single_pair_features(
        matches=matches,
        competition=args.league,
        home_team_input=args.home,
        away_team_input=args.away,
        ref_datetime=ref_dt
    )

    # ترتيب الأعمدة
    expected_cols = getattr(pipeline, "feature_names_expected_", None)
    if expected_cols is not None:
        X = X.reindex(columns=list(expected_cols), fill_value=0)
    elif hasattr(pipeline, "feature_names_in_"):
        X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)
    else:
        # ضمان الترتيب العام إذا لم يحدده النموذج
        X = X.reindex(columns=list_feature_columns(), fill_value=0)

    # تنبؤ واحتمالات
    proba = pipeline.predict_proba(X)[0]
    classes_model = list(getattr(pipeline, "classes_", []))

    prob_map = map_proba_to_HDA(classes_model, proba)
    p_home, p_draw, p_away = prob_map["H"], prob_map["D"], prob_map["A"]

    print("========================================")
    print(f"{meta['home_team_resolved']} vs {meta['away_team_resolved']} | League: {args.league}")
    print("----------------------------------------")
    print(f"H: {p_home:.1%} | D: {p_draw:.1%} | A: {p_away:.1%}")
    print("----------------------------------------")
    top = max([("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)], key=lambda x: x[1])
    print(f"Prediction: {top[0]} ({top[1]:.1%})")
    print("========================================")

if __name__ == "__main__":
    main()
