# predict_match.py
# -*- coding: utf-8 -*-
import argparse
from datetime import datetime, timezone
import pandas as pd
import joblib
import importlib

from features_lib import compute_single_pair_features, FEATURE_VERSION

try:
    from skops.io import load as skops_load
    SKOPS_AVAILABLE = True
except Exception:
    SKOPS_AVAILABLE = False

def ensure_sklearn_unpickle_shims():
    try:
        ct_mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(ct_mod, "_RemainderColsList"):
            class _RemainderColsList(list): pass
            ct_mod._RemainderColsList = _RemainderColsList
    except Exception:
        pass

def load_model_any(path: str):
    if SKOPS_AVAILABLE and path.endswith(".skops"):
        return skops_load(path, trusted=True)
    ensure_sklearn_unpickle_shims()
    return joblib.load(path)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict match outcome H/D/A probabilities.")
    parser.add_argument("--league", type=str, default="PL")
    parser.add_argument("--home", type=str, required=True)
    parser.add_argument("--away", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)  # .skops مفضل
    parser.add_argument("--data", type=str, default="matches_data.csv")
    parser.add_argument("--when", type=str, default=None)  # ISO datetime UTC
    args, _ = parser.parse_known_args(argv)

    ref_dt = datetime.now(timezone.utc) if not args.when else datetime.fromisoformat(args.when).astimezone(timezone.utc)
    matches = pd.read_csv(args.data)
    pipeline = load_model_any(args.model)

    model_feature_version = getattr(pipeline, "feature_version_", "Unknown")
    if model_feature_version != "Unknown" and model_feature_version != FEATURE_VERSION:
        print(f"! Warning: Feature version differs: model={model_feature_version}, code={FEATURE_VERSION}")

    X, meta = compute_single_pair_features(
        matches=matches, competition=args.league,
        home_team_input=args.home, away_team_input=args.away,
        ref_datetime=ref_dt
    )

    expected_cols = getattr(pipeline, "feature_names_expected_", None)
    if expected_cols is not None:
        X = X.reindex(columns=list(expected_cols), fill_value=0)
    elif hasattr(pipeline, "feature_names_in_"):
        X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)

    proba = pipeline.predict_proba(X)[0]
    classes_model = list(pipeline.classes_)
    prob_map = {cls: float(p) for cls, p in zip(classes_model, proba)}

    print("========================================")
    print(f"{meta['home_team_resolved']} vs {meta['away_team_resolved']} | League: {args.league}")
    print("----------------------------------------")
    print(f"H: {prob_map.get('H',0.0):.1%} | D: {prob_map.get('D',0.0):.1%} | A: {prob_map.get('A',0.0):.1%}")
    print("========================================")

if __name__ == "__main__":
    main()
