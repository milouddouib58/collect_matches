# predict_match.py
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import joblib
from datetime import datetime, timezone
from sklearn.pipeline import Pipeline

from features_lib import compute_single_pair_features, FEATURE_VERSION


def maybe_refresh_data(league: str, data_csv: str):
    try:
        # Depends on collect_matches.py from stage 1
        from collect_matches import collect_league
        print(f"- Quick refresh of the last 12 months for league {league}...")
        collect_league(
            league_code=league,
            start_season=2018,  # Not used with current_only
            end_season=2018,    # Not used with current_only
            out_csv=data_csv,
            current_only=True
        )
        print("- Data refreshed.")
    except Exception as e:
        print(f"! Automatic refresh failed: {e}")
        print("> Continuing without refresh…")


def load_model(model_path):
    artifact = joblib.load(model_path)

    # Case 1: if the file is a dict (old format)
    if isinstance(artifact, dict):
        pipeline = artifact["pipeline"]
        feat_cols = artifact["feature_cols"]
        feature_version = artifact.get("feature_version", "v1")

    # Case 2: if it’s a Pipeline directly (like best_grid_model_meta.joblib)
    elif isinstance(artifact, Pipeline):
        pipeline = artifact
        feat_cols = (
            list(pipeline.feature_names_in_)
            if hasattr(pipeline, "feature_names_in_")
            else []
        )
        feature_version = "v1"

    else:
        raise ValueError("❌ Unknown model file format.")

    return pipeline, feat_cols, feature_version


def main():
    parser = argparse.ArgumentParser(description="Predict a match outcome (probabilities H/D/A).")
    parser.add_argument("--league", type=str, default="PL", help="League code (PL, PD, SA, BL1, FL1...)")
    parser.add_argument("--home", type=str, required=True, help="Home team name")
    parser.add_argument("--away", type=str, required=True, help="Away team name")
    parser.add_argument("--model", type=str, default="model_PL_v1.joblib", help="Path to the joblib model file")
    parser.add_argument("--data", type=str, default="matches_data.csv", help="CSV file with match data")
    parser.add_argument("--refresh", action="store_true", help="Refresh data for the last 12 months before prediction")
    args = parser.parse_args()

    if args.refresh:
        maybe_refresh_data(args.league, args.data)

    matches = pd.read_csv(args.data)

    # Load model
    pipeline, feat_cols, feature_version = load_model(args.model)

    if feature_version != FEATURE_VERSION:
        print(f"! Warning: Feature version in model ({feature_version}) differs from current ({FEATURE_VERSION}).")

    # Compute match features
    X, meta = compute_single_pair_features(
        matches=matches,
        competition=args.league,
        home_team_input=args.home,
        away_team_input=args.away,
        ref_datetime=datetime.now(timezone.utc)
    )

    # Reorder columns as the model was trained
    if feat_cols:
        X = X.reindex(columns=feat_cols, fill_value=0)

    proba = pipeline.predict_proba(X)[0]
    classes_model = list(pipeline.classes_)  # Model classes order

    # Map probabilities to H, D, A
    prob_map = {cls: float(p) for cls, p in zip(classes_model, proba)}
    p_home = prob_map.get("H", 0.0)
    p_draw = prob_map.get("D", 0.0)
    p_away = prob_map.get("A", 0.0)

    print("========================================")
    print(f"League: {args.league}")
    print(f"Match: {meta['home_team_resolved']} vs {meta['away_team_resolved']}")
    print(f"Reference time (UTC): {meta['ref_time_utc']}")
    print("----------------------------------------")
    print(f"Home win probability (H): {p_home:.1%}")
    print(f"Draw probability (D):     {p_draw:.1%}")
    print(f"Away win probability (A): {p_away:.1%}")
    print("----------------------------------------")

    top = max(
        [("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)],
        key=lambda x: x[1]
    )

    print(f"Prediction: {top[0]} ({top[1]:.1%})")
    print("========================================")


if __name__ == "__main__":
    main()
