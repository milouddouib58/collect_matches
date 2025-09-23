# predict_match.py
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import joblib
from datetime import datetime, timezone

from features_lib import compute_single_pair_features, FEATURE_VERSION


def maybe_refresh_data(league: str, data_csv: str):
    try:
        from collect_matches import collect_league
        print(f"- Quick refresh of the last 12 months for league {league}...")
        collect_league(league_code=league, out_csv=data_csv, current_only=True)
        print("- Data refreshed.")
    except Exception as e:
        print(f"! Automatic refresh failed: {e}")
        print("> Continuing without refresh…")


# --- تم تبسيط وتصحيح هذه الدالة ---
def load_model(model_path):
    """
    Loads any scikit-learn compatible model object directly from a joblib file.
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from {model_path}")
        raise e


def main():
    parser = argparse.ArgumentParser(description="Predict a match outcome (probabilities H/D/A).")
    parser.add_argument("--league", type=str, default="PL", help="League code (PL, PD, SA, BL1, FL1...)")
    parser.add_argument("--home", type=str, required=True, help="Home team name")
    parser.add_argument("--away", type=str, required=True, help="Away team name")
    parser.add_argument("--model", type=str, required=True, help="Path to the joblib model file")
    parser.add_argument("--data", type=str, default="matches_data.csv", help="CSV file with match data")
    parser.add_argument("--refresh", action="store_true", help="Refresh data for the last 12 months before prediction")
    args = parser.parse_args()

    if args.refresh:
        maybe_refresh_data(args.league, args.data)

    matches = pd.read_csv(args.data)
    pipeline = load_model(args.model)

    # التحقق من إصدار الميزات (إذا كان متوفراً في النموذج)
    model_feature_version = getattr(pipeline, "feature_version_", "Unknown")
    if model_feature_version != "Unknown" and model_feature_version != FEATURE_VERSION:
        print(f"! Warning: Feature version in model ({model_feature_version}) differs from current lib ({FEATURE_VERSION}).")

    X, meta = compute_single_pair_features(
        matches=matches,
        competition=args.league,
        home_team_input=args.home,
        away_team_input=args.away,
        ref_datetime=datetime.now(timezone.utc)
    )
    
    # لا حاجة لإعادة ترتيب الأعمدة لأن النموذج المدمج لا يعتمد عليها مباشرة
    # لكنها ممارسة جيدة إذا كان النموذج الداخلي يتطلب ذلك
    if hasattr(pipeline, "feature_names_in_"):
        X = X.reindex(columns=pipeline.feature_names_in_, fill_value=0)

    proba = pipeline.predict_proba(X)[0]
    classes_model = list(pipeline.classes_)

    prob_map = {cls: float(p) for cls, p in zip(classes_model, proba)}
    p_home = prob_map.get("H", 0.0)
    p_draw = prob_map.get("D", 0.0)
    p_away = prob_map.get("A", 0.0)

    print("========================================")
    print(f"League: {args.league}")
    print(f"Match: {meta['home_team_resolved']} vs {meta['away_team_resolved']}")
    print("----------------------------------------")
    print(f"Home win probability (H): {p_home:.1%}")
    print(f"Draw probability (D):     {p_draw:.1%}")
    print(f"Away win probability (A): {p_away:.1%}")
    print("----------------------------------------")
    top = max([("Home win", p_home), ("Draw", p_draw), ("Away win", p_away)], key=lambda x: x[1])
    print(f"Prediction: {top[0]} ({top[1]:.1%})")
    print("========================================")


if __name__ == "__main__":
    main()
