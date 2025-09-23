# engineer_features.py
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from features_lib import engineer_match_features, list_feature_columns, FEATURE_VERSION

def main(argv=None):
    parser = argparse.ArgumentParser(description="هندسة ميزات من ملف المباريات إلى ملف ميزات للتدريب.")
    parser.add_argument("--input", type=str, default="matches_data.csv")
    parser.add_argument("--league", type=str, default="PL")
    parser.add_argument("--output", type=str, default=None)
    args, _ = parser.parse_known_args(argv)

    out_path = args.output or f"features_{args.league}.csv"
    matches = pd.read_csv(args.input)
    feats = engineer_match_features(matches, competition=args.league)
    feats.to_csv(out_path, index=False)

    print(f"- تم إنشاء ملف الميزات: {out_path}")
    print(f"- عدد الأسطر: {len(feats)} | عدد الميزات: {len(list_feature_columns())}")
    print(f"- نسخة الميزات: {FEATURE_VERSION}")

if __name__ == "__main__":
    main()
