# engineer_features.py
# -*- coding: utf-8 -*-
"""
هندسة الميزات من ملف المباريات الخام إلى ملف ميزات جاهز للتدريب.

الاستخدام:
    python engineer_features.py --league PL
    python engineer_features.py --league PL --input matches_data.csv --output features_PL.csv
    python engineer_features.py --leagues PL,PD,SA          ← عدة دوريات
    python engineer_features.py --league PL --profile       ← مع تقرير مفصل
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from features_lib import (
    engineer_match_features,
    list_feature_columns,
    parse_dates,
    FEATURE_VERSION,
)

# ===================== ثوابت =====================

REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "season_start",
    "matchday",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "competition",
]

# ===================== دوال التحقق =====================


def validate_input_file(path: str) -> pd.DataFrame:
    """
    التحقق من ملف الإدخال وتحميله.

    الفحوصات:
    ├── 1. هل الملف موجود؟
    ├── 2. هل الملف فارغ؟
    ├── 3. هل كل الأعمدة المطلوبة موجودة؟
    ├── 4. هل يوجد بيانات بعد التحميل؟
    └── 5. تحويل الأنواع الأساسية
    """
    # 1. وجود الملف
    if not os.path.exists(path):
        print(f"❌ الملف غير موجود: {path}")
        print(f"   الحل: شغّل أولاً:")
        print(f"   python collect_matches.py --league PL --current-only")
        sys.exit(1)

    # 2. حجم الملف
    file_size = os.path.getsize(path)
    if file_size == 0:
        print(f"❌ الملف فارغ: {path}")
        sys.exit(1)

    print(f"📂 تحميل: {path} ({file_size / 1024:.1f} KB)")

    # 3. تحميل
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ خطأ في قراءة الملف: {e}")
        sys.exit(1)

    # 4. التحقق من الأعمدة
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"❌ أعمدة مفقودة: {missing_cols}")
        print(f"   الأعمدة الموجودة: {list(df.columns)}")
        print(f"   الأعمدة المطلوبة: {REQUIRED_COLUMNS}")
        sys.exit(1)

    # 5. التحقق من وجود صفوف
    if len(df) == 0:
        print(f"❌ الملف لا يحتوي على أي صفوف")
        sys.exit(1)

    print(f"   ✅ تم تحميل {len(df)} مباراة | {df.columns.size} عمود")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    تنظيف البيانات قبل هندسة الميزات.

    العمليات:
    ├── 1. حذف الصفوف المكررة (نفس match_id)
    ├── 2. حذف صفوف بدون أهداف
    ├── 3. تحويل أنواع البيانات
    ├── 4. حذف صفوف بتواريخ غير صالحة
    └── 5. ترتيب زمني
    """
    original_count = len(df)
    issues = []

    # 1. حذف المكرر
    if "match_id" in df.columns:
        dupes = df.duplicated(subset=["match_id"], keep="last").sum()
        if dupes > 0:
            df = df.drop_duplicates(subset=["match_id"], keep="last")
            issues.append(f"حذف {dupes} مباراة مكررة")

    # 2. حذف بدون أهداف
    before = len(df)
    df = df.dropna(subset=["home_goals", "away_goals"])
    dropped = before - len(df)
    if dropped > 0:
        issues.append(f"حذف {dropped} مباراة بدون نتيجة")

    # 3. تحويل الأنواع
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype("Int64")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype("Int64")
    df["matchday"] = pd.to_numeric(df["matchday"], errors="coerce").astype("Int64")

    # 4. تواريخ
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    dropped = before - len(df)
    if dropped > 0:
        issues.append(f"حذف {dropped} مباراة بتاريخ غير صالح")

    # 5. ترتيب
    df = df.sort_values("date").reset_index(drop=True)

    # تقرير التنظيف
    final_count = len(df)
    if issues:
        print(f"   🧹 التنظيف:")
        for issue in issues:
            print(f"      - {issue}")
    print(f"   📊 النتيجة: {original_count} → {final_count} مباراة")

    return df


def filter_league(
    df: pd.DataFrame, league: str
) -> pd.DataFrame:
    """
    تصفية المباريات حسب الدوري والتحقق.
    """
    available = sorted(df["competition"].dropna().unique())

    league_df = df[df["competition"] == league]

    if len(league_df) == 0:
        print(f"\n❌ لا توجد مباريات للدوري '{league}'")
        print(f"   الدوريات المتاحة: {available}")
        sys.exit(1)

    # إحصائيات سريعة
    teams = sorted(
        set(league_df["home_team"].unique())
        | set(league_df["away_team"].unique())
    )
    seasons = sorted(league_df["season_start"].dropna().unique())

    print(f"\n📋 الدوري: {league}")
    print(f"   المباريات: {len(league_df)}")
    print(f"   الفرق: {len(teams)}")
    print(f"   المواسم: {len(seasons)}")
    print(f"   الفترة: {league_df['date'].min().strftime('%Y-%m-%d')}"
          f" → {league_df['date'].max().strftime('%Y-%m-%d')}")

    # تحذير إذا كانت البيانات قليلة
    if len(league_df) < 50:
        print(f"   ⚠️ تحذير: عدد المباريات قليل جداً ({len(league_df)})")
        print(f"   النتائج قد تكون غير موثوقة. يُنصح بـ 200+ مباراة")

    return league_df


# ===================== تقرير الجودة =====================


def generate_quality_report(
    feats: pd.DataFrame,
    feat_cols: List[str],
    league: str,
    output_dir: str = ".",
) -> dict:
    """
    إنشاء تقرير جودة شامل للميزات المحسوبة.

    التقرير يشمل:
    ┌──────────────────────────────────────┐
    │ 1. نسبة القيم المفقودة لكل ميزة     │
    │ 2. توزيع المتغير الهدف               │
    │ 3. ميزات ثابتة القيمة (صفر تباين)    │
    │ 4. ارتباطات عالية بين الميزات         │
    │ 5. إحصائيات وصفية                    │
    └──────────────────────────────────────┘
    """
    report = {}
    available_feat_cols = [c for c in feat_cols if c in feats.columns]

    print(f"\n{'='*55}")
    print(f"   📊 تقرير جودة الميزات — {league}")
    print(f"{'='*55}")

    # ── 1. نسبة NaN ──
    print(f"\n📉 نسبة القيم المفقودة (NaN):")
    nan_pct = feats[available_feat_cols].isna().mean().sort_values(ascending=False)
    high_nan = nan_pct[nan_pct > 0.3]  # أكثر من 30%
    mid_nan = nan_pct[(nan_pct > 0.05) & (nan_pct <= 0.3)]
    low_nan = nan_pct[(nan_pct > 0) & (nan_pct <= 0.05)]

    total_nan_pct = feats[available_feat_cols].isna().mean().mean()

    if len(high_nan) > 0:
        print(f"   🔴 ميزات بنسبة عالية (>30%):")
        for col, pct in high_nan.head(10).items():
            bar = "█" * int(pct * 30)
            print(f"      {col:.<40s} {pct:.1%} {bar}")
    if len(mid_nan) > 0:
        print(f"   🟡 ميزات بنسبة متوسطة (5-30%): {len(mid_nan)} ميزة")
    if len(low_nan) > 0:
        print(f"   🟢 ميزات بنسبة منخفضة (<5%): {len(low_nan)} ميزة")

    clean_count = (nan_pct == 0).sum()
    print(f"   ✅ ميزات نظيفة (0% NaN): {clean_count}/{len(available_feat_cols)}")
    print(f"   📊 متوسط NaN الإجمالي: {total_nan_pct:.2%}")

    report["nan_summary"] = {
        "total_mean_pct": float(total_nan_pct),
        "high_nan_count": len(high_nan),
        "clean_count": int(clean_count),
    }

    # ── 2. توزيع الهدف ──
    if "target" in feats.columns:
        print(f"\n🎯 توزيع المتغير الهدف:")
        target_dist = feats["target"].value_counts()
        target_pct = feats["target"].value_counts(normalize=True)
        total = len(feats)

        labels = {"H": "فوز مضيف", "D": "تعادل", "A": "فوز ضيف"}
        for label in ["H", "D", "A"]:
            count = target_dist.get(label, 0)
            pct = target_pct.get(label, 0)
            bar = "█" * int(pct * 40)
            name = labels.get(label, label)
            print(f"   {label} ({name}): {count:>5d} ({pct:.1%}) {bar}")

        # تحذير عدم التوازن
        max_pct = target_pct.max()
        min_pct = target_pct.min()
        if max_pct / max(min_pct, 0.001) > 3:
            print(f"   ⚠️ تحذير: الفئات غير متوازنة بشكل ملحوظ")
            print(f"      النسبة بين الأعلى والأدنى: {max_pct/max(min_pct,0.001):.1f}x")

        report["target_distribution"] = {
            k: {"count": int(v), "pct": float(target_pct.get(k, 0))}
            for k, v in target_dist.items()
        }

    # ── 3. ميزات ثابتة القيمة ──
    print(f"\n📐 تحليل التباين:")
    numeric_feats = feats[available_feat_cols].select_dtypes(include=[np.number])
    zero_var = numeric_feats.columns[numeric_feats.std() == 0].tolist()
    low_var = numeric_feats.columns[
        (numeric_feats.std() > 0) & (numeric_feats.std() < 0.01)
    ].tolist()

    if zero_var:
        print(f"   🔴 ميزات ثابتة (تباين = 0): {len(zero_var)}")
        for col in zero_var[:5]:
            print(f"      - {col}")
        if len(zero_var) > 5:
            print(f"      ... و {len(zero_var)-5} أخرى")
    else:
        print(f"   ✅ لا توجد ميزات ثابتة القيمة")

    if low_var:
        print(f"   🟡 ميزات ذات تباين منخفض جداً: {len(low_var)}")

    report["zero_variance_features"] = zero_var

    # ── 4. ارتباطات عالية ──
    print(f"\n🔗 ارتباطات عالية (>0.95):")
    try:
        corr_matrix = numeric_feats.dropna(axis=1, how="all").corr().abs()
        # أخذ المثلث العلوي فقط
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_pairs = []
        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val) and val > 0.95:
                    high_corr_pairs.append((idx, col, val))

        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        if high_corr_pairs:
            print(f"   وُجد {len(high_corr_pairs)} زوج بارتباط >0.95:")
            for f1, f2, corr in high_corr_pairs[:8]:
                print(f"      {f1} ↔ {f2}: {corr:.3f}")
            if len(high_corr_pairs) > 8:
                print(f"      ... و {len(high_corr_pairs)-8} زوج آخر")
        else:
            print(f"   ✅ لا توجد ارتباطات مفرطة")
        report["high_corr_pairs_count"] = len(high_corr_pairs)
    except Exception as e:
        print(f"   ⚠️ تعذر حساب الارتباطات: {e}")
        report["high_corr_pairs_count"] = -1

    # ── 5. إحصائيات وصفية ──
    print(f"\n📊 ملخص سريع:")
    print(f"   عدد الصفوف: {len(feats)}")
    print(f"   عدد الميزات المطلوبة: {len(feat_cols)}")
    print(f"   عدد الميزات المتوفرة: {len(available_feat_cols)}")
    missing_feats = set(feat_cols) - set(feats.columns)
    if missing_feats:
        print(f"   ⚠️ ميزات مفقودة من الملف: {sorted(missing_feats)[:10]}")

    report["row_count"] = len(feats)
    report["feature_count_expected"] = len(feat_cols)
    report["feature_count_actual"] = len(available_feat_cols)
    report["missing_features"] = sorted(list(missing_feats))

    return report


# ===================== المعالجة الرئيسية =====================


def process_single_league(
    df: pd.DataFrame,
    league: str,
    output_path: str,
    show_profile: bool = False,
) -> Optional[str]:
    """
    معالجة دوري واحد: هندسة الميزات + تقرير الجودة.

    الخطوات:
    1. تصفية الدوري
    2. هندسة الميزات
    3. تقرير الجودة
    4. الحفظ
    """
    # 1. تصفية
    league_df = filter_league(df, league)

    # 2. هندسة الميزات
    print(f"\n⚙️ هندسة الميزات...")
    start_time = time.time()

    try:
        feats = engineer_match_features(df, competition=league)
    except Exception as e:
        print(f"❌ خطأ أثناء هندسة الميزات: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - start_time
    print(f"   ✅ اكتملت في {elapsed:.1f} ثانية")

    # 3. التحقق من النتيجة
    feat_cols = list_feature_columns()
    if len(feats) == 0:
        print(f"❌ لم يتم إنتاج أي صف من الميزات")
        return None

    # 4. تقرير الجودة
    report = generate_quality_report(feats, feat_cols, league)

    # 5. تقرير مفصل (اختياري)
    if show_profile:
        print(f"\n📋 الإحصائيات الوصفية:")
        available = [c for c in feat_cols if c in feats.columns]
        desc = feats[available].describe().T
        desc["nan_pct"] = feats[available].isna().mean()
        print(desc[["count", "mean", "std", "min", "max", "nan_pct"]].to_string())

    # 6. الحفظ
    feats.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / 1024
    print(f"\n💾 تم الحفظ: {output_path} ({file_size:.1f} KB)")

    # 7. حفظ تقرير الجودة
    report_path = output_path.replace(".csv", "_quality_report.json")
    report["league"] = league
    report["feature_version"] = FEATURE_VERSION
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["processing_time_seconds"] = round(elapsed, 2)
    report["output_file"] = output_path

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 تقرير الجودة: {report_path}")

    return output_path


# ===================== نقطة الدخول =====================


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="هندسة ميزات من ملف المباريات إلى ملف ميزات للتدريب.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python engineer_features.py --league PL
  python engineer_features.py --league PL --profile
  python engineer_features.py --leagues PL,PD,SA
  python engineer_features.py --input my_data.csv --league BL1 --output feats_BL1.csv
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="matches_data.csv",
        help="مسار ملف المباريات الخام (الافتراضي: matches_data.csv)",
    )
    parser.add_argument(
        "--league",
        type=str,
        default="PL",
        help="رمز الدوري (الافتراضي: PL)",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default=None,
        help="عدة دوريات مفصولة بفاصلة مثل: PL,PD,SA",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="مسار ملف الإخراج (الافتراضي: features_{league}.csv)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="عرض إحصائيات وصفية مفصلة",
    )
    args, _ = parser.parse_known_args(argv)

    # ── بداية ──
    print(f"{'='*55}")
    print(f"   ⚙️ هندسة الميزات — النسخة {FEATURE_VERSION}")
    print(f"{'='*55}")

    # ── تحميل وتحقق ──
    df = validate_input_file(args.input)
    df = clean_data(df)

    # ── عرض الدوريات المتاحة ──
    available_leagues = sorted(df["competition"].dropna().unique())
    league_counts = df["competition"].value_counts()
    print(f"\n📋 الدوريات المتاحة في الملف:")
    for lg in available_leagues:
        print(f"   {lg}: {league_counts.get(lg, 0)} مباراة")

    # ── تحديد الدوريات المطلوبة ──
    if args.leagues:
        leagues_to_process = [
            lg.strip().upper() for lg in args.leagues.split(",")
        ]
    else:
        leagues_to_process = [args.league.upper()]

    # ── معالجة كل دوري ──
    results = []
    for league in leagues_to_process:
        print(f"\n{'─'*55}")
        out_path = (
            args.output
            if args.output and len(leagues_to_process) == 1
            else f"features_{league}.csv"
        )
        result = process_single_league(
            df, league, out_path, show_profile=args.profile
        )
        if result:
            results.append(result)

    # ── ملخص نهائي ──
    print(f"\n{'='*55}")
    print(f"   ✅ الملخص النهائي")
    print(f"{'='*55}")
    print(f"   الدوريات المعالجة: {len(results)}/{len(leagues_to_process)}")
    for r in results:
        print(f"   📁 {r}")
    print(f"   نسخة الميزات: {FEATURE_VERSION}")
    print(f"   عدد الميزات: {len(list_feature_columns())}")

    if len(results) < len(leagues_to_process):
        failed = set(leagues_to_process) - set(
            r.replace("features_", "").replace(".csv", "")
            for r in results
        )
        print(f"   ❌ فشلت: {failed}")
        sys.exit(1)

    print(f"\n🎉 اكتمل بنجاح!")


if __name__ == "__main__":
    main()
