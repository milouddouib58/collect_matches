#!/usr/bin/env python3
# retrain_all.py
# -*- coding: utf-8 -*-
"""
سكربت أتمتة لإعادة تدريب جميع النماذج بعد تغيير الميزات.

الاستخدام:
    python retrain_all.py
    python retrain_all.py --leagues PL,PD
    python retrain_all.py --skip-collect
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(description: str, command: list, env: dict = None) -> bool:
    """تشغيل خطوة واحدة مع عرض النتيجة."""
    print(f"\n{'─'*60}")
    print(f"▶ {description}")
    print(f"  الأمر: {' '.join(command)}")
    print(f"{'─'*60}")

    start = time.time()

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    result = subprocess.run(
        command,
        env=merged_env,
        capture_output=False,  # عرض المخرجات مباشرة
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✅ نجاح ({elapsed:.1f} ثانية)")
        return True
    else:
        print(f"\n❌ فشل (كود: {result.returncode}) ({elapsed:.1f} ثانية)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="إعادة تدريب جميع النماذج"
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default="PL",
        help="الدوريات مفصولة بفاصلة (الافتراضي: PL)",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="تخطي جمع البيانات (استخدام الموجود)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="تخطي هندسة الميزات (استخدام الموجود)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="بدون معايرة احتمالات",
    )
    args = parser.parse_args()

    leagues = [lg.strip().upper() for lg in args.leagues.split(",")]
    python = sys.executable

    print(f"{'='*60}")
    print(f"   🔄 إعادة تدريب النماذج")
    print(f"   الدوريات: {leagues}")
    print(f"{'='*60}")

    # ── التحقق من نسخة الميزات ──
    print("\n📋 التحقق من نسخة الكود...")
    result = subprocess.run(
        [python, "-c",
         "from features_lib import FEATURE_VERSION; print(FEATURE_VERSION)"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        print(f"   نسخة الميزات: {version}")
    else:
        print(f"   ❌ خطأ في التحقق: {result.stderr}")
        sys.exit(1)

    total_start = time.time()
    results = {}

    for league in leagues:
        print(f"\n{'═'*60}")
        print(f"   🏆 الدوري: {league}")
        print(f"{'═'*60}")

        league_success = True

        # ── الخطوة 1: جمع البيانات ──
        if not args.skip_collect:
            if not os.path.exists("matches_data.csv"):
                ok = run_step(
                    f"جمع بيانات {league}",
                    [python, "collect_matches.py",
                     "--league", league,
                     "--current-only"],
                )
                if not ok:
                    print(f"⚠️ فشل جمع البيانات لـ {league}")
                    results[league] = "❌ فشل الجمع"
                    continue
            else:
                print(f"\n📂 matches_data.csv موجود — تخطي الجمع")

        # ── الخطوة 2: هندسة الميزات ──
        features_file = f"features_{league}.csv"

        if not args.skip_features:
            ok = run_step(
                f"هندسة ميزات {league}",
                [python, "engineer_features.py",
                 "--league", league,
                 "--output", features_file],
            )
            if not ok:
                results[league] = "❌ فشل الميزات"
                continue
        else:
            if not Path(features_file).exists():
                print(f"❌ {features_file} غير موجود ولا يمكن تخطيه")
                results[league] = "❌ ملف ميزات مفقود"
                continue
            print(f"\n📂 {features_file} موجود — تخطي الهندسة")

        # ── الخطوة 3: تدريب النموذج ──
        model_file = f"ensemble_model_v3_{league}.joblib"

        train_cmd = [
            python, "train_model_advanced.py",
            "--features-file", features_file,
            "--league", league,
            "--model-out", model_file,
            "--cv-splits", "3",
        ]
        if args.no_calibrate:
            train_cmd.append("--no-calibrate")

        ok = run_step(
            f"تدريب نموذج {league}",
            train_cmd,
        )
        if not ok:
            results[league] = "❌ فشل التدريب"
            continue

        # ── الخطوة 4: التحقق ──
        verify_code = f"""
import joblib
from features_lib import FEATURE_VERSION, list_feature_columns
model = joblib.load('{model_file}')
mv = getattr(model, 'feature_version_', '?')
n_feats = len(getattr(model, 'feature_names_expected_', []))
code_feats = len(list_feature_columns())
ok = mv == FEATURE_VERSION and n_feats == code_feats
status = 'PASS' if ok else 'FAIL'
print(f'{{status}}|{{mv}}|{{n_feats}}|{{code_feats}}')
"""
        verify_result = subprocess.run(
            [python, "-c", verify_code],
            capture_output=True,
            text=True,
        )
        if verify_result.returncode == 0:
            parts = verify_result.stdout.strip().split("|")
            if parts[0] == "PASS":
                results[league] = f"✅ نجاح (v{parts[1]}, {parts[2]} ميزة)"
            else:
                results[league] = f"⚠️ تحذير: v{parts[1]}, {parts[2]}/{parts[3]} ميزة"
        else:
            results[league] = "⚠️ تم التدريب لكن التحقق فشل"

    # ── الملخص النهائي ──
    total_time = time.time() - total_start

    print(f"\n{'═'*60}")
    print(f"   📊 الملخص النهائي")
    print(f"{'═'*60}")
    print(f"   الوقت الإجمالي: {total_time/60:.1f} دقيقة")
    print()
    for league, status in results.items():
        print(f"   {league}: {status}")

    # كود خروج
    all_ok = all("✅" in s for s in results.values())
    if all_ok:
        print(f"\n🎉 اكتمل كل شيء بنجاح!")
    else:
        print(f"\n⚠️ بعض الدوريات لم تكتمل")
        sys.exit(1)


if __name__ == "__main__":
    main()
