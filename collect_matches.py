#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
جمع نتائج المباريات المنتهية من football-data.org وحفظها في CSV.

الاستخدام:
    # جمع الموسم الحالي للدوري الإنجليزي
    python collect_matches.py --league PL --current-only

    # جمع مواسم محددة
    python collect_matches.py --league PL --start-season 2022 --end-season 2024

    # جمع عدة دوريات
    python collect_matches.py --leagues PL,PD,SA,BL1 --current-only

    # تحديد ملف الإخراج
    python collect_matches.py --league PL --output my_data.csv

    # تحديد تواريخ مخصصة
    python collect_matches.py --league PL --date-from 2024-01-01 --date-to 2024-12-31

المتطلبات:
    - مفتاح API من football-data.org (مجاني)
    - ضبطه كمتغير بيئة: FOOTBALL_DATA_API_KEY=your_key_here

الدوريات المدعومة (الخطة المجانية):
    PL  = الدوري الإنجليزي الممتاز
    PD  = الدوري الإسباني (La Liga)
    SA  = الدوري الإيطالي (Serie A)
    BL1 = الدوري الألماني (Bundesliga)
    FL1 = الدوري الفرنسي (Ligue 1)
    CL  = دوري أبطال أوروبا
    ELC = دوري الدرجة الأولى الإنجليزي (Championship)
    PPL = الدوري البرتغالي (Primeira Liga)
    DED = الدوري الهولندي (Eredivisie)
"""

import os
import sys
import time
import csv
import json
import argparse
import calendar
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
import pandas as pd


# =====================================================================
#  ثوابت
# =====================================================================

API_BASE = "https://api.football-data.org/v4"
DEFAULT_LEAGUE = "PL"
DEFAULT_OUTPUT = "matches_data.csv"

# تأخير بين الطلبات (الخطة المجانية: 10 طلبات/دقيقة)
BASE_DELAY_SECONDS = 6.5
MAX_RETRIES = 5

# الدوريات المدعومة مع أسمائها
SUPPORTED_LEAGUES = {
    "PL": "Premier League (England)",
    "PD": "La Liga (Spain)",
    "SA": "Serie A (Italy)",
    "BL1": "Bundesliga (Germany)",
    "FL1": "Ligue 1 (France)",
    "CL": "Champions League",
    "ELC": "Championship (England)",
    "PPL": "Primeira Liga (Portugal)",
    "DED": "Eredivisie (Netherlands)",
}

# الأعمدة المطلوبة في ملف CSV
CSV_FIELDNAMES = [
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

# إعداد التسجيل (logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =====================================================================
#  دوال مساعدة عامة
# =====================================================================


def get_api_key() -> str:
    """
    الحصول على مفتاح API من متغيرات البيئة.

    طرق ضبط المفتاح:
        Linux/Mac:  export FOOTBALL_DATA_API_KEY=your_key
        Windows:    set FOOTBALL_DATA_API_KEY=your_key
        Python:     os.environ['FOOTBALL_DATA_API_KEY'] = 'your_key'
    """
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()

    if not api_key:
        logger.error("=" * 55)
        logger.error("  مفتاح API غير موجود!")
        logger.error("=" * 55)
        logger.error("  الحل:")
        logger.error("  1. سجّل في https://www.football-data.org/")
        logger.error("  2. احصل على مفتاح API المجاني")
        logger.error("  3. اضبطه كمتغير بيئة:")
        logger.error("     Linux/Mac: export FOOTBALL_DATA_API_KEY=your_key")
        logger.error("     Windows:   set FOOTBALL_DATA_API_KEY=your_key")
        logger.error("=" * 55)
        sys.exit(1)

    # التحقق من شكل المفتاح (عادةً 32 حرف)
    if len(api_key) < 10:
        logger.warning(f"مفتاح API يبدو قصيراً جداً ({len(api_key)} حرف)")

    return api_key


def safe_sleep(seconds: float) -> None:
    """
    انتظار آمن مع إمكانية الإلغاء بـ Ctrl+C.
    """
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        logger.info("تم الإلغاء بواسطة المستخدم")
        raise


def month_ranges(start_d: date, end_d: date) -> List[Tuple[date, date]]:
    """
    تقسيم فترة زمنية إلى أشهر.

    مثال:
        month_ranges(date(2024,1,15), date(2024,3,20))
        →  [(2024-01-15, 2024-01-31),
            (2024-02-01, 2024-02-29),
            (2024-03-01, 2024-03-20)]

    السبب: API يحدّ من نطاق التاريخ في الطلب الواحد
    """
    ranges = []
    current = date(start_d.year, start_d.month, 1)

    while current <= end_d:
        # آخر يوم في الشهر
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)

        # لا نتجاوز تاريخ النهاية
        if month_end > end_d:
            month_end = end_d

        # لا نبدأ قبل تاريخ البداية
        start_slice = max(current, start_d)

        ranges.append((start_slice, month_end))

        # الانتقال للشهر التالي
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    return ranges


def format_duration(seconds: float) -> str:
    """تنسيق المدة الزمنية بشكل مقروء."""
    if seconds < 60:
        return f"{seconds:.0f} ثانية"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins} دقيقة و {secs} ثانية"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours} ساعة و {mins} دقيقة"


# =====================================================================
#  التواصل مع API
# =====================================================================


def validate_api_key(api_key: str) -> bool:
    """
    التحقق من صلاحية مفتاح API بطلب بسيط.
    """
    url = f"{API_BASE}/competitions/PL"
    headers = {"X-Auth-Token": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code == 200:
            logger.info("✅ مفتاح API صالح")
            return True
        elif resp.status_code == 403:
            logger.error("❌ مفتاح API غير صالح أو منتهي الصلاحية")
            return False
        elif resp.status_code == 429:
            logger.warning("⏳ تجاوز حد الطلبات — المفتاح صالح لكن انتظر قليلاً")
            return True  # المفتاح صالح لكن الحد متجاوز
        else:
            logger.error(f"❌ استجابة غير متوقعة: HTTP {resp.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error("❌ لا يوجد اتصال بالإنترنت")
        return False
    except requests.exceptions.Timeout:
        logger.error("❌ انتهت مهلة الاتصال")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ خطأ في الاتصال: {e}")
        return False


def fetch_matches_chunk(
    league_code: str,
    start_d: date,
    end_d: date,
    api_key: str,
    delay: float = BASE_DELAY_SECONDS,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    جلب مباريات منتهية من API لفترة زمنية محددة.

    المدخلات:
        league_code: رمز الدوري (PL, PD, SA...)
        start_d: تاريخ البداية
        end_d: تاريخ النهاية
        api_key: مفتاح API
        delay: التأخير بين المحاولات

    المخرجات:
        (قائمة_المباريات, معلومات_الاستجابة)

    معلومات الاستجابة تشمل:
        status_code: كود HTTP
        remaining_requests: عدد الطلبات المتبقية
        retry_count: عدد المحاولات
    """
    url = f"{API_BASE}/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {
        "dateFrom": start_d.isoformat(),
        "dateTo": end_d.isoformat(),
        "status": "FINISHED",
    }

    response_info = {
        "status_code": None,
        "remaining_requests": None,
        "retry_count": 0,
    }

    for attempt in range(MAX_RETRIES):
        response_info["retry_count"] = attempt

        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30,
            )
            response_info["status_code"] = resp.status_code

            # استخراج عدد الطلبات المتبقية من الرأس
            remaining = resp.headers.get("X-Requests-Available-Minute")
            if remaining is not None:
                response_info["remaining_requests"] = int(remaining)

                # تكيّف ذكي: إذا بقي طلبات قليلة، ننتظر أكثر
                if int(remaining) <= 2:
                    logger.warning(
                        f"⚠️ بقي {remaining} طلب فقط — انتظار إضافي"
                    )
                    safe_sleep(delay * 2)

            # نجاح
            if resp.status_code == 200:
                data = resp.json()
                matches = data.get("matches", [])
                return matches, response_info

            # تجاوز حد الطلبات
            elif resp.status_code == 429:
                wait_s = delay * (attempt + 2)
                logger.warning(
                    f"⏳ تجاوز حد الطلبات (429). "
                    f"المحاولة {attempt + 1}/{MAX_RETRIES}. "
                    f"الانتظار {wait_s:.0f} ثانية..."
                )
                safe_sleep(wait_s)

            # خطأ خادم
            elif 500 <= resp.status_code < 600:
                wait_s = min(2 ** attempt * 2, 60)
                logger.warning(
                    f"🔧 خطأ خادم ({resp.status_code}). "
                    f"المحاولة {attempt + 1}/{MAX_RETRIES}. "
                    f"الانتظار {wait_s} ثانية..."
                )
                safe_sleep(wait_s)

            # مفتاح غير صالح
            elif resp.status_code == 403:
                logger.error("❌ مفتاح API غير صالح (403)")
                return [], response_info

            # الدوري غير موجود
            elif resp.status_code == 404:
                logger.error(
                    f"❌ الدوري '{league_code}' غير موجود (404). "
                    f"تحقق من رمز الدوري."
                )
                return [], response_info

            # خطأ آخر
            else:
                logger.error(
                    f"❌ HTTP {resp.status_code}: "
                    f"{resp.text[:200] if resp.text else 'لا يوجد تفاصيل'}"
                )
                return [], response_info

        except requests.exceptions.ConnectionError:
            wait_s = min(2 ** attempt * 3, 60)
            logger.warning(
                f"🔌 خطأ اتصال. المحاولة {attempt + 1}/{MAX_RETRIES}. "
                f"الانتظار {wait_s} ثانية..."
            )
            safe_sleep(wait_s)

        except requests.exceptions.Timeout:
            wait_s = min(2 ** attempt * 2, 60)
            logger.warning(
                f"⏰ انتهت المهلة. المحاولة {attempt + 1}/{MAX_RETRIES}. "
                f"الانتظار {wait_s} ثانية..."
            )
            safe_sleep(wait_s)

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ خطأ غير متوقع: {e}")
            return [], response_info

    # استنفدت كل المحاولات
    logger.error(
        f"❌ فشلت كل المحاولات ({MAX_RETRIES}) "
        f"للفترة {start_d} → {end_d}"
    )
    return [], response_info


# =====================================================================
#  معالجة البيانات
# =====================================================================


def safe_get(d: Optional[Dict], *keys, default=None) -> Any:
    """
    الوصول الآمن لقيمة في dict متداخل.

    مثال:
        safe_get({"a": {"b": 5}}, "a", "b")  →  5
        safe_get({"a": None}, "a", "b")       →  None
        safe_get(None, "a", "b", default=0)   →  0
    """
    current = d
    for key in keys:
        if current is None or not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def normalize_row(m: Dict) -> Dict:
    """
    تحويل بيانات مباراة واحدة من تنسيق API إلى تنسيق CSV.

    تنسيق API:
    {
        "id": 436244,
        "utcDate": "2024-08-17T14:00:00Z",
        "matchday": 1,
        "homeTeam": {"shortName": "Arsenal", "tla": "ARS", "name": "Arsenal FC"},
        "awayTeam": {"shortName": "Chelsea", "tla": "CHE", "name": "Chelsea FC"},
        "score": {"fullTime": {"home": 2, "away": 1}},
        "season": {"startDate": "2024-08-01"},
        "competition": {"code": "PL", "name": "Premier League"}
    }

    تنسيق CSV:
    {
        "match_id": 436244,
        "date": "2024-08-17T14:00:00Z",
        "season_start": "2024-08-01",
        "matchday": 1,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_goals": 2,
        "away_goals": 1,
        "competition": "PL"
    }
    """
    # استخراج النتيجة بأمان
    home_goals = safe_get(m, "score", "fullTime", "home")
    away_goals = safe_get(m, "score", "fullTime", "away")

    # استخراج بداية الموسم
    season_start = safe_get(m, "season", "startDate")

    # استخراج اسم الفريق المضيف (أولوية: shortName → tla → name)
    home_team = (
        safe_get(m, "homeTeam", "shortName")
        or safe_get(m, "homeTeam", "tla")
        or safe_get(m, "homeTeam", "name")
    )

    # استخراج اسم الفريق الضيف
    away_team = (
        safe_get(m, "awayTeam", "shortName")
        or safe_get(m, "awayTeam", "tla")
        or safe_get(m, "awayTeam", "name")
    )

    # استخراج رمز المسابقة
    competition = (
        safe_get(m, "competition", "code")
        or safe_get(m, "competition", "name")
    )

    return {
        "match_id": m.get("id"),
        "date": m.get("utcDate"),
        "season_start": season_start,
        "matchday": m.get("matchday"),
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "competition": competition,
    }


def validate_row(row: Dict) -> Tuple[bool, str]:
    """
    التحقق من صحة صف بيانات قبل حفظه.

    الفحوصات:
    ├── 1. match_id موجود وصالح
    ├── 2. أسماء الفرق موجودة وغير فارغة
    ├── 3. الأهداف أرقام غير سالبة ومنطقية
    ├── 4. التاريخ موجود
    └── 5. رمز المسابقة موجود

    المخرجات:
        (صالح؟, سبب_الرفض)
    """
    # 1. match_id
    match_id = row.get("match_id")
    if match_id is None:
        return False, "match_id مفقود"
    try:
        mid = int(match_id)
        if mid <= 0:
            return False, f"match_id غير صالح: {match_id}"
    except (ValueError, TypeError):
        return False, f"match_id ليس رقماً: {match_id}"

    # 2. أسماء الفرق
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    if not home_team or not str(home_team).strip():
        return False, "اسم الفريق المضيف مفقود"
    if not away_team or not str(away_team).strip():
        return False, "اسم الفريق الضيف مفقود"
    if str(home_team).strip() == str(away_team).strip():
        return False, f"الفريقان متطابقان: {home_team}"

    # 3. الأهداف
    home_goals = row.get("home_goals")
    away_goals = row.get("away_goals")
    if home_goals is None or away_goals is None:
        return False, "الأهداف مفقودة"
    try:
        hg = int(home_goals)
        ag = int(away_goals)
        if hg < 0 or ag < 0:
            return False, f"أهداف سالبة: {hg}-{ag}"
        if hg > 30 or ag > 30:
            return False, f"أهداف غير منطقية: {hg}-{ag}"
    except (ValueError, TypeError):
        return False, f"الأهداف ليست أرقاماً: {home_goals}-{away_goals}"

    # 4. التاريخ
    match_date = row.get("date")
    if not match_date:
        return False, "التاريخ مفقود"
    try:
        parsed_date = pd.to_datetime(match_date)
        # تحقق أن التاريخ منطقي (بين 2000 و المستقبل القريب)
        if parsed_date.year < 2000:
            return False, f"تاريخ قديم جداً: {match_date}"
        if parsed_date.year > datetime.now().year + 1:
            return False, f"تاريخ في المستقبل البعيد: {match_date}"
    except Exception:
        return False, f"تاريخ غير صالح: {match_date}"

    # 5. المسابقة
    competition = row.get("competition")
    if not competition or not str(competition).strip():
        return False, "رمز المسابقة مفقود"

    return True, "صالح"


# =====================================================================
#  إدارة ملف CSV
# =====================================================================


def load_existing_ids(csv_path: str) -> Set[int]:
    """
    تحميل معرّفات المباريات الموجودة مسبقاً في الملف.
    يمنع تكرار نفس المباراة.
    """
    if not os.path.exists(csv_path):
        return set()

    try:
        df = pd.read_csv(
            csv_path,
            dtype={"match_id": "Int64"},
            usecols=["match_id"],
        )
        existing = set(
            int(x) for x in df["match_id"].dropna().astype(int).tolist()
        )
        return existing

    except pd.errors.EmptyDataError:
        logger.warning(f"الملف {csv_path} فارغ")
        return set()
    except KeyError:
        logger.warning(f"عمود match_id غير موجود في {csv_path}")
        return set()
    except Exception as e:
        logger.warning(f"خطأ في قراءة {csv_path}: {e}")
        return set()


def append_rows(csv_path: str, rows: List[Dict]) -> int:
    """
    إضافة صفوف جديدة إلى ملف CSV.

    المخرجات:
        عدد الصفوف المكتوبة فعلاً
    """
    if not rows:
        return 0

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    written = 0

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

            # كتابة الرأس فقط إذا الملف جديد
            if not file_exists:
                writer.writeheader()

            for row in rows:
                # تنظيف القيم
                clean_row = {}
                for field in CSV_FIELDNAMES:
                    value = row.get(field)
                    # تحويل None و NaN إلى فراغ
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        clean_row[field] = ""
                    else:
                        clean_row[field] = value

                writer.writerow(clean_row)
                written += 1

    except PermissionError:
        logger.error(f"❌ لا يمكن الكتابة في {csv_path} — الملف مفتوح في برنامج آخر؟")
        return 0
    except Exception as e:
        logger.error(f"❌ خطأ في الكتابة: {e}")
        return 0

    return written


def verify_csv_integrity(csv_path: str) -> Dict[str, Any]:
    """
    التحقق من سلامة ملف CSV بعد الكتابة.

    الفحوصات:
    ├── عدد الصفوف
    ├── وجود كل الأعمدة
    ├── عدد القيم المفقودة
    ├── مباريات مكررة
    └── نطاق التواريخ
    """
    result = {
        "valid": True,
        "row_count": 0,
        "issues": [],
    }

    if not os.path.exists(csv_path):
        result["valid"] = False
        result["issues"].append("الملف غير موجود")
        return result

    try:
        df = pd.read_csv(csv_path)
        result["row_count"] = len(df)

        # التحقق من الأعمدة
        missing_cols = [c for c in CSV_FIELDNAMES if c not in df.columns]
        if missing_cols:
            result["issues"].append(f"أعمدة مفقودة: {missing_cols}")

        # المكررات
        if "match_id" in df.columns:
            dupes = df.duplicated(subset=["match_id"]).sum()
            if dupes > 0:
                result["issues"].append(f"{dupes} مباراة مكررة")

        # القيم المفقودة في الأعمدة الأساسية
        critical_cols = ["match_id", "home_team", "away_team", "home_goals", "away_goals"]
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    result["issues"].append(f"{null_count} قيمة مفقودة في {col}")

        # نطاق التواريخ
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce").dropna()
            if not dates.empty:
                result["date_range"] = {
                    "from": dates.min().strftime("%Y-%m-%d"),
                    "to": dates.max().strftime("%Y-%m-%d"),
                }

        # الدوريات
        if "competition" in df.columns:
            result["competitions"] = dict(df["competition"].value_counts())

        if result["issues"]:
            result["valid"] = False

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"خطأ في القراءة: {e}")

    return result


# =====================================================================
#  جمع البيانات
# =====================================================================


def collect_league(
    league_code: str,
    start_season: int,
    end_season: int,
    out_csv: str,
    current_only: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """
    الدالة الرئيسية لجمع مباريات دوري واحد.

    المدخلات:
        league_code: رمز الدوري (PL, PD, SA...)
        start_season: سنة بداية أول موسم (مثل 2022)
        end_season: سنة بداية آخر موسم (مثل 2024)
        out_csv: مسار ملف الإخراج
        current_only: جمع آخر سنة فقط
        date_from: تاريخ بداية مخصص (YYYY-MM-DD)
        date_to: تاريخ نهاية مخصص (YYYY-MM-DD)

    المخرجات:
        dict بإحصائيات الجمع
    """
    # إحصائيات الجمع
    stats = {
        "league": league_code,
        "total_found": 0,
        "new_added": 0,
        "skipped_duplicate": 0,
        "skipped_invalid": 0,
        "api_requests": 0,
        "errors": [],
        "start_time": time.time(),
    }

    api_key = get_api_key()
    league_name = SUPPORTED_LEAGUES.get(league_code, league_code)

    logger.info("=" * 55)
    logger.info(f"  🏆 جمع بيانات: {league_name} ({league_code})")
    logger.info("=" * 55)

    # تحميل المعرّفات الموجودة
    existing_ids = load_existing_ids(out_csv)
    logger.info(f"📂 الملف: {out_csv}")
    logger.info(f"   المباريات المسجلة سابقاً: {len(existing_ids)}")

    all_new_rows: List[Dict] = []

    # ── تحديد النطاق الزمني ──
    if date_from and date_to:
        # نطاق مخصص
        try:
            start_d = date.fromisoformat(date_from)
            end_d = date.fromisoformat(date_to)
        except ValueError:
            logger.error(f"❌ تنسيق تاريخ غير صالح: {date_from} أو {date_to}")
            logger.error("   التنسيق المطلوب: YYYY-MM-DD")
            stats["errors"].append("تنسيق تاريخ غير صالح")
            return stats

        logger.info(f"📅 نطاق مخصص: {start_d} → {end_d}")
        date_ranges = [(start_d, end_d)]

    elif current_only:
        # آخر سنة
        end_d = date.today()
        start_d = end_d - timedelta(days=365)
        logger.info(f"📅 الموسم الحالي: {start_d} → {end_d}")
        date_ranges = [(start_d, end_d)]

    else:
        # مواسم محددة
        logger.info(f"📅 المواسم: {start_season}/{start_season+1} → {end_season}/{end_season+1}")
        date_ranges = []
        for y in range(start_season, end_season + 1):
            s_d = date(y, 7, 1)
            e_d = date(y + 1, 6, 30)
            date_ranges.append((s_d, e_d))

    # ── جمع البيانات لكل نطاق ──
    for range_idx, (range_start, range_end) in enumerate(date_ranges):
        if len(date_ranges) > 1:
            logger.info(f"\n📅 النطاق {range_idx + 1}/{len(date_ranges)}: {range_start} → {range_end}")

        # تقسيم إلى أشهر
        month_chunks = month_ranges(range_start, range_end)
        total_chunks = len(month_chunks)

        for chunk_idx, (chunk_start, chunk_end) in enumerate(month_chunks):
            # عرض التقدم
            progress = (chunk_idx + 1) / total_chunks * 100
            logger.info(
                f"  📡 [{chunk_idx + 1}/{total_chunks}] "
                f"{chunk_start} → {chunk_end} "
                f"({progress:.0f}%)"
            )

            # جلب المباريات
            matches, resp_info = fetch_matches_chunk(
                league_code, chunk_start, chunk_end, api_key
            )
            stats["api_requests"] += 1

            # عرض معلومات الطلبات المتبقية
            remaining = resp_info.get("remaining_requests")
            if remaining is not None and remaining <= 3:
                logger.warning(f"  ⚠️ طلبات متبقية: {remaining}")

            # معالجة المباريات
            chunk_new = 0
            chunk_skip_dupe = 0
            chunk_skip_invalid = 0

            for match_data in matches:
                row = normalize_row(match_data)
                match_id = row.get("match_id")

                # التحقق من المعرّف
                if match_id is None or pd.isna(match_id):
                    chunk_skip_invalid += 1
                    continue

                match_id = int(match_id)

                # التحقق من التكرار
                if match_id in existing_ids:
                    chunk_skip_dupe += 1
                    continue

                # التحقق من صحة البيانات
                is_valid, reason = validate_row(row)
                if not is_valid:
                    chunk_skip_invalid += 1
                    logger.debug(
                        f"    ⚠️ مباراة {match_id} مرفوضة: {reason}"
                    )
                    continue

                # إضافة المباراة
                all_new_rows.append(row)
                existing_ids.add(match_id)
                chunk_new += 1

            stats["total_found"] += len(matches)
            stats["skipped_duplicate"] += chunk_skip_dupe
            stats["skipped_invalid"] += chunk_skip_invalid

            # تفاصيل الدفعة
            if len(matches) > 0:
                logger.info(
                    f"    → وُجد {len(matches)} | "
                    f"جديد: {chunk_new} | "
                    f"مكرر: {chunk_skip_dupe} | "
                    f"مرفوض: {chunk_skip_invalid}"
                )

            # تأخير بين الطلبات
            if chunk_idx < total_chunks - 1:
                safe_sleep(BASE_DELAY_SECONDS)

    # ── حفظ النتائج ──
    if all_new_rows:
        logger.info(f"\n💾 حفظ {len(all_new_rows)} مباراة جديدة...")
        written = append_rows(out_csv, all_new_rows)
        stats["new_added"] = written

        if written != len(all_new_rows):
            logger.warning(
                f"⚠️ كُتب {written} من أصل {len(all_new_rows)} مباراة"
            )
    else:
        logger.info("\nℹ️ لا توجد مباريات جديدة لإضافتها")
        stats["new_added"] = 0

    # ── حساب الوقت ──
    stats["elapsed_seconds"] = time.time() - stats["start_time"]

    return stats


def print_collection_summary(stats: Dict[str, Any]) -> None:
    """
    طباعة ملخص عملية الجمع.
    """
    logger.info("")
    logger.info("─" * 55)
    logger.info(f"  📊 ملخص جمع {stats['league']}")
    logger.info("─" * 55)
    logger.info(f"  المباريات الموجودة في API: {stats['total_found']}")
    logger.info(f"  المباريات الجديدة المضافة: {stats['new_added']}")
    logger.info(f"  المتخطاة (مكررة):         {stats['skipped_duplicate']}")
    logger.info(f"  المرفوضة (غير صالحة):     {stats['skipped_invalid']}")
    logger.info(f"  عدد طلبات API:            {stats['api_requests']}")
    logger.info(f"  الوقت المستغرق:           {format_duration(stats['elapsed_seconds'])}")

    if stats["errors"]:
        logger.warning(f"  ⚠️ أخطاء: {stats['errors']}")
    logger.info("─" * 55)


def print_final_report(
    all_stats: List[Dict[str, Any]],
    csv_path: str,
) -> None:
    """
    طباعة التقرير النهائي الشامل بعد جمع كل الدوريات.
    """
    logger.info("")
    logger.info("=" * 55)
    logger.info("  📋 التقرير النهائي")
    logger.info("=" * 55)

    total_new = sum(s["new_added"] for s in all_stats)
    total_found = sum(s["total_found"] for s in all_stats)
    total_requests = sum(s["api_requests"] for s in all_stats)
    total_time = sum(s["elapsed_seconds"] for s in all_stats)

    # ملخص كل دوري
    for stats in all_stats:
        league = stats["league"]
        league_name = SUPPORTED_LEAGUES.get(league, league)
        icon = "✅" if stats["new_added"] > 0 else "ℹ️"
        logger.info(
            f"  {icon} {league_name}: "
            f"+{stats['new_added']} جديدة "
            f"(من {stats['total_found']} في API)"
        )

    logger.info("")
    logger.info(f"  📊 الإجمالي: +{total_new} مباراة جديدة")
    logger.info(f"  📡 طلبات API: {total_requests}")
    logger.info(f"  ⏱️ الوقت: {format_duration(total_time)}")

    # التحقق من سلامة الملف
    logger.info("")
    logger.info(f"  🔍 فحص سلامة الملف: {csv_path}")
    integrity = verify_csv_integrity(csv_path)

    if integrity["valid"]:
        logger.info(f"     ✅ سليم")
    else:
        for issue in integrity["issues"]:
            logger.warning(f"     ⚠️ {issue}")

    logger.info(f"     📊 إجمالي المباريات في الملف: {integrity['row_count']}")

    if "date_range" in integrity:
        logger.info(
            f"     📅 الفترة: {integrity['date_range']['from']} "
            f"→ {integrity['date_range']['to']}"
        )

    if "competitions" in integrity:
        logger.info(f"     🏆 الدوريات:")
        for comp, count in sorted(
            integrity["competitions"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            comp_name = SUPPORTED_LEAGUES.get(comp, comp)
            logger.info(f"        {comp}: {count} مباراة ({comp_name})")

    logger.info("=" * 55)
    logger.info("  🎉 اكتمل!")
    logger.info("=" * 55)


# =====================================================================
#  نقطة الدخول
# =====================================================================


def parse_args() -> argparse.Namespace:
    """
    معالجة وسائط سطر الأوامر.
    """
    parser = argparse.ArgumentParser(
        description="جمع نتائج المباريات المنتهية من football-data.org",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python collect_matches.py --league PL --current-only
  python collect_matches.py --league PL --start-season 2022 --end-season 2024
  python collect_matches.py --leagues PL,PD,SA --current-only
  python collect_matches.py --league PL --date-from 2024-01-01 --date-to 2024-12-31

الدوريات المدعومة:
  PL  = Premier League       PD  = La Liga
  SA  = Serie A              BL1 = Bundesliga
  FL1 = Ligue 1              CL  = Champions League
  ELC = Championship         PPL = Primeira Liga
  DED = Eredivisie
        """,
    )

    parser.add_argument(
        "--league",
        type=str,
        default=DEFAULT_LEAGUE,
        help=f"رمز الدوري (الافتراضي: {DEFAULT_LEAGUE})",
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default=None,
        help="عدة دوريات مفصولة بفاصلة (مثل: PL,PD,SA)",
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2024,
        help="سنة بداية أول موسم (الافتراضي: 2024)",
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="سنة بداية آخر موسم (الافتراضي: 2024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"مسار ملف الإخراج (الافتراضي: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="جمع آخر 365 يوم فقط",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="تاريخ بداية مخصص (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="تاريخ نهاية مخصص (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--validate-key",
        action="store_true",
        help="التحقق من صلاحية مفتاح API فقط",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="فحص سلامة ملف CSV فقط",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="عرض تفاصيل أكثر",
    )

    return parser.parse_args()


def main() -> None:
    """
    الدالة الرئيسية.
    """
    args = parse_args()

    # ضبط مستوى التفصيل
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── وضع التحقق من المفتاح فقط ──
    if args.validate_key:
        api_key = get_api_key()
        valid = validate_api_key(api_key)
        sys.exit(0 if valid else 1)

    # ── وضع فحص الملف فقط ──
    if args.verify:
        logger.info(f"🔍 فحص سلامة: {args.output}")
        result = verify_csv_integrity(args.output)

        if result["valid"]:
            logger.info("✅ الملف سليم")
        else:
            for issue in result["issues"]:
                logger.warning(f"⚠️ {issue}")

        logger.info(f"📊 عدد المباريات: {result['row_count']}")

        if "date_range" in result:
            logger.info(
                f"📅 الفترة: {result['date_range']['from']} "
                f"→ {result['date_range']['to']}"
            )
        if "competitions" in result:
            for comp, count in result["competitions"].items():
                logger.info(f"  {comp}: {count}")

        sys.exit(0 if result["valid"] else 1)

    # ── التحقق من المفتاح قبل البدء ──
    api_key = get_api_key()
    logger.info("🔑 التحقق من مفتاح API...")
    if not validate_api_key(api_key):
        sys.exit(1)
    safe_sleep(BASE_DELAY_SECONDS)

    # ── تحديد الدوريات ──
    if args.leagues:
        leagues = [lg.strip().upper() for lg in args.leagues.split(",")]
    else:
        leagues = [args.league.upper()]

    # تحقق من الرموز
    for league in leagues:
        if league not in SUPPORTED_LEAGUES:
            logger.warning(
                f"⚠️ الدوري '{league}' غير معروف. "
                f"الدوريات المدعومة: {list(SUPPORTED_LEAGUES.keys())}"
            )
            # نستمر على أي حال — ربما يكون رمزاً صالحاً

    logger.info(f"\n📋 الدوريات المطلوبة: {leagues}")
    logger.info(f"📂 ملف الإخراج: {args.output}")

    # ── التحقق من التواريخ المخصصة ──
    if args.date_from and not args.date_to:
        args.date_to = date.today().isoformat()
        logger.info(f"ℹ️ تاريخ النهاية لم يُحدد — استخدام اليوم: {args.date_to}")

    if args.date_to and not args.date_from:
        logger.error("❌ يجب تحديد --date-from مع --date-to")
        sys.exit(1)

    # ── جمع البيانات ──
    all_stats: List[Dict[str, Any]] = []

    for league_idx, league in enumerate(leagues):
        if league_idx > 0:
            logger.info(f"\n⏳ انتظار قبل الدوري التالي...")
            safe_sleep(BASE_DELAY_SECONDS * 2)

        try:
            stats = collect_league(
                league_code=league,
                start_season=args.start_season,
                end_season=args.end_season,
                out_csv=args.output,
                current_only=args.current_only,
                date_from=args.date_from,
                date_to=args.date_to,
            )
            print_collection_summary(stats)
            all_stats.append(stats)

        except KeyboardInterrupt:
            logger.info("\n⛔ تم الإلغاء بواسطة المستخدم")
            # حفظ ما تم جمعه حتى الآن
            if all_stats:
                logger.info("💾 حفظ ما تم جمعه...")
                print_final_report(all_stats, args.output)
            sys.exit(130)

        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع في {league}: {e}")
            import traceback
            traceback.print_exc()
            all_stats.append({
                "league": league,
                "total_found": 0,
                "new_added": 0,
                "skipped_duplicate": 0,
                "skipped_invalid": 0,
                "api_requests": 0,
                "errors": [str(e)],
                "elapsed_seconds": 0,
            })

    # ── التقرير النهائي ──
    print_final_report(all_stats, args.output)

    # ── حفظ سجل العملية ──
    try:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "leagues": leagues,
            "output_file": args.output,
            "stats": [
                {
                    "league": s["league"],
                    "new_added": s["new_added"],
                    "total_found": s["total_found"],
                    "api_requests": s["api_requests"],
                    "elapsed_seconds": round(s["elapsed_seconds"], 1),
                }
                for s in all_stats
            ],
        }

        log_path = "collection_log.json"
        log_entries = []

        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    log_entries = json.load(f)
                if not isinstance(log_entries, list):
                    log_entries = [log_entries]
            except Exception:
                log_entries = []

        log_entries.append(log_entry)

        # الاحتفاظ بآخر 50 عملية فقط
        log_entries = log_entries[-50:]

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)

        logger.debug(f"📄 سجل العملية: {log_path}")

    except Exception as e:
        logger.debug(f"تعذر حفظ السجل: {e}")

    # ── كود الخروج ──
    total_errors = sum(len(s.get("errors", [])) for s in all_stats)
    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
