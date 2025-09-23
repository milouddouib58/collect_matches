#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
جمع نتائج المباريات المنتهية من دوريات football-data.org وحفظها في CSV.
"""

import os
import time
import csv
import argparse
import calendar
from typing import Dict, List, Tuple, Set
from datetime import date, datetime, timedelta
import requests
import pandas as pd

API_BASE = "https://api.football-data.org/v4"
DEFAULT_LEAGUE = "PL"
REQUEST_DELAY_SECONDS = 6  # احترام حد 10 طلبات/الدقيقة

def get_api_key() -> str:
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("FOOTBALL_DATA_API_KEY غير موجود. رجاءً اضبطه كمتغير بيئي قبل التشغيل.")
    return api_key

def month_ranges(start_d: date, end_d: date):
    """تقسيم المدى الزمني إلى أشهر كاملة."""
    current = date(start_d.year, start_d.month, 1)
    while current <= end_d:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        if month_end > end_d:
            month_end = end_d
        start_slice = max(current, start_d)
        yield (start_slice, month_end)
        # إلى أول يوم من الشهر التالي
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

def season_date_range(start_season_year: int) -> Tuple[date, date]:
    start_d = date(start_season_year, 7, 1)
    end_d = date(start_season_year + 1, 6, 30)
    return start_d, end_d

def safe_sleep(seconds: float):
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        raise

def fetch_matches_chunk(league_code: str, start_d: date, end_d: date, api_key: str) -> List[Dict]:
    url = f"{API_BASE}/competitions/{league_code}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {
        "dateFrom": start_d.isoformat(),
        "dateTo": end_d.isoformat(),
        "status": "FINISHED"
    }
    for attempt in range(5):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("matches", [])
        elif resp.status_code == 429:
            wait_s = REQUEST_DELAY_SECONDS * (attempt + 2)
            print(f"تجاوز حد الطلبات (429). الانتظار {wait_s} ثانية...")
            safe_sleep(wait_s)
        elif 500 <= resp.status_code < 600:
            wait_s = 2 ** attempt
            print(f"خطأ خادم {resp.status_code}. إعادة المحاولة بعد {wait_s} ثانية...")
            safe_sleep(wait_s)
        else:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
    return []

def normalize_row(m: Dict) -> Dict:
    score = (m.get("score") or {}).get("fullTime") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    season_info = m.get("season") or {}
    season_start = season_info.get("startDate")

    row = {
        "match_id": m.get("id"),
        "date": m.get("utcDate"),
        "season_start": season_start,
        "matchday": m.get("matchday"),
        "home_team": (
            (m.get("homeTeam") or {}).get("shortName")
            or (m.get("homeTeam") or {}).get("tla")
            or (m.get("homeTeam") or {}).get("name")
        ),
        "away_team": (
            (m.get("awayTeam") or {}).get("shortName")
            or (m.get("awayTeam") or {}).get("tla")
            or (m.get("awayTeam") or {}).get("name")
        ),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "competition": (
            (m.get("competition") or {}).get("code")
            or (m.get("competition") or {}).get("name")
        ),
    }
    return row

def load_existing_ids(csv_path: str) -> Set[int]:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, dtype={"match_id": "Int64"})
        ids = set(int(x) for x in df["match_id"].dropna().astype(int).tolist())
        return ids
    except Exception:
        return set()

def append_rows(csv_path: str, rows: List[Dict]):
    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "match_id", "date", "season_start", "matchday",
        "home_team", "away_team", "home_goals", "away_goals", "competition"
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def collect_league(
    league_code: str,
    start_season: int,
    end_season: int,
    out_csv: str,
    current_only: bool = False
):
    api_key = get_api_key()
    existing_ids = load_existing_ids(out_csv)
    print(f"- الملف الحالي: {out_csv} | عدد المباريات المسجلة سابقًا: {len(existing_ids)}")

    all_new_rows: List[Dict] = []
    total_found = 0

    if current_only:
        end_d = date.today()
        start_d = end_d - timedelta(days=365)
        print(f"- جمع نافذة زمنية: {start_d} -> {end_d}")

        for s, e in month_ranges(start_d, end_d):
            print(f"  طلب: {s} -> {e}")
            matches = fetch_matches_chunk(league_code, s, e, api_key)
            safe_sleep(REQUEST_DELAY_SECONDS)
            total_found += len(matches)
            for m in matches:
                row = normalize_row(m)
                mid = row.get("match_id")
                if pd.isna(mid):
                    continue
                mid = int(mid)
                if mid not in existing_ids:
                    all_new_rows.append(row)
                    existing_ids.add(mid)
    else:
        print(f"- جمع المواسم من {start_season} إلى {end_season} (شاملًا)")
        for y in range(start_season, end_season + 1):
            s_d, e_d = season_date_range(y)
            print(f"  الموسم {y}/{y+1}: {s_d} -> {e_d}")
            for s, e in month_ranges(s_d, e_d):
                print(f"    طلب: {s} -> {e}")
                matches = fetch_matches_chunk(league_code, s, e, api_key)
                safe_sleep(REQUEST_DELAY_SECONDS)
                total_found += len(matches)
                for m in matches:
                    row = normalize_row(m)
                    mid = row.get("match_id")
                    if pd.isna(mid):
                        continue
                    mid = int(mid)
                    if mid not in existing_ids:
                        all_new_rows.append(row)
                        existing_ids.add(mid)

    if all_new_rows:
        append_rows(out_csv, all_new_rows)
    print(f"- تم إيجاد {total_found} مباراة ضمن النطاق المطلوب.")
    print(f"- تمت إضافة {len(all_new_rows)} مباراة جديدة إلى {out_csv}.")
    print("- جاهز!")

def parse_args():
    parser = argparse.ArgumentParser(description="مجمع نتائج مباريات الدوري وحفظها في CSV.")
    parser.add_argument("--league", type=str, default=DEFAULT_LEAGUE, help="رمز الدوري (PL/PD/SA/BL1/FL1)")
    parser.add_argument("--start-season", type=int, default=2023, help="بداية المواسم (سنة بدء الموسم). مثال: 2023")
    parser.add_argument("--end-season", type=int, default=date.today().year, help="نهاية المواسم (سنة بدء الموسم).")
    parser.add_argument("--output", type=str, default="matches_data.csv", help="اسم ملف الإخراج CSV")
    parser.add_argument("--current-only", action="store_true", help="جمع آخر ~12 شهرًا فقط (لتحديث سريع).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collect_league(
        league_code=args.league,
        start_season=args.start_season,
        end_season=args.end_season,
        out_csv=args.output,
        current_only=args.current_only
    )
