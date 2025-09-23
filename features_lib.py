# features_lib.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from difflib import get_close_matches
from typing import Tuple, List, Dict, Optional

# ===================== أدوات مساعدة عامة =====================
RESULT_LABELS = ["H", "D", "A"]  # Home, Draw, Away
FEATURE_VERSION = "v1"


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def match_result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    elif home_goals < away_goals:
        return "A"
    return "D"


def points_from_score(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    elif gf == ga:
        return 1
    return 0


def infer_season_start_for_date(ts: pd.Timestamp) -> str:
    # موسم أوروبي: 1 يوليو للسنة الجارية حتى 30 يونيو للسنة التالية
    y = ts.year if ts.month >= 7 else ts.year - 1
    return f"{y}-07-01"


def list_feature_columns() -> List[str]:
    # حافظ على نفس الترتيب بين التدريب والتنبؤ
    return [
        "h_pts5", "a_pts5",
        "h_gf5", "a_gf5",
        "h_ga5", "a_ga5",
        "h_win_rate5", "a_win_rate5",
        "h_season_home_avg_gf", "h_season_home_avg_ga",
        "a_season_away_avg_gf", "a_season_away_avg_ga",
        "h_days_since_last", "a_days_since_last",
        "h2h_home_pts3", "h2h_home_avg_gf3", "h2h_home_avg_ga3",
        "diff_pts5", "diff_gf5", "diff_ga5", "diff_win_rate5",
        "diff_season_avg_gf", "diff_season_avg_ga",
        "diff_days_since_last",
    ]


def fuzzy_pick_team(user_input: str, valid_names: List[str]) -> str:
    # يطابق بالتقريب (Case-insensitive)
    if user_input in valid_names:
        return user_input

    lower_map = {n.lower(): n for n in valid_names}
    if user_input.lower() in lower_map:
        return lower_map[user_input.lower()]

    candidates = get_close_matches(user_input, valid_names, n=1, cutoff=0.6)
    if candidates:
        return candidates[0]
    
    # إن لم توجد مطابقة قريبة، نعيد كما هو (قد يفشل لاحقًا)
    return user_input


# ===================== بناء إطار مباريات الفريق =====================
def build_team_matches_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """
    يحول كل مباراة إلى صفّين (من منظور كل فريق):
    - team, opponent, is_home, gf, ga, points, date, season_start, competition, match_id, matchday
    """
    req_cols = ["match_id", "date", "season_start", "matchday", "home_team", "away_team", "home_goals", "away_goals", "competition"]
    for c in req_cols:
        if c not in matches.columns:
            raise ValueError(f"العمود مفقود في البيانات: {c}")

    m = matches.copy()
    m = parse_dates(m)

    home = m[["match_id", "date", "season_start", "matchday", "home_team", "away_team", "home_goals", "away_goals", "competition"]].copy()
    home.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_goals": "gf", "away_goals": "ga"
    }, inplace=True)
    home["is_home"] = True

    away = m[["match_id", "date", "season_start", "matchday", "home_team", "away_team", "home_goals", "away_goals", "competition"]].copy()
    away.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_goals": "gf", "home_goals": "ga"
    }, inplace=True)
    away["is_home"] = False

    tm = pd.concat([home, away], ignore_index=True)
    tm["points"] = (tm["gf"] > tm["ga"]).astype(int) * 3 + (tm["gf"] == tm["ga"]).astype(int) * 1
    tm = tm.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return tm


def add_rolling_team_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """
    يضيف ميزات Rolling للفريق قبل كل مباراة (Shift(1) لمنع تسرب المستقبل).
    """
    tm = team_matches.copy()
    tm = tm.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

    # مجموع/متوسط آخر 5
    g = tm.groupby(["team", "competition"], sort=False)
    for col, fn, out in [
        ("points", "sum", "pts5"),
        ("gf", "mean", "gf5"),
        ("ga", "mean", "ga5"),
    ]:
        # Use transform for operations that return a series of the same shape
        tm[out] = g[col].shift(1).rolling(5, min_periods=1).agg(fn)

    tm["win"] = (tm["gf"] > tm["ga"]).astype(int)
    tm["win_rate5"] = tm.groupby(["team", "competition"], sort=False)["win"].shift(1).rolling(5, min_periods=1).mean()

    # أيام الراحة منذ آخر مباراة
    tm["days_since_last"] = g["date"].apply(lambda s: (s - s.shift(1)).dt.total_seconds() / 86400.0).reset_index(level=[0,1], drop=True)

    # متوسطات الموسم حسب (ملعب/خارج) قبل المباراة
    kg = ["team", "competition", "season_start", "is_home"]
    tm["cum_gf_prev"] = tm.groupby(kg)["gf"].cumsum() - tm["gf"]
    tm["cum_ga_prev"] = tm.groupby(kg)["ga"].cumsum() - tm["ga"]
    tm["count_prev"] = tm.groupby(kg).cumcount()

    tm["season_loc_avg_gf"] = np.where(
        tm["count_prev"] > 0,
        tm["cum_gf_prev"] / tm["count_prev"],
        np.nan
    )
    tm["season_loc_avg_ga"] = np.where(
        tm["count_prev"] > 0,
        tm["cum_ga_prev"] / tm["count_prev"],
        np.nan
    )
    return tm


# ===================== ميزات المواجهات المباشرة =====================
def compute_h2h_for_home(row: pd.Series, matches: pd.DataFrame, k: int = 3) -> Tuple[float, float, float]:
    """
    يحسب خصائص H2H من منظور الفريق المضيف:
    - مجموع نقاط المضيف في آخر k مواجهات مباشرة
    - متوسط أهداف المضيف ضد هذا الخصم
    - متوسط الأهداف المستقبلة للمضيف ضد هذا الخصم
    """
    comp = row["competition"]
    home = row["home_team"]
    away = row["away_team"]
    dt = row["date"]

    prev = matches[
        (matches["competition"] == comp) &
        (matches["date"] < dt) &
        (
            ((matches["home_team"] == home) & (matches["away_team"] == away)) |
            ((matches["home_team"] == away) & (matches["away_team"] == home))
        )
    ].sort_values("date", ascending=False).head(k)

    if prev.empty:
        return (np.nan, np.nan, np.nan)

    pts, gf_sum, ga_sum = 0, 0.0, 0.0
    for _, m in prev.iterrows():
        # نتائج من منظور "home"
        if m["home_team"] == home:
            gf = m["home_goals"]; ga = m["away_goals"]
        else:
            gf = m["away_goals"]; ga = m["home_goals"]
        
        gf_sum += gf; ga_sum += ga
        pts += points_from_score(gf, ga)
    
    n = len(prev)
    return (float(pts), gf_sum / n, ga_sum / n)


# ===================== تجميع الميزات على مستوى المباراة =====================
def engineer_match_features(matches: pd.DataFrame, competition: Optional[str] = None) -> pd.DataFrame:
    """
    يبني جدول الميزات لكل مباراة تاريخية لاستخدامها في التدريب/التحليل.
    """
    df = matches.copy()
    df = parse_dates(df)

    if competition:
        df = df[df["competition"] == competition].copy()

    # تجاهل أي سطور لا تحتوي نتيجة
    df = df.dropna(subset=["home_goals", "away_goals"])

    tm = build_team_matches_frame(df)
    tm = add_rolling_team_features(tm)

    # ميزات المضيف
    home_feats = tm[tm["is_home"]].copy()
    home_feats = home_feats[[
        "match_id", "team", "pts5", "gf5", "ga5", "win_rate5",
        "season_loc_avg_gf", "season_loc_avg_ga", "days_since_last"
    ]].rename(columns={
        "team": "home_team",
        "pts5": "h_pts5", "gf5": "h_gf5", "ga5": "h_ga5", "win_rate5": "h_win_rate5",
        "season_loc_avg_gf": "h_season_home_avg_gf",
        "season_loc_avg_ga": "h_season_home_avg_ga",
        "days_since_last": "h_days_since_last"
    })

    # ميزات الضيف
    away_feats = tm[~tm["is_home"]].copy()
    away_feats = away_feats[[
        "match_id", "team", "pts5", "gf5", "ga5", "win_rate5",
        "season_loc_avg_gf", "season_loc_avg_ga", "days_since_last"
    ]].rename(columns={
        "team": "away_team",
        "pts5": "a_pts5", "gf5": "a_gf5", "ga5": "a_ga5", "win_rate5": "a_win_rate5",
        "season_loc_avg_gf": "a_season_away_avg_gf",
        "season_loc_avg_ga": "a_season_away_avg_ga",
        "days_since_last": "a_days_since_last"
    })

    out = df.merge(home_feats, on=["match_id", "home_team"], how="left") \
            .merge(away_feats, on=["match_id", "away_team"], how="left")

    # ميزات H2H
    h2h_vals = out.apply(lambda r: compute_h2h_for_home(r, df, k=3), axis=1, result_type="expand")
    h2h_vals.columns = ["h2h_home_pts3", "h2h_home_avg_gf3", "h2h_home_avg_ga3"]
    out = pd.concat([out, h2h_vals], axis=1)

    # فروقات
    out["diff_pts5"] = out["h_pts5"] - out["a_pts5"]
    out["diff_gf5"] = out["h_gf5"] - out["a_gf5"]
    out["diff_ga5"] = out["h_ga5"] - out["a_ga5"]
    out["diff_win_rate5"] = out["h_win_rate5"] - out["a_win_rate5"]
    out["diff_season_avg_gf"] = out["h_season_home_avg_gf"] - out["a_season_away_avg_gf"]
    out["diff_season_avg_ga"] = out["h_season_home_avg_ga"] - out["a_season_away_avg_ga"]
    out["diff_days_since_last"] = out["h_days_since_last"] - out["a_days_since_last"]

    # الهدف (H/D/A)
    out["target"] = out.apply(lambda r: match_result_label(int(r["home_goals"]), int(r["away_goals"])), axis=1)

    # الترتيب النهائي
    feature_cols = list_feature_columns()
    cols = [
        "match_id", "date", "season_start", "matchday", "competition",
        "home_team", "away_team",
    ] + feature_cols + ["target"]
    
    out = out[cols].sort_values("date").reset_index(drop=True)
    return out


# ===================== ميزات مباراة واحدة للتنبؤ =====================
def compute_single_pair_features(
    matches: pd.DataFrame,
    competition: str,
    home_team_input: str,
    away_team_input: str,
    ref_datetime: Optional[datetime] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    يبني صف ميزات واحد لمباراة مستقبلية (أو في نفس اليوم)
    اعتمادًا على نتائج سابقة فقط.
    """
    df = matches.copy()
    df = parse_dates(df)
    df = df[df["competition"] == competition].copy()

    if ref_datetime is None:
        ref_datetime = datetime.now(timezone.utc)
    ref_ts = pd.Timestamp(ref_datetime)

    # اختيار اسم الفريق بدقة/تقريب من البيانات
    teams = sorted(set(df["home_team"]).union(set(df["away_team"])))
    home_team = fuzzy_pick_team(home_team_input, teams)
    away_team = fuzzy_pick_team(away_team_input, teams)

    # قيّد البيانات إلى الماضي فقط
    past = df[df["date"] < ref_ts].copy()

    # وظائف مساعدة سريعة
    def last_n_team_matches(team: str, n: int = 5) -> pd.DataFrame:
        h = past[past["home_team"] == team][["date", "home_goals", "away_goals"]].assign(is_home=True, gf=lambda x: x["home_goals"], ga=lambda x: x["away_goals"])
        a = past[past["away_team"] == team][["date", "home_goals", "away_goals"]].assign(is_home=False, gf=lambda x: x["away_goals"], ga=lambda x: x["home_goals"])
        t = pd.concat([h, a], ignore_index=True).sort_values("date")
        return t.tail(n)

    def season_loc_avg(team: str, is_home: bool, season_start: str) -> Tuple[float, float]:
        if is_home:
            d = past[(past["home_team"] == team) & (past["season_start"] == season_start)]
            if d.empty: return (np.nan, np.nan)
            return (d["home_goals"].mean(), d["away_goals"].mean())
        else:
            d = past[(past["away_team"] == team) & (past["season_start"] == season_start)]
            if d.empty: return (np.nan, np.nan)
            return (d["away_goals"].mean(), d["home_goals"].mean())

    def days_since_last(team: str) -> float:
        h = past[past["home_team"] == team]["date"]
        a = past[past["away_team"] == team]["date"]
        allm = pd.concat([h, a], ignore_index=True)
        if allm.empty:
            return np.nan
        return float((ref_ts - allm.max()).total_seconds() / 86400.0)

    # form/attack/defense/winrate
    def summarize_last5(team: str) -> Tuple[float, float, float, float]:
        t5 = last_n_team_matches(team, 5)
        if t5.empty:
            return (np.nan, np.nan, np.nan, np.nan)
        
        pts = t5.apply(lambda r: points_from_score(int(r["gf"]), int(r["ga"])), axis=1).sum()
        gf5 = t5["gf"].mean()
        ga5 = t5["ga"].mean()
        win_rate5 = (t5["gf"] > t5["ga"]).astype(int).mean()
        return (float(pts), float(gf5), float(ga5), float(win_rate5))

    # H2H
    def h2h_last3(home: str, away: str) -> Tuple[float, float, float]:
        hist = past[
            ((past["home_team"] == home) & (past["away_team"] == away)) |
            ((past["home_team"] == away) & (past["away_team"] == home))
        ].sort_values("date", ascending=False).head(3)

        if hist.empty:
            return (np.nan, np.nan, np.nan)

        pts, gf_sum, ga_sum = 0, 0.0, 0.0
        for _, m in hist.iterrows():
            if m["home_team"] == home:
                gf = m["home_goals"]; ga = m["away_goals"]
            else:
                gf = m["away_goals"]; ga = m["home_goals"]
            
            pts += points_from_score(int(gf), int(ga))
            gf_sum += gf; ga_sum += ga
        
        n = len(hist)
        return (float(pts), gf_sum / n, ga_sum / n)

    # حساب الميزات
    h_pts5, h_gf5, h_ga5, h_win_rate5 = summarize_last5(home_team)
    a_pts5, a_gf5, a_ga5, a_win_rate5 = summarize_last5(away_team)

    season_start = infer_season_start_for_date(ref_ts)
    h_season_home_avg_gf, h_season_home_avg_ga = season_loc_avg(home_team, True, season_start)
    a_season_away_avg_gf, a_season_away_avg_ga = season_loc_avg(away_team, False, season_start)

    h_days = days_since_last(home_team)
    a_days = days_since_last(away_team)

    h2h_pts3, h2h_avg_gf3, h2h_avg_ga3 = h2h_last3(home_team, away_team)

    # الفروقات
    diff_pts5 = (h_pts5 - a_pts5) if (pd.notna(h_pts5) and pd.notna(a_pts5)) else np.nan
    diff_gf5 = (h_gf5 - a_gf5) if (pd.notna(h_gf5) and pd.notna(a_gf5)) else np.nan
    diff_ga5 = (h_ga5 - a_ga5) if (pd.notna(h_ga5) and pd.notna(a_ga5)) else np.nan
    diff_wr5 = (h_win_rate5 - a_win_rate5) if (pd.notna(h_win_rate5) and pd.notna(a_win_rate5)) else np.nan
    diff_season_avg_gf = (h_season_home_avg_gf - a_season_away_avg_gf) if (pd.notna(h_season_home_avg_gf) and pd.notna(a_season_away_avg_gf)) else np.nan
    diff_season_avg_ga = (h_season_home_avg_ga - a_season_away_avg_ga) if (pd.notna(h_season_home_avg_ga) and pd.notna(a_season_away_avg_ga)) else np.nan
    diff_days = (h_days - a_days) if (pd.notna(h_days) and pd.notna(a_days)) else np.nan

    features = {
        "h_pts5": h_pts5, "a_pts5": a_pts5,
        "h_gf5": h_gf5, "a_gf5": a_gf5,
        "h_ga5": h_ga5, "a_ga5": a_ga5,
        "h_win_rate5": h_win_rate5, "a_win_rate5": a_win_rate5,
        "h_season_home_avg_gf": h_season_home_avg_gf,
        "h_season_home_avg_ga": h_season_home_avg_ga,
        "a_season_away_avg_gf": a_season_away_avg_gf,
        "a_season_away_avg_ga": a_season_away_avg_ga,
        "h_days_since_last": h_days,
        "a_days_since_last": a_days,
        "h2h_home_pts3": h2h_pts3,
        "h2h_home_avg_gf3": h2h_avg_gf3,
        "h2h_home_avg_ga3": h2h_avg_ga3,
        "diff_pts5": diff_pts5,
        "diff_gf5": diff_gf5,
        "diff_ga5": diff_ga5,
        "diff_win_rate5": diff_wr5,
        "diff_season_avg_gf": diff_season_avg_gf,
        "diff_season_avg_ga": diff_season_avg_ga,
        "diff_days_since_last": diff_days,
    }

    feat_cols = list_feature_columns()
    X = pd.DataFrame([features])[feat_cols]

    meta = {
        "home_team_resolved": home_team,
        "away_team_resolved": away_team,
        "competition": competition,
        "ref_time_utc": ref_ts.isoformat(),
        "feature_version": FEATURE_VERSION,
        "season_start_assumed": season_start
    }

    return X, meta

