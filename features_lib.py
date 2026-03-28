# features_lib.py
# -*- coding: utf-8 -*-
"""
مكتبة هندسة الميزات — النسخة v4
الإصلاحات:
  - إصلاح خطأ Rolling عبر حدود المجموعات
  - إصلاح EWM sum → mean
الإضافات:
  - نظام تصنيف Elo
  - سلاسل الفوز / عدم الخسارة
  - معدل BTTS و Over2.5
  - تذبذب الأهداف (الانحراف المعياري)
  - معدل التسجيل أولاً
"""

import re
from collections import defaultdict
from difflib import get_close_matches
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd

# ===================== ثوابت عامة =====================
RESULT_LABELS = ["H", "D", "A"]
FEATURE_VERSION = "v4"  # ← ترقية النسخة

# ===================== أدوات مساعدة عامة =====================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """تحويل عمود التاريخ وترتيب البيانات زمنياً."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def match_result_label(home_goals: int, away_goals: int) -> str:
    """تحديد نتيجة المباراة: H فوز مضيف، A فوز ضيف، D تعادل."""
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def points_from_score(gf: int, ga: int) -> int:
    """حساب النقاط من النتيجة: 3 فوز، 1 تعادل، 0 خسارة."""
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


# ===================== نظام Elo =====================

class EloSystem:
    """
    نظام تصنيف Elo للفرق.
    
    الفكرة:
    - كل فريق يبدأ بـ 1500 نقطة
    - بعد كل مباراة يكسب الفائز نقاط ويخسر الخاسر
    - كلما كان الفوز مفاجئاً (فريق ضعيف يهزم قوي) كلما تغيرت النقاط أكثر
    - فارق الأهداف يضخّم التغيير
    
        elo_home_before_match
               │
               ▼
    ┌─────────────────────┐
    │  Expected = 1/(1+10^│((elo_away - elo_home - bonus)/400))
    │                     │
    │  Actual = 1 (فوز)   │
    │         = 0.5 (تعادل)│
    │         = 0 (خسارة) │
    │                     │
    │  new_elo = old_elo  │
    │    + K × GD_mult    │
    │    × (Actual-Expected)│
    └─────────────────────┘
    """

    def __init__(self, k: float = 28, home_advantage: float = 65,
                 initial: float = 1500):
        self.k = k
        self.home_advantage = home_advantage
        self.initial = initial
        self.ratings: Dict[str, float] = defaultdict(lambda: self.initial)

    def expected_score(self, elo_a: float, elo_b: float) -> float:
        """احتمال فوز A على B بناءً على تصنيفهما."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def goal_diff_multiplier(self, goal_diff: int) -> float:
        """مضاعف فارق الأهداف — فوز كبير يغيّر التصنيف أكثر."""
        return max(1.0, np.log(abs(goal_diff) + 1))

    def update(self, home: str, away: str,
               home_goals: int, away_goals: int) -> Tuple[float, float]:
        """
        تحديث تصنيف الفريقين بعد مباراة.
        يرجع: (elo_home_before, elo_away_before)
        """
        elo_h = self.ratings[home]
        elo_a = self.ratings[away]

        # إضافة ميزة الأرض للمضيف
        exp_h = self.expected_score(elo_h + self.home_advantage, elo_a)
        exp_a = 1.0 - exp_h

        # النتيجة الفعلية
        if home_goals > away_goals:
            actual_h, actual_a = 1.0, 0.0
        elif home_goals < away_goals:
            actual_h, actual_a = 0.0, 1.0
        else:
            actual_h, actual_a = 0.5, 0.5

        gd_mult = self.goal_diff_multiplier(home_goals - away_goals)

        self.ratings[home] += self.k * gd_mult * (actual_h - exp_h)
        self.ratings[away] += self.k * gd_mult * (actual_a - exp_a)

        return elo_h, elo_a  # القيم قبل التحديث

    def get(self, team: str) -> float:
        return self.ratings[team]


def compute_elo_for_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """
    حساب تصنيف Elo لكل مباراة في البيانات.
    يرجع DataFrame بأعمدة: match_id, home_elo, away_elo, elo_diff
    """
    matches = parse_dates(matches.copy())
    elo_sys = EloSystem()
    records = []

    for _, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg = int(row["home_goals"]) if pd.notna(row["home_goals"]) else 0
        ag = int(row["away_goals"]) if pd.notna(row["away_goals"]) else 0

        elo_h_before, elo_a_before = elo_sys.update(h, a, hg, ag)

        records.append({
            "match_id": row["match_id"],
            "home_elo": elo_h_before,
            "away_elo": elo_a_before,
            "elo_diff": elo_h_before - elo_a_before,
        })

    return pd.DataFrame(records)


# ===================== قائمة الميزات =====================

def list_feature_columns() -> List[str]:
    """
    ترجع قائمة مرتبة بجميع أسماء الميزات.
    يجب أن تتطابق مع ما تنتجه engineer_match_features
    و compute_single_pair_features.
    """
    features: List[str] = []
    time_windows = [3, 5, 10]
    metrics = ["pts", "gf", "ga", "win_rate", "cs_rate"]

    # ── ميزات كل فريق (مضيف h_ وضيف a_) ──
    for side in ["h", "a"]:
        # Rolling و EMA
        for metric in metrics:
            for w in time_windows:
                features.append(f"{side}_{metric}{w}")
                features.append(f"{side}_ema_{metric}{w}")

        # معدلات الموسم حسب الموقع
        if side == "h":
            features.extend([
                "h_season_home_avg_gf",
                "h_season_home_avg_ga",
            ])
        else:
            features.extend([
                "a_season_away_avg_gf",
                "a_season_away_avg_ga",
            ])

        # ميزات أخرى لكل فريق
        features.extend([
            f"{side}_days_since_last",
            f"{side}_season_cum_gd",
            f"{side}_ppg_vs_top5",
            f"{side}_ppg_vs_bot5",
        ])

        # ── ميزات جديدة v4 ──
        features.extend([
            f"{side}_elo",                   # تصنيف Elo
            f"{side}_win_streak",            # سلسلة الانتصارات
            f"{side}_unbeaten_streak",       # سلسلة عدم الخسارة
            f"{side}_loss_streak",           # سلسلة الخسائر
            f"{side}_btts_rate5",            # معدل تسجيل الفريقين (5 مباريات)
            f"{side}_over25_rate5",          # معدل أكثر من 2.5 هدف
            f"{side}_gf_std5",              # تذبذب التسجيل
            f"{side}_scored_first_rate5",    # معدل التسجيل أولاً (تقريبي)
        ])

    # ── ميزات المواجهات المباشرة ──
    features.extend([
        "h2h_home_pts3",
        "h2h_home_avg_gf3",
        "h2h_home_avg_ga3",
    ])

    # ── Elo ──
    features.append("elo_diff")

    # ── فروقات بين الفريقين ──
    for metric in metrics:
        for w in time_windows:
            features.append(f"diff_{metric}{w}")
            features.append(f"diff_ema_{metric}{w}")

    features.extend([
        "diff_season_avg_gf",
        "diff_season_avg_ga",
        "diff_days_since_last",
        "diff_season_cum_gd",
        "diff_ppg_vs_top5",
        "diff_ppg_vs_bot5",
        # فروقات v4
        "diff_elo",
        "diff_win_streak",
        "diff_unbeaten_streak",
    ])

    return sorted(list(set(features)))


# ===================== بناء إطار مباريات الفرق =====================

def build_team_matches_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """
    تحويل كل مباراة إلى صفين:
    صف من منظور المضيف وصف من منظور الضيف.
    
    مباراة: Arsenal 2-1 Chelsea
         ↓
    صف 1: team=Arsenal, opponent=Chelsea, gf=2, ga=1, is_home=True
    صف 2: team=Chelsea, opponent=Arsenal, gf=1, ga=2, is_home=False
    """
    req = [
        "match_id", "date", "season_start", "matchday",
        "home_team", "away_team", "home_goals", "away_goals",
        "competition",
    ]
    for c in req:
        if c not in matches.columns:
            raise ValueError(f"العمود مفقود في البيانات: {c}")

    m = parse_dates(matches.copy())

    # صفوف المضيف
    home = m[req].rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "gf",
        "away_goals": "ga",
    })
    home["is_home"] = True

    # صفوف الضيف
    away = m[req].rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "gf",
        "home_goals": "ga",
    })
    away["is_home"] = False

    tm = pd.concat([home, away], ignore_index=True)
    tm["points"] = tm.apply(
        lambda r: points_from_score(int(r["gf"]), int(r["ga"])), axis=1
    )
    tm = tm.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return tm


# ===================== الميزات المتدحرجة (مُصلَحة) =====================

def add_rolling_team_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    """
    حساب الميزات المتدحرجة لكل فريق.
    
    ⚠️ إصلاح حرج: استخدام transform بدل الطريقة القديمة
    لمنع تسرب البيانات بين المجموعات.
    """
    tm = team_matches.copy()
    tm = tm.sort_values(["team", "date", "match_id"]).reset_index(drop=True)

    # أعمدة مساعدة
    tm["win"] = (tm["gf"] > tm["ga"]).astype(int)
    tm["loss"] = (tm["gf"] < tm["ga"]).astype(int)
    tm["draw"] = (tm["gf"] == tm["ga"]).astype(int)
    tm["clean_sheet"] = (tm["ga"] == 0).astype(int)
    tm["btts"] = ((tm["gf"] > 0) & (tm["ga"] > 0)).astype(int)
    tm["over25"] = ((tm["gf"] + tm["ga"]) > 2.5).astype(int)
    tm["scored_first"] = (tm["gf"] > 0).astype(int)  # تقريبي

    g = tm.groupby(["team", "competition"], sort=False)
    time_windows = [3, 5, 10]

    for w in time_windows:
        # ✅ إصلاح: transform يضمن العمل داخل كل مجموعة فقط
        # النقاط المتدحرجة
        tm[f"pts{w}"] = g["points"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).sum()
        )
        # معدل الأهداف المسجلة
        tm[f"gf{w}"] = g["gf"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        # معدل الأهداف المستقبلة
        tm[f"ga{w}"] = g["ga"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        # معدل الفوز
        tm[f"win_rate{w}"] = g["win"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        # معدل الشباك النظيفة
        tm[f"cs_rate{w}"] = g["clean_sheet"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )

        # ✅ إصلاح: EMA مع mean بدل sum
        tm[f"ema_pts{w}"] = g["points"].transform(
            lambda x: x.shift(1).ewm(span=w, min_periods=1).mean()
        )
        tm[f"ema_gf{w}"] = g["gf"].transform(
            lambda x: x.shift(1).ewm(span=w, min_periods=1).mean()
        )
        tm[f"ema_ga{w}"] = g["ga"].transform(
            lambda x: x.shift(1).ewm(span=w, min_periods=1).mean()
        )
        tm[f"ema_win_rate{w}"] = g["win"].transform(
            lambda x: x.shift(1).ewm(span=w, min_periods=1).mean()
        )
        tm[f"ema_cs_rate{w}"] = g["clean_sheet"].transform(
            lambda x: x.shift(1).ewm(span=w, min_periods=1).mean()
        )

    # ── ميزات جديدة v4 ──

    # BTTS rate (آخر 5 مباريات)
    tm["btts_rate5"] = g["btts"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Over 2.5 rate (آخر 5 مباريات)
    tm["over25_rate5"] = g["over25"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # تذبذب التسجيل (الانحراف المعياري لآخر 5 مباريات)
    tm["gf_std5"] = g["gf"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    ).fillna(0)

    # معدل التسجيل أولاً (تقريبي)
    tm["scored_first_rate5"] = g["scored_first"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # ── سلاسل الفوز/الخسارة ──
    tm["win_streak"] = _compute_streaks(tm, g, "win")
    tm["unbeaten_streak"] = _compute_streaks(tm, g, "unbeaten")
    tm["loss_streak"] = _compute_streaks(tm, g, "loss")

    # ── أيام منذ آخر مباراة ──
    tm["days_since_last"] = g["date"].diff().dt.total_seconds() / 86400.0

    # ── فارق الأهداف التراكمي ──
    kg = ["team", "competition", "season_start"]
    tm["cum_gf_prev"] = tm.groupby(kg)["gf"].cumsum() - tm["gf"]
    tm["cum_ga_prev"] = tm.groupby(kg)["ga"].cumsum() - tm["ga"]
    tm["season_cum_gd"] = tm["cum_gf_prev"] - tm["cum_ga_prev"]

    # ── معدلات الموسم حسب الموقع (بيت/خارج) ──
    loc_kg = kg + ["is_home"]
    loc_counts = tm.groupby(loc_kg).cumcount()
    loc_counts_safe = loc_counts.replace(0, np.nan)  # تجنب القسمة على صفر

    tm["season_loc_avg_gf"] = (
        (tm.groupby(loc_kg)["gf"].cumsum() - tm["gf"]) / loc_counts_safe
    )
    tm["season_loc_avg_ga"] = (
        (tm.groupby(loc_kg)["ga"].cumsum() - tm["ga"]) / loc_counts_safe
    )

    tm.replace([np.inf, -np.inf], np.nan, inplace=True)
    return tm


def _compute_streaks(tm: pd.DataFrame, groups, streak_type: str) -> pd.Series:
    """
    حساب سلاسل الفوز / عدم الخسارة / الخسارة.
    
    streak_type:
      "win"      → سلسلة انتصارات متتالية
      "unbeaten" → سلسلة عدم خسارة (فوز أو تعادل)
      "loss"     → سلسلة خسائر متتالية
    
    مثال:
    النتائج:  W  W  L  W  W  W  D  L
    win:      1  2  0  1  2  3  0  0
    unbeaten: 1  2  0  1  2  3  4  0
    loss:     0  0  1  0  0  0  0  1
    """
    def _streak_for_group(series):
        if streak_type == "win":
            condition = series
        elif streak_type == "unbeaten":
            # لم يخسر = فاز أو تعادل
            losses = tm.loc[series.index, "loss"]
            condition = 1 - losses
        elif streak_type == "loss":
            condition = tm.loc[series.index, "loss"]
        else:
            return pd.Series(0, index=series.index)

        streaks = []
        current_streak = 0
        for val in condition:
            if val == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)

        # shift(1) لأننا نريد السلسلة قبل المباراة الحالية
        result = pd.Series(streaks, index=series.index).shift(1).fillna(0)
        return result

    return groups["win"].transform(_streak_for_group)


# ===================== ترتيب المواسم السابقة =====================

def get_season_final_ranks(
    matches: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    حساب الترتيب النهائي لكل فريق في كل موسم.
    يُستخدم لتحديد قوة المنافس (top5/mid/bot5).
    """
    ranks_by_season: Dict[str, Dict[str, int]] = {}
    matches = parse_dates(matches)

    for season_start, season_df in matches.groupby("season_start"):
        table = defaultdict(lambda: {"pts": 0, "gd": 0, "gf": 0})

        for row in season_df.itertuples():
            h, a = row.home_team, row.away_team
            hg, ag = int(row.home_goals), int(row.away_goals)

            table[h]["pts"] += points_from_score(hg, ag)
            table[h]["gd"] += hg - ag
            table[h]["gf"] += hg

            table[a]["pts"] += points_from_score(ag, hg)
            table[a]["gd"] += ag - hg
            table[a]["gf"] += ag

        sorted_table = sorted(
            table.items(),
            key=lambda item: (item[1]["pts"], item[1]["gd"], item[1]["gf"]),
            reverse=True,
        )
        ranks = {team: i + 1 for i, (team, _) in enumerate(sorted_table)}
        ranks_by_season[season_start] = ranks

    return ranks_by_season


def add_performance_vs_ranks_features(
    team_matches: pd.DataFrame,
    season_ranks: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    """
    إضافة أداء الفريق ضد الفرق القوية والضعيفة.
    ppg_vs_top5 = معدل النقاط ضد أفضل 5 فرق الموسم السابق
    ppg_vs_bot5 = معدل النقاط ضد أضعف 5 فرق الموسم السابق
    """
    tm = team_matches.copy()

    # حساب بداية الموسم السابق
    tm["prev_season_start"] = (
        pd.to_datetime(tm["season_start"]).dt.year - 1
    ).astype(str) + "-07-01"

    # ترتيب المنافس في الموسم السابق (20 = غير معروف)
    tm["opponent_rank_prev_season"] = tm.apply(
        lambda r: season_ranks.get(r["prev_season_start"], {}).get(
            r["opponent"], 20
        ),
        axis=1,
    )

    # تصنيف قوة المنافس
    tm["opponent_strength"] = pd.cut(
        tm["opponent_rank_prev_season"],
        bins=[0, 5, 15, 21],
        labels=["top5", "mid", "bot5"],
    )

    # نقاط ضد كل فئة
    tm["points_vs_top5"] = tm.where(
        tm["opponent_strength"] == "top5"
    )["points"]
    tm["points_vs_bot5"] = tm.where(
        tm["opponent_strength"] == "bot5"
    )["points"]

    g = tm.groupby(["team", "season_start"])

    prev_pts_top5 = g["points_vs_top5"].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    prev_games_top5 = g["points_vs_top5"].transform(
        lambda x: x.shift(1).notna().cumsum()
    )
    prev_pts_bot5 = g["points_vs_bot5"].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    prev_games_bot5 = g["points_vs_bot5"].transform(
        lambda x: x.shift(1).notna().cumsum()
    )

    # تجنب القسمة على صفر
    tm["ppg_vs_top5"] = prev_pts_top5 / prev_games_top5.replace(0, np.nan)
    tm["ppg_vs_bot5"] = prev_pts_bot5 / prev_games_bot5.replace(0, np.nan)
    tm.fillna({"ppg_vs_top5": 0, "ppg_vs_bot5": 0}, inplace=True)

    return tm


# ===================== المواجهات المباشرة H2H =====================

def compute_h2h_for_home(
    row: pd.Series,
    matches: pd.DataFrame,
    k: int = 3,
) -> Tuple[float, float, float]:
    """
    حساب إحصائيات المواجهات المباشرة لآخر k مباريات.
    يرجع: (نقاط المضيف، معدل أهداف المضيف، معدل أهداف عليه)
    """
    comp = row.competition
    home = row.home_team
    away = row.away_team
    dt = row.date

    prev = matches[
        (matches.competition == comp)
        & (matches.date < dt)
        & (
            ((matches.home_team == home) & (matches.away_team == away))
            | ((matches.home_team == away) & (matches.away_team == home))
        )
    ].sort_values("date", ascending=False).head(k)

    if prev.empty:
        return (np.nan, np.nan, np.nan)

    pts, gf_sum, ga_sum = 0, 0.0, 0.0
    for _, m in prev.iterrows():
        if m.home_team == home:
            gf, ga = m.home_goals, m.away_goals
        else:
            gf, ga = m.away_goals, m.home_goals
        gf_sum += gf
        ga_sum += ga
        pts += points_from_score(int(gf), int(ga))

    n = len(prev)
    return (float(pts), gf_sum / n, ga_sum / n)


# ===================== هندسة ميزات المباريات التاريخية =====================

def engineer_match_features(
    matches: pd.DataFrame,
    competition: Optional[str] = None,
) -> pd.DataFrame:
    """
    الدالة الرئيسية لهندسة الميزات لجميع المباريات التاريخية.
    تُستخدم لإنشاء ملف التدريب.
    
    المدخل: DataFrame بالمباريات الخام
    المخرج: DataFrame بالميزات المحسوبة + عمود target
    """
    df = matches.copy()
    if competition:
        df = df[df["competition"] == competition].copy()
    df = parse_dates(df).dropna(subset=["home_goals", "away_goals"])

    # ── 1. Elo ──
    elo_df = compute_elo_for_matches(df)
    df = df.merge(elo_df, on="match_id", how="left")

    # ── 2. ترتيب المواسم السابقة ──
    season_final_ranks = get_season_final_ranks(df)

    # ── 3. بناء إطار الفرق ──
    tm = build_team_matches_frame(df)
    tm = add_rolling_team_features(tm)
    tm = add_performance_vs_ranks_features(tm, season_final_ranks)

    # ── 4. فصل ميزات المضيف والضيف ──
    all_feature_cols = list_feature_columns()
    team_features = sorted(
        list(
            set(
                c.split("_", 1)[1]
                for c in all_feature_cols
                if c.startswith(("h_", "a_"))
            )
        )
    )

    home_feats = tm[tm["is_home"]].copy()
    away_feats = tm[~tm["is_home"]].copy()

    # إعادة تسمية الأعمدة
    home_map = {feat: f"h_{feat}" for feat in team_features}
    away_map = {feat: f"a_{feat}" for feat in team_features}
    home_map.update({
        "team": "home_team",
        "season_loc_avg_gf": "h_season_home_avg_gf",
        "season_loc_avg_ga": "h_season_home_avg_ga",
    })
    away_map.update({
        "team": "away_team",
        "season_loc_avg_gf": "a_season_away_avg_gf",
        "season_loc_avg_ga": "a_season_away_avg_ga",
    })

    home_feats.rename(columns=home_map, inplace=True)
    away_feats.rename(columns=away_map, inplace=True)

    home_cols = ["match_id", "home_team"] + [
        c for c in home_feats.columns if c.startswith("h_")
    ]
    away_cols = ["match_id", "away_team"] + [
        c for c in away_feats.columns if c.startswith("a_")
    ]

    # ── 5. دمج الميزات مع المباريات ──
    out = (
        df.merge(home_feats[home_cols], on=["match_id", "home_team"], how="left")
          .merge(away_feats[away_cols], on=["match_id", "away_team"], how="left")
    )

    # ── 6. Elo features (already merged from step 1) ──
    out.rename(columns={
        "home_elo": "h_elo",
        "away_elo": "a_elo",
    }, inplace=True)
    if "elo_diff" not in out.columns:
        out["elo_diff"] = out.get("h_elo", 0) - out.get("a_elo", 0)
    out["diff_elo"] = out["elo_diff"]

    # ── 7. المواجهات المباشرة ──
    print("  حساب المواجهات المباشرة H2H...")
    h2h_vals = out.apply(
        lambda r: compute_h2h_for_home(r, df, k=3),
        axis=1,
        result_type="expand",
    )
    h2h_vals.columns = ["h2h_home_pts3", "h2h_home_avg_gf3", "h2h_home_avg_ga3"]
    out = pd.concat([out, h2h_vals], axis=1)

    # ── 8. الفروقات بين الفريقين ──
    for col_name in team_features:
        h_col = f"h_{col_name}"
        a_col = f"a_{col_name}"
        if h_col in out.columns and a_col in out.columns:
            out[f"diff_{col_name}"] = out[h_col] - out[a_col]

    # فروقات الموقع
    if "h_season_home_avg_gf" in out.columns and "a_season_away_avg_gf" in out.columns:
        out["diff_season_avg_gf"] = (
            out["h_season_home_avg_gf"] - out["a_season_away_avg_gf"]
        )
    else:
        out["diff_season_avg_gf"] = np.nan

    if "h_season_home_avg_ga" in out.columns and "a_season_away_avg_ga" in out.columns:
        out["diff_season_avg_ga"] = (
            out["h_season_home_avg_ga"] - out["a_season_away_avg_ga"]
        )
    else:
        out["diff_season_avg_ga"] = np.nan

    # فروقات v4
    out["diff_win_streak"] = out.get("h_win_streak", 0) - out.get("a_win_streak", 0)
    out["diff_unbeaten_streak"] = (
        out.get("h_unbeaten_streak", 0) - out.get("a_unbeaten_streak", 0)
    )

    # ── 9. المتغير الهدف ──
    out["target"] = out.apply(
        lambda r: match_result_label(int(r["home_goals"]), int(r["away_goals"])),
        axis=1,
    )

    # ── 10. ترتيب الأعمدة النهائية ──
    meta_cols = [
        "match_id", "date", "season_start", "competition",
        "home_team", "away_team",
    ]
    final_cols = meta_cols + list_feature_columns() + ["target"]
    final_cols = [c for c in final_cols if c in out.columns]

    return out[final_cols].sort_values("date").reset_index(drop=True)


# ===================== أدوات تطابق أسماء الفرق =====================

def _to_utc(ts: Any) -> pd.Timestamp:
    """تحويل أي تمثيل زمني إلى Timestamp بتوقيت UTC."""
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _normalize_name(s: str) -> str:
    """توحيد اسم الفريق بإزالة الاختصارات والأحرف الخاصة."""
    s = (s or "").lower()
    s = re.sub(r"\b(fc|cf|sc)\b\.?", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_team_name(name_input: str, team_pool: List[str]) -> str:
    """
    محاولة مطابقة اسم الفريق مع الأسماء الموجودة في البيانات.
    
    مراحل المطابقة:
    1. مطابقة تامة
    2. مطابقة بعد التوحيد (إزالة FC, SC...)
    3. مطابقة تقريبية (fuzzy matching)
    """
    # 1. مطابقة تامة
    if name_input in team_pool:
        return name_input

    # 2. مطابقة بعد التوحيد
    norm_pool_map = {_normalize_name(t): t for t in team_pool}
    norm_in = _normalize_name(name_input)
    if norm_in in norm_pool_map:
        return norm_pool_map[norm_in]

    # 3. مطابقة تقريبية على الأسماء الأصلية
    cand = get_close_matches(name_input, team_pool, n=1, cutoff=0.55)
    if cand:
        return cand[0]

    # 4. مطابقة تقريبية على الأسماء الموحدة
    cand2 = get_close_matches(
        norm_in, list(norm_pool_map.keys()), n=1, cutoff=0.55
    )
    if cand2:
        return norm_pool_map[cand2[0]]

    # 5. لم نجد — نرجع الاسم كما هو
    return name_input


# ===================== ميزات مباراة واحدة للتنبؤ =====================

def compute_single_pair_features(
    matches: pd.DataFrame,
    competition: str,
    home_team_input: str,
    away_team_input: str,
    ref_datetime: Any,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    حساب ميزات مباراة واحدة (للتنبؤ الآني).
    
    المدخلات:
    - matches: كل المباريات التاريخية
    - competition: رمز الدوري
    - home_team_input: اسم المضيف
    - away_team_input: اسم الضيف
    - ref_datetime: تاريخ المباراة المراد التنبؤ بها
    
    المخرجات:
    - X: DataFrame صف واحد بالميزات
    - meta: dict بالأسماء الموحدة
    """
    req = [
        "match_id", "date", "season_start", "matchday",
        "home_team", "away_team", "home_goals", "away_goals",
        "competition",
    ]
    for c in req:
        if c not in matches.columns:
            raise ValueError(f"العمود مفقود في البيانات: {c}")

    df = matches.copy()
    if competition:
        df = df[df["competition"] == competition].copy()

    df = parse_dates(df)
    ref_ts = _to_utc(ref_datetime)
    hist = df[df["date"] < ref_ts].copy()

    cols = list_feature_columns()

    # إذا لا توجد بيانات تاريخية
    if hist.empty:
        X_empty = pd.DataFrame([np.nan] * len(cols), index=cols).T
        return X_empty, {
            "home_team_resolved": home_team_input,
            "away_team_resolved": away_team_input,
        }

    # مطابقة أسماء الفرق
    team_pool = sorted(set(hist["home_team"]).union(set(hist["away_team"])))
    home_res = _resolve_team_name(home_team_input, team_pool)
    away_res = _resolve_team_name(away_team_input, team_pool)

    # ── Elo ──
    elo_df = compute_elo_for_matches(hist)
    elo_sys = EloSystem()
    # إعادة بناء التصنيفات
    for _, row in hist.iterrows():
        elo_sys.update(
            row["home_team"], row["away_team"],
            int(row["home_goals"]), int(row["away_goals"]),
        )
    home_elo = elo_sys.get(home_res)
    away_elo = elo_sys.get(away_res)

    # ── بناء الميزات ──
    season_final_ranks = get_season_final_ranks(hist)
    tm = build_team_matches_frame(hist)
    tm = add_rolling_team_features(tm)
    tm = add_performance_vs_ranks_features(tm, season_final_ranks)

    def last_row(
        team: str, is_home: Optional[bool] = None
    ) -> Optional[pd.Series]:
        """جلب آخر صف لفريق معين."""
        sub = tm[
            (tm["team"] == team) & (tm["competition"] == competition)
        ]
        if is_home is not None:
            sub = sub[sub["is_home"] == is_home]
        if sub.empty:
            return None
        return sub.sort_values(["date", "match_id"]).iloc[-1]

    h_last_all = last_row(home_res, is_home=None)
    a_last_all = last_row(away_res, is_home=None)
    h_last_home = last_row(home_res, is_home=True)
    a_last_away = last_row(away_res, is_home=False)

    all_feature_cols = list_feature_columns()
    team_features = sorted(
        list(
            set(
                c.split("_", 1)[1]
                for c in all_feature_cols
                if c.startswith(("h_", "a_"))
            )
        )
    )

    out: Dict[str, float] = {}

    def get_feat_from_rows(feat_suffix: str, side: str) -> float:
        """جلب قيمة ميزة من آخر صف متاح."""
        if side == "h":
            row_all, row_loc = h_last_all, h_last_home
            if feat_suffix == "season_home_avg_gf":
                return (
                    np.nan
                    if row_loc is None
                    else row_loc.get("season_loc_avg_gf", np.nan)
                )
            if feat_suffix == "season_home_avg_ga":
                return (
                    np.nan
                    if row_loc is None
                    else row_loc.get("season_loc_avg_ga", np.nan)
                )
        else:
            row_all, row_loc = a_last_all, a_last_away
            if feat_suffix == "season_away_avg_gf":
                return (
                    np.nan
                    if row_loc is None
                    else row_loc.get("season_loc_avg_gf", np.nan)
                )
            if feat_suffix == "season_away_avg_ga":
                return (
                    np.nan
                    if row_loc is None
                    else row_loc.get("season_loc_avg_ga", np.nan)
                )
        if row_all is None:
            return np.nan
        return row_all.get(feat_suffix, np.nan)

    for feat in team_features:
        out[f"h_{feat}"] = get_feat_from_rows(feat, "h")
        out[f"a_{feat}"] = get_feat_from_rows(feat, "a")

    # ── Elo ──
    out["h_elo"] = home_elo
    out["a_elo"] = away_elo
    out["elo_diff"] = home_elo - away_elo
    out["diff_elo"] = home_elo - away_elo

    # ── H2H ──
    h2h_series = pd.Series({
        "competition": competition,
        "home_team": home_res,
        "away_team": away_res,
        "date": ref_ts,
    })
    h2h_pts, h2h_avg_gf, h2h_avg_ga = compute_h2h_for_home(
        h2h_series, hist, k=3
    )
    out["h2h_home_pts3"] = h2h_pts
    out["h2h_home_avg_gf3"] = h2h_avg_gf
    out["h2h_home_avg_ga3"] = h2h_avg_ga

    # ── الفروقات ──
    time_windows = [3, 5, 10]
    metrics = ["pts", "gf", "ga", "win_rate", "cs_rate"]
    for w in time_windows:
        for m in metrics:
            h_val = out.get(f"h_{m}{w}", np.nan)
            a_val = out.get(f"a_{m}{w}", np.nan)
            out[f"diff_{m}{w}"] = _safe_diff(h_val, a_val)

            h_ema = out.get(f"h_ema_{m}{w}", np.nan)
            a_ema = out.get(f"a_ema_{m}{w}", np.nan)
            out[f"diff_ema_{m}{w}"] = _safe_diff(h_ema, a_ema)

    out["diff_season_avg_gf"] = _safe_diff(
        out.get("h_season_home_avg_gf", np.nan),
        out.get("a_season_away_avg_gf", np.nan),
    )
    out["diff_season_avg_ga"] = _safe_diff(
        out.get("h_season_home_avg_ga", np.nan),
        out.get("a_season_away_avg_ga", np.nan),
    )
    out["diff_days_since_last"] = _safe_diff(
        out.get("h_days_since_last", np.nan),
        out.get("a_days_since_last", np.nan),
    )
    out["diff_season_cum_gd"] = _safe_diff(
        out.get("h_season_cum_gd", np.nan),
        out.get("a_season_cum_gd", np.nan),
    )
    out["diff_ppg_vs_top5"] = _safe_diff(
        out.get("h_ppg_vs_top5", np.nan),
        out.get("a_ppg_vs_top5", np.nan),
    )
    out["diff_ppg_vs_bot5"] = _safe_diff(
        out.get("h_ppg_vs_bot5", np.nan),
        out.get("a_ppg_vs_bot5", np.nan),
    )
    out["diff_win_streak"] = _safe_diff(
        out.get("h_win_streak", np.nan),
        out.get("a_win_streak", np.nan),
    )
    out["diff_unbeaten_streak"] = _safe_diff(
        out.get("h_unbeaten_streak", np.nan),
        out.get("a_unbeaten_streak", np.nan),
    )

    # ── بناء DataFrame ──
    X = pd.DataFrame([out])
    for c in all_feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[all_feature_cols]

    meta = {
        "home_team_resolved": home_res,
        "away_team_resolved": away_res,
    }
    return X, meta


def _safe_diff(a, b):
    """فرق آمن يتعامل مع NaN."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return a - b
