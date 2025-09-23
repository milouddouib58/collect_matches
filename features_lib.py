# features_lib.py
# -*- coding: utf-8 -*-
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import get_close_matches
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd

# ===================== ثوابت عامة =====================
RESULT_LABELS = ["H", "D", "A"]
FEATURE_VERSION = "v3"

# ===================== أدوات مساعدة عامة =====================
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def match_result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"

def points_from_score(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0

def infer_season_start_for_date(ts: pd.Timestamp) -> str:
    y = ts.year if ts.month >= 7 else ts.year - 1
    return f"{y}-07-01"

def list_feature_columns() -> List[str]:
    features: List[str] = []
    time_windows = [3, 5, 10]
    metrics = ["pts", "gf", "ga", "win_rate", "cs_rate"]

    for side in ["h", "a"]:
        for metric in metrics:
            for w in time_windows:
                features.append(f"{side}_{metric}{w}")
                features.append(f"{side}_ema_{metric}{w}")

        # ميزات إضافية لكل جانب
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

        features.extend([
            f"{side}_days_since_last",
            f"{side}_season_cum_gd",
            f"{side}_ppg_vs_top5",
            f"{side}_ppg_vs_bot5",
        ])

    # ميزات المواجهات المباشرة
    features.extend(["h2h_home_pts3", "h2h_home_avg_gf3", "h2h_home_avg_ga3"])

    # فروقات
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
    ])

    return sorted(list(set(features)))

# ===================== بناء إطار مباريات الفرق =====================
def build_team_matches_frame(matches: pd.DataFrame) -> pd.DataFrame:
    req_cols = ["match_id", "date", "season_start", "matchday", "home_team", "away_team", "home_goals", "away_goals", "competition"]
    for c in req_cols:
        if c not in matches.columns:
            raise ValueError(f"العمود مفقود في البيانات: {c}")

    m = parse_dates(matches.copy())

    home = m[req_cols].rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "gf",
        "away_goals": "ga"
    })
    home["is_home"] = True

    away = m[req_cols].rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "gf",
        "home_goals": "ga"
    })
    away["is_home"] = False

    tm = pd.concat([home, away], ignore_index=True)
    tm["points"] = tm.apply(lambda r: points_from_score(r["gf"], r["ga"]), axis=1)
    tm = tm.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    return tm

def add_rolling_team_features(team_matches: pd.DataFrame) -> pd.DataFrame:
    tm = team_matches.copy().sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g = tm.groupby(["team", "competition"], sort=False)
    time_windows = [3, 5, 10]

    tm["win"] = (tm["gf"] > tm["ga"]).astype(int)
    tm["clean_sheet"] = (tm["ga"] == 0).astype(int)

    for w in time_windows:
        tm[f'pts{w}'] = g['points'].shift(1).rolling(w, min_periods=1).sum()
        tm[f'gf{w}'] = g['gf'].shift(1).rolling(w, min_periods=1).mean()
        tm[f'ga{w}'] = g['ga'].shift(1).rolling(w, min_periods=1).mean()
        tm[f'win_rate{w}'] = g['win'].shift(1).rolling(w, min_periods=1).mean()
        tm[f'cs_rate{w}'] = g['clean_sheet'].shift(1).rolling(w, min_periods=1).mean()

        tm[f'ema_pts{w}'] = g['points'].shift(1).ewm(span=w, min_periods=1).sum()
        tm[f'ema_gf{w}'] = g['gf'].shift(1).ewm(span=w, min_periods=1).mean()
        tm[f'ema_ga{w}'] = g['ga'].shift(1).ewm(span=w, min_periods=1).mean()
        tm[f'ema_win_rate{w}'] = g['win'].shift(1).ewm(span=w, min_periods=1).mean()
        tm[f'ema_cs_rate{w}'] = g['clean_sheet'].shift(1).ewm(span=w, min_periods=1).mean()

    tm["days_since_last"] = g["date"].diff().dt.total_seconds() / 86400.0

    kg = ["team", "competition", "season_start"]
    loc_kg = kg + ["is_home"]

    tm["cum_gf_prev"] = tm.groupby(kg)["gf"].cumsum() - tm["gf"]
    tm["cum_ga_prev"] = tm.groupby(kg)["ga"].cumsum() - tm["ga"]
    tm["season_cum_gd"] = tm["cum_gf_prev"] - tm["cum_ga_prev"]

    loc_counts = tm.groupby(loc_kg).cumcount()
    tm["season_loc_avg_gf"] = (tm.groupby(loc_kg)["gf"].cumsum() - tm["gf"]) / loc_counts
    tm["season_loc_avg_ga"] = (tm.groupby(loc_kg)["ga"].cumsum() - tm["ga"]) / loc_counts

    tm.replace([np.inf, -np.inf], np.nan, inplace=True)
    return tm

def get_season_final_ranks(matches: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    ranks_by_season: Dict[str, Dict[str, int]] = {}
    matches = parse_dates(matches)
    for season_start, season_df in matches.groupby("season_start"):
        table = defaultdict(lambda: {'pts': 0, 'gd': 0, 'gf': 0})
        for row in season_df.itertuples():
            h, a, hg, ag = row.home_team, row.away_team, row.home_goals, row.away_goals
            table[h]['pts'] += points_from_score(hg, ag); table[h]['gd'] += hg - ag; table[h]['gf'] += hg
            table[a]['pts'] += points_from_score(ag, hg); table[a]['gd'] += ag - hg; table[a]['gf'] += ag
        sorted_table = sorted(table.items(), key=lambda item: (item[1]['pts'], item[1]['gd'], item[1]['gf']), reverse=True)
        ranks = {team: i + 1 for i, (team, stats) in enumerate(sorted_table)}
        ranks_by_season[season_start] = ranks
    return ranks_by_season

def add_performance_vs_ranks_features(team_matches: pd.DataFrame, season_ranks: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    tm = team_matches.copy()
    tm['prev_season_start'] = (pd.to_datetime(tm['season_start']).dt.year - 1).astype(str) + "-07-01"
    tm['opponent_rank_prev_season'] = tm.apply(lambda r: season_ranks.get(r['prev_season_start'], {}).get(r['opponent'], 20), axis=1)
    tm['opponent_strength'] = pd.cut(tm['opponent_rank_prev_season'], bins=[0, 5, 15, 21], labels=['top5', 'mid', 'bot5'])
    tm['points_vs_top5'] = tm.where(tm['opponent_strength'] == 'top5')['points']
    tm['points_vs_bot5'] = tm.where(tm['opponent_strength'] == 'bot5')['points']

    g = tm.groupby(['team', 'season_start'])
    prev_cum_pts_vs_top5 = g['points_vs_top5'].shift(1).fillna(0).cumsum()
    prev_cum_games_vs_top5 = g['points_vs_top5'].shift(1).notna().cumsum()
    prev_cum_pts_vs_bot5 = g['points_vs_bot5'].shift(1).fillna(0).cumsum()
    prev_cum_games_vs_bot5 = g['points_vs_bot5'].shift(1).notna().cumsum()

    tm['ppg_vs_top5'] = prev_cum_pts_vs_top5 / prev_cum_games_vs_top5
    tm['ppg_vs_bot5'] = prev_cum_pts_vs_bot5 / prev_cum_games_vs_bot5

    tm.fillna({'ppg_vs_top5': 0, 'ppg_vs_bot5': 0}, inplace=True)
    return tm

# ===================== هندسة الميزات للمباريات التاريخية =====================
def compute_h2h_for_home(row: pd.Series, matches: pd.DataFrame, k: int = 3) -> Tuple[float, float, float]:
    comp, home, away, dt = row.competition, row.home_team, row.away_team, row.date
    prev = matches[
        (matches.competition == comp) &
        (matches.date < dt) &
        (
            ((matches.home_team == home) & (matches.away_team == away)) |
            ((matches.home_team == away) & (matches.away_team == home))
        )
    ].sort_values("date", ascending=False).head(k)
    if prev.empty:
        return (np.nan, np.nan, np.nan)
    pts, gf_sum, ga_sum = 0, 0.0, 0.0
    for _, m in prev.iterrows():
        gf, ga = (m.home_goals, m.away_goals) if m.home_team == home else (m.away_goals, m.home_goals)
        gf_sum += gf; ga_sum += ga; pts += points_from_score(gf, ga)
    return (float(pts), gf_sum / len(prev), ga_sum / len(prev))

def engineer_match_features(matches: pd.DataFrame, competition: Optional[str] = None) -> pd.DataFrame:
    df = matches.copy()
    if competition:
        df = df[df["competition"] == competition].copy()
    df = parse_dates(df).dropna(subset=["home_goals", "away_goals"])

    season_final_ranks = get_season_final_ranks(df)
    tm = build_team_matches_frame(df)
    tm = add_rolling_team_features(tm)
    tm = add_performance_vs_ranks_features(tm, season_final_ranks)

    all_feature_cols = list_feature_columns()
    team_features = sorted(list(set([c.split('_', 1)[1] for c in all_feature_cols if c.startswith(('h_', 'a_'))])))

    home_feats = tm[tm["is_home"]].copy()
    away_feats = tm[~tm["is_home"]].copy()

    home_map = {feat: f"h_{feat}" for feat in team_features}
    away_map = {feat: f"a_{feat}" for feat in team_features}
    home_map.update({"team": "home_team", "season_loc_avg_gf": "h_season_home_avg_gf", "season_loc_avg_ga": "h_season_home_avg_ga"})
    away_map.update({"team": "away_team", "season_loc_avg_gf": "a_season_away_avg_gf", "season_loc_avg_ga": "a_season_away_avg_ga"})

    home_feats.rename(columns=home_map, inplace=True)
    away_feats.rename(columns=away_map, inplace=True)

    home_cols = ['match_id', 'home_team'] + [c for c in home_feats.columns if c.startswith('h_')]
    away_cols = ['match_id', 'away_team'] + [c for c in away_feats.columns if c.startswith('a_')]

    out = df.merge(home_feats[home_cols], on=["match_id", "home_team"], how="left") \
            .merge(away_feats[away_cols], on=["match_id", "away_team"], how="left")

    h2h_vals = out.apply(lambda r: compute_h2h_for_home(r, df, k=3), axis=1, result_type="expand")
    h2h_vals.columns = ["h2h_home_pts3", "h2h_home_avg_gf3", "h2h_home_avg_ga3"]
    out = pd.concat([out, h2h_vals], axis=1)

    for col_name in team_features:
        h_col, a_col = f"h_{col_name}", f"a_{col_name}"
        if h_col in out.columns and a_col in out.columns:
            out[f"diff_{col_name}"] = out[h_col] - out[a_col]

    out["target"] = out.apply(lambda r: match_result_label(int(r["home_goals"]), int(r["away_goals"])), axis=1)

    final_cols = ["match_id", "date", "season_start", "competition", "home_team", "away_team"] + list_feature_columns() + ["target"]
    final_cols = [c for c in final_cols if c in out.columns]
    return out[final_cols].sort_values("date").reset_index(drop=True)

# ===================== ميزات مباراة واحدة للتنبؤ =====================
def _to_utc(ts: Any) -> pd.Timestamp:
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t

def _normalize_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'\b(fc|cf|sc)\b\.?', '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _resolve_team_name(name_input: str, team_pool: List[str]) -> str:
    if name_input in team_pool:
        return name_input
    norm_pool_map = {_normalize_name(t): t for t in team_pool}
    norm_in = _normalize_name(name_input)
    if norm_in in norm_pool_map:
        return norm_pool_map[norm_in]
    cand = get_close_matches(name_input, team_pool, n=1, cutoff=0.6)
    if cand:
        return cand[0]
    cand2 = get_close_matches(norm_in, list(norm_pool_map.keys()), n=1, cutoff=0.6)
    if cand2:
        return norm_pool_map[cand2[0]]
    return name_input

def compute_single_pair_features(
    matches: pd.DataFrame,
    competition: str,
    home_team_input: str,
    away_team_input: str,
    ref_datetime: Any
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    استخراج ميزات مباراة واحدة قبل ref_datetime بنَفَس منطق engineer_match_features.
    """
    req_cols = ["match_id", "date", "season_start", "matchday", "home_team", "away_team", "home_goals", "away_goals", "competition"]
    for c in req_cols:
        if c not in matches.columns:
            raise ValueError(f"العمود مفقود في البيانات: {c}")

    df = matches.copy()
    if competition:
        df = df[df["competition"] == competition].copy()

    df = parse_dates(df)
    ref_ts = _to_utc(ref_datetime)
    hist = df[df["date"] < ref_ts].copy()

    # لا توجد بيانات تاريخية
    cols = list_feature_columns()
    if hist.empty:
        X_empty = pd.DataFrame([np.nan] * len(cols), index=cols).T
        return X_empty, {"home_team_resolved": home_team_input, "away_team_resolved": away_team_input}

    # حل الأسماء
    team_pool = sorted(set(hist["home_team"]).union(set(hist["away_team"])))
    home_res = _resolve_team_name(home_team_input, team_pool)
    away_res = _resolve_team_name(away_team_input, team_pool)

    # بناء ميزات الفرق
    season_final_ranks = get_season_final_ranks(hist)
    tm = build_team_matches_frame(hist)
    tm = add_rolling_team_features(tm)
    tm = add_performance_vs_ranks_features(tm, season_final_ranks)

    def last_row(team: str, is_home: Optional[bool] = None) -> Optional[pd.Series]:
        sub = tm[(tm["team"] == team) & (tm["competition"] == competition)]
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
    team_features = sorted(list(set([c.split('_', 1)[1] for c in all_feature_cols if c.startswith(('h_', 'a_'))])))

    out: Dict[str, float] = {}

    def get_feat_from_rows(feat_suffix: str, side: str) -> float:
        if side == "h":
            row_all, row_loc = h_last_all, h_last_home
            if feat_suffix == "season_home_avg_gf":
                return np.nan if row_loc is None else row_loc.get("season_loc_avg_gf", np.nan)
            if feat_suffix == "season_home_avg_ga":
                return np.nan if row_loc is None else row_loc.get("season_loc_avg_ga", np.nan)
        else:
            row_all, row_loc = a_last_all, a_last_away
            if feat_suffix == "season_away_avg_gf":
                return np.nan if row_loc is None else row_loc.get("season_loc_avg_gf", np.nan)
            if feat_suffix == "season_away_avg_ga":
                return np.nan if row_loc is None else row_loc.get("season_loc_avg_ga", np.nan)
        if row_all is None:
            return np.nan
        return row_all.get(feat_suffix, np.nan)

    # تعبئة h_* و a_*
    for feat in team_features:
        out[f"h_{feat}"] = get_feat_from_rows(feat, "h")
        out[f"a_{feat}"] = get_feat_from_rows(feat, "a")

    # h2h قبل ref_ts
    h2h_series = pd.Series({
        "competition": competition,
        "home_team": home_res,
        "away_team": away_res,
        "date": ref_ts
    })
    h2h_pts, h2h_avg_gf, h2h_avg_ga = compute_h2h_for_home(h2h_series, hist, k=3)
    out["h2h_home_pts3"] = h2h_pts
    out["h2h_home_avg_gf3"] = h2h_avg_gf
    out["h2h_home_avg_ga3"] = h2h_avg_ga

    # فروقات diff_*
    time_windows = [3, 5, 10]
    metrics = ["pts", "gf", "ga", "win_rate", "cs_rate"]
    for w in time_windows:
        for m in metrics:
            out[f"diff_{m}{w}"] = (out.get(f"h_{m}{w}", np.nan) - out.get(f"a_{m}{w}", np.nan))
            out[f"diff_ema_{m}{w}"] = (out.get(f"h_ema_{m}{w}", np.nan) - out.get(f"a_ema_{m}{w}", np.nan))

    out["diff_season_avg_gf"] = (out.get("h_season_home_avg_gf", np.nan) - out.get("a_season_away_avg_gf", np.nan))
    out["diff_season_avg_ga"] = (out.get("h_season_home_avg_ga", np.nan) - out.get("a_season_away_avg_ga", np.nan))
    out["diff_days_since_last"] = (out.get("h_days_since_last", np.nan) - out.get("a_days_since_last", np.nan))
    out["diff_season_cum_gd"] = (out.get("h_season_cum_gd", np.nan) - out.get("a_season_cum_gd", np.nan))
    out["diff_ppg_vs_top5"] = (out.get("h_ppg_vs_top5", np.nan) - out.get("a_ppg_vs_top5", np.nan))
    out["diff_ppg_vs_bot5"] = (out.get("h_ppg_vs_bot5", np.nan) - out.get("a_ppg_vs_bot5", np.nan))

    # DataFrame بترتيب الأعمدة
    X = pd.DataFrame([out])
    for c in all_feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[all_feature_cols]

    meta = {
        "home_team_resolved": home_res,
        "away_team_resolved": away_res
    }
    return X, meta
