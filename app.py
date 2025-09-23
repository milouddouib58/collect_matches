# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
import inspect
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
import joblib
from typing import Dict, List, Optional, Tuple

# NEW: Skellam Bessel I0 (مع fallback لو scipy مش متوفّر)
try:
    from scipy.special import i0 as bessel_i0
except Exception:
    def bessel_i0(x):
        x = np.asarray(x, dtype=float)
        y = 1.0
        t = 1.0
        k = 1
        for _ in range(30):
            t *= (x / (2.0 * k)) ** 2
            y += t
            k += 1
        return y

# إعدادات
INPUT_CSV_FILE = "historical_dataset.csv"
MODEL_OUTPUT_FILE = "football_model_meta_stacking.joblib"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# بحث وضبط
N_TRIALS_BASE = 8                 # تجارب لاختيار baseline
ACC_DROP_ALLOWED = 0.01           # لا نسمح بخفض الدقة أكثر من 1%

# اكتشاف نمط الإيقاف المبكر
def detect_es_mode() -> str:
    dummy = xgb.XGBClassifier()
    sig = inspect.signature(dummy.fit)
    if 'early_stopping_rounds' in sig.parameters:
        return 'early_stopping_rounds'
    if 'callbacks' in sig.parameters:
        return 'callbacks'
    return 'none'
ES_MODE = detect_es_mode()
N_ESTIMATORS_RS = 3000 if ES_MODE in ('early_stopping_rounds', 'callbacks') else 800

def xgb_fit_with_es(model, X_train, y_train, sw_train, X_val, y_val, es_rounds=150):
    kwargs = dict(sample_weight=sw_train, eval_set=[(X_val, y_val)])
    if 'verbose' in inspect.signature(model.fit).parameters:
        kwargs['verbose'] = False
    if ES_MODE == 'early_stopping_rounds':
        kwargs['early_stopping_rounds'] = es_rounds
    elif ES_MODE == 'callbacks':
        try:
            from xgboost.callback import EarlyStopping
            kwargs['callbacks'] = [EarlyStopping(rounds=es_rounds, save_best=True)]
        except Exception:
            pass
    return model.fit(X_train, y_train, **kwargs)

def get_best_iteration(model) -> Optional[int]:
    bi = getattr(model, "best_iteration", None)
    if bi is not None:
        return int(bi)
    try:
        booster = model.get_booster()
        if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
            return int(booster.best_iteration)
        if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
            return int(booster.best_ntree_limit)
    except Exception:
        pass
    return None

# تقسيم زمني على مستوى اليوم
def time_split_by_day(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df['day'] = df['date'].dt.normalize()

    unique_days = pd.Series(df['day'].unique()).sort_values().to_list()
    n_days = len(unique_days)
    if n_days < 10:
        n = len(df)
        tr_end = max(1, int(n * train_frac))
        va_end = max(tr_end + 1, int(n * (train_frac + val_frac)))
        va_end = min(va_end, n - 1)
        return df.iloc[:tr_end].copy(), df.iloc[tr_end:va_end].copy(), df.iloc[va_end:].copy()

    tr_end_day = max(1, int(n_days * train_frac))
    va_end_day = max(tr_end_day + 1, int(n_days * (train_frac + val_frac)))
    va_end_day = min(va_end_day, n_days - 1)

    train_days = set(unique_days[:tr_end_day])
    val_days = set(unique_days[tr_end_day:va_end_day])
    test_days = set(unique_days[va_end_day:])

    df_train = df[df['day'].isin(train_days)].drop(columns=['day'])
    df_val = df[df['day'].isin(val_days)].drop(columns=['day'])
    df_test = df[df['day'].isin(test_days)].drop(columns=['day'])
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# معايرة Temperature Scaling
@dataclass
class TemperatureScaler:
    T: float = 1.0
    eps: float = 1e-12
    def fit(self, proba: np.ndarray, y_true: np.ndarray):
        y_true = np.asarray(y_true)
        proba = np.clip(proba, self.eps, 1 - self.eps)
        grid = np.linspace(0.6, 2.5, 26)
        best_T, best_loss = 1.0, np.inf
        for T in grid:
            p_cal = self._softmax(np.log(proba) / T)
            ll = log_loss(y_true, p_cal, labels=np.arange(proba.shape[1]))
            if ll < best_loss:
                best_loss, best_T = ll, T
        fine = np.linspace(max(0.2, best_T - 0.5), best_T + 0.5, 31)
        for T in fine:
            p_cal = self._softmax(np.log(proba) / T)
            ll = log_loss(y_true, p_cal, labels=np.arange(proba.shape[1]))
            if ll < best_loss:
                best_loss, best_T = ll, T
        self.T = float(best_T); return self
    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = np.clip(proba, self.eps, 1 - self.eps)
        return self._softmax(np.log(proba) / self.T)
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

# مُولّد الميزات
class FeatureBuilder:
    def __init__(self, comp_alpha: float = 30.0):
        self.comp_alpha = comp_alpha
        self.global_prior_: Optional[np.ndarray] = None
        self.comp_priors_: Dict[str, np.ndarray] = {}
        self.features_: List[str] = []
    def fit(self, df: pd.DataFrame, y: pd.Series):
        assert 'competition' in df.columns, "عمود competition مفقود"
        counts = y.value_counts().reindex([0, 1, 2], fill_value=0).values.astype(float)
        self.global_prior_ = counts / counts.sum()
        tmp = df[['competition']].copy()
        tmp['target'] = y.values
        for comp, g in tmp.groupby('competition'):
            c = g['target'].value_counts().reindex([0, 1, 2], fill_value=0).values.astype(float)
            total = c.sum()
            smooth = (c + self.comp_alpha * self.global_prior_) / (total + self.comp_alpha)
            self.comp_priors_[comp] = smooth
        return self
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        elo_home = pd.to_numeric(df['home_team_elo'], errors='coerce').astype(float)
        elo_away = pd.to_numeric(df['away_team_elo'], errors='coerce').astype(float)
        elo_diff = elo_home - elo_away
        X['elo_diff'] = elo_diff
        X['elo_sum'] = elo_home + elo_away
        X['elo_ratio'] = (elo_home + 1.0) / (elo_away + 1.0)
        X['elo_prob_home'] = 1.0 / (1.0 + 10 ** (-(elo_diff) / 400.0))
        X['abs_elo_diff'] = np.abs(elo_diff)
        h_pts = pd.to_numeric(df['home_form_points'], errors='coerce').astype(float)
        a_pts = pd.to_numeric(df['away_form_points'], errors='coerce').astype(float)
        h_gs = pd.to_numeric(df['home_form_gs'], errors='coerce').astype(float)
        h_gc = pd.to_numeric(df['home_form_gc'], errors='coerce').astype(float)
        a_gs = pd.to_numeric(df['away_form_gs'], errors='coerce').astype(float)
        a_gc = pd.to_numeric(df['away_form_gc'], errors='coerce').astype(float)
        X['form_points_diff'] = h_pts - a_pts
        X['abs_form_points_diff'] = np.abs(X['form_points_diff'])
        X['form_gd_home'] = h_gs - h_gc
        X['form_gd_away'] = a_gs - a_gc
        X['form_gd_diff'] = X['form_gd_home'] - X['form_gd_away']
        X['abs_form_gd_diff'] = np.abs(X['form_gd_diff'])
        X['total_form_gs'] = h_gs + a_gs
        X['total_form_gc'] = h_gc + a_gc
        X['closeness_elo'] = np.exp(-X['abs_elo_diff'] / 200.0)
        X['closeness_points'] = np.exp(-X['abs_form_points_diff'] / 6.0)
        X['closeness_gd'] = np.exp(-X['abs_form_gd_diff'] / 4.0)
        X['low_total_proxy'] = np.exp(-(X['total_form_gs'] + X['total_form_gc']) / 12.0)
        if 'home_team_league_pos' in df.columns and 'away_team_league_pos' in df.columns:
            h_pos = pd.to_numeric(df['home_team_league_pos'], errors='coerce')
            a_pos = pd.to_numeric(df['away_team_league_pos'], errors='coerce')
            X['league_pos_diff'] = (a_pos - h_pos).fillna(0.0)
            X['home_top4'] = (h_pos <= 4).fillna(False).astype(int)
            X['away_top4'] = (a_pos <= 4).fillna(False).astype(int)
        else:
            X['league_pos_diff'] = 0.0
            X['home_top4'] = 0
            X['away_top4'] = 0
        dates = pd.to_datetime(df['date'], errors='coerce')
        X['is_weekend'] = dates.dt.weekday.isin([5, 6]).astype(int)
        X['month_sin'] = np.sin(2 * np.pi * (dates.dt.month.fillna(1) / 12))
        X['month_cos'] = np.cos(2 * np.pi * (dates.dt.month.fillna(1) / 12))
        comp = df['competition'].astype(str)
        comp_home, comp_draw, comp_away = [], [], []
        for c in comp:
            p = self.comp_priors_.get(c, self.global_prior_)
            comp_home.append(p[0]); comp_draw.append(p[1]); comp_away.append(p[2])
        X['comp_home_prior'] = comp_home
        X['comp_draw_prior'] = comp_draw
        X['comp_away_prior'] = comp_away
        self.features_ = list(X.columns)
        return X.fillna(0.0)
    def get_feature_names_out(self):
        return self.features_

# Priors على مستوى الفريق
class TeamPriors:
    def __init__(self, alpha: float = 50.0):
        self.alpha = alpha
        self.global_draw_rate = 0.3
        self.global_home_gf = 1.4
        self.global_home_ga = 1.1
        self.global_away_gf = 1.1
        self.global_away_ga = 1.4
        self.team_stats_: Dict[str, dict] = {}
        self.features_: List[str] = []
    def fit(self, df_train: pd.DataFrame):
        df = df_train.copy()
        self.global_draw_rate = (df['target'] == 1).mean()
        self.global_home_gf = df['result_ft_home_goals'].mean()
        self.global_home_ga = df['result_ft_away_goals'].mean()
        self.global_away_gf = df['result_ft_away_goals'].mean()
        self.global_away_ga = df['result_ft_home_goals'].mean()
        home_grp = df.groupby('home_team_id').agg(
            home_n=('home_team_id', 'size'),
            home_draws=('target', lambda s: (s == 1).sum()),
            home_gf=('result_ft_home_goals', 'sum'),
            home_ga=('result_ft_away_goals', 'sum')
        )
        away_grp = df.groupby('away_team_id').agg(
            away_n=('away_team_id', 'size'),
            away_draws=('target', lambda s: (s == 1).sum()),
            away_gf=('result_ft_away_goals', 'sum'),
            away_ga=('result_ft_home_goals', 'sum')
        )
        teams = set(home_grp.index.astype(str)).union(set(away_grp.index.astype(str)))
        self.team_stats_ = {}
        for t in teams:
            hs = home_grp.loc[t] if t in home_grp.index else None
            as_ = away_grp.loc[t] if t in away_grp.index else None
            home_n = int(hs['home_n']) if hs is not None else 0
            away_n = int(as_['away_n']) if as_ is not None else 0
            home_draws = int(hs['home_draws']) if hs is not None else 0
            away_draws = int(as_['away_draws']) if as_ is not None else 0
            home_gf = float(hs['home_gf']) if hs is not None else 0.0
            home_ga = float(hs['home_ga']) if hs is not None else 0.0
            away_gf = float(as_['away_gf']) if as_ is not None else 0.0
            away_ga = float(as_['away_ga']) if as_ is not None else 0.0
            a = self.alpha
            p_home_draw = (home_draws + a * self.global_draw_rate) / (home_n + a) if (home_n + a) > 0 else self.global_draw_rate
            p_away_draw = (away_draws + a * self.global_draw_rate) / (away_n + a) if (away_n + a) > 0 else self.global_draw_rate
            p_overall_draw = (home_draws + away_draws + a * self.global_draw_rate) / (home_n + away_n + a) if (home_n + away_n + a) > 0 else self.global_draw_rate
            home_gf_avg = (home_gf + a * self.global_home_gf) / (home_n + a) if (home_n + a) > 0 else self.global_home_gf
            home_ga_avg = (home_ga + a * self.global_home_ga) / (home_n + a) if (home_n + a) > 0 else self.global_home_ga
            away_gf_avg = (away_gf + a * self.global_away_gf) / (away_n + a) if (away_n + a) > 0 else self.global_away_gf
            away_ga_avg = (away_ga + a * self.global_away_ga) / (away_n + a) if (away_n + a) > 0 else self.global_away_ga
            self.team_stats_[str(t)] = {
                'home_draw_rate': p_home_draw,
                'away_draw_rate': p_away_draw,
                'overall_draw_rate': p_overall_draw,
                'home_gf_avg': home_gf_avg, 'home_ga_avg': home_ga_avg,
                'away_gf_avg': away_gf_avg, 'away_ga_avg': away_ga_avg,
            }
        return self
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        home_ids = df['home_team_id'].astype(str)
        away_ids = df['away_team_id'].astype(str)
        def get_stat(team_id: str, key: str, default: float):
            return self.team_stats_.get(team_id, {}).get(key, default)
        X['home_team_draw_rate_home'] = [get_stat(t, 'home_draw_rate', self.global_draw_rate) for t in home_ids]
        X['home_team_draw_rate_overall'] = [get_stat(t, 'overall_draw_rate', self.global_draw_rate) for t in home_ids]
        X['away_team_draw_rate_away'] = [get_stat(t, 'away_draw_rate', self.global_draw_rate) for t in away_ids]
        X['away_team_draw_rate_overall'] = [get_stat(t, 'overall_draw_rate', self.global_draw_rate) for t in away_ids]
        X['draw_rate_diff'] = X['home_team_draw_rate_home'] - X['away_team_draw_rate_away']
        X['home_team_home_gf_avg'] = [get_stat(t, 'home_gf_avg', self.global_home_gf) for t in home_ids]
        X['home_team_home_ga_avg'] = [get_stat(t, 'home_ga_avg', self.global_home_ga) for t in home_ids]
        X['away_team_away_gf_avg'] = [get_stat(t, 'away_gf_avg', self.global_away_gf) for t in away_ids]
        X['away_team_away_ga_avg'] = [get_stat(t, 'away_ga_avg', self.global_away_ga) for t in away_ids]
        self.features_ = list(X.columns)
        return X.fillna(0.0)
    def get_feature_names_out(self):
        return self.features_

# سكورات التقارب
def compute_closeness_score(X: pd.DataFrame) -> np.ndarray:
    return (0.5 * X['closeness_elo'].values
            + 0.25 * X['closeness_points'].values
            + 0.25 * X['closeness_gd'].values)

# NEW: تقدير λ_home و λ_away ثم p_draw بشكيلام
FORM_N = 5.0
def estimate_lambdas(df_block: pd.DataFrame, X_block: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    h_att_form = pd.to_numeric(df_block['home_form_gs'], errors='coerce').fillna(0).values / FORM_N
    a_att_form = pd.to_numeric(df_block['away_form_gs'], errors='coerce').fillna(0).values / FORM_N
    h_def_form = pd.to_numeric(df_block['home_form_gc'], errors='coerce').fillna(0).values / FORM_N
    a_def_form = pd.to_numeric(df_block['away_form_gc'], errors='coerce').fillna(0).values / FORM_N
    h_gf_avg = X_block['home_team_home_gf_avg'].values if 'home_team_home_gf_avg' in X_block.columns else np.full(len(df_block), 1.4)
    h_ga_avg = X_block['home_team_home_ga_avg'].values if 'home_team_home_ga_avg' in X_block.columns else np.full(len(df_block), 1.1)
    a_gf_avg = X_block['away_team_away_gf_avg'].values if 'away_team_away_gf_avg' in X_block.columns else np.full(len(df_block), 1.1)
    a_ga_avg = X_block['away_team_away_ga_avg'].values if 'away_team_away_ga_avg' in X_block.columns else np.full(len(df_block), 1.4)
    lam_h = 0.5 * h_att_form + 0.25 * h_gf_avg + 0.25 * a_ga_avg
    lam_a = 0.5 * a_att_form + 0.25 * a_gf_avg + 0.25 * h_ga_avg
    elo_diff = X_block['elo_diff'].values if 'elo_diff' in X_block.columns else np.zeros_like(lam_h)
    adj = np.tanh(elo_diff / 400.0)
    lam_h *= (1.0 + 0.15 * adj)
    lam_a *= (1.0 - 0.15 * adj)
    lam_h = np.clip(lam_h, 0.05, 4.0)
    lam_a = np.clip(lam_a, 0.05, 4.0)
    return lam_h, lam_a

def skellam_draw_prob(lam_h: np.ndarray, lam_a: np.ndarray) -> np.ndarray:
    z = 2.0 * np.sqrt(lam_h * lam_a)
    return np.exp(-(lam_h + lam_a)) * bessel_i0(z)

# قرار بالتعادل باستخدام العتبة والبوابة
def predict_with_gating(P: np.ndarray, X: pd.DataFrame, tau: float, gamma: float) -> np.ndarray:
    s = compute_closeness_score(X)
    cond_draw = (P[:, 1] >= tau) & (s >= gamma)
    pred = np.where(cond_draw, 1, np.where(P[:, 0] >= P[:, 2], 0, 2))
    return pred

def eval_from_preds(y_true, y_pred, name=""):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro')
    if name:
        print(f"{name} -> Acc: {acc:.2%} | BalAcc: {bal_acc:.2%} | Macro-F1: {f1m:.2%}")
    return acc, bal_acc, f1m

def sample_params(rng):
    return {
        'learning_rate': float(rng.uniform(0.03, 0.12)),
        'max_depth': int(rng.integers(3, 7)),
        'min_child_weight': float(rng.choice([1.0, 2.0, 3.0, 5.0])),
        'subsample': float(rng.uniform(0.7, 1.0)),
        'colsample_bytree': float(rng.uniform(0.6, 1.0)),
        'gamma': float(rng.choice([0.0, 0.1, 0.2, 0.4])),
        'reg_alpha': float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        'reg_lambda': float(rng.choice([0.5, 1.0, 2.0, 5.0])),
        'n_estimators': N_ESTIMATORS_RS,
    }

def tune_gating(P_val: np.ndarray, X_val: pd.DataFrame, y_val: pd.Series,
                acc_floor: float,
                tau_grid=np.linspace(0.32, 0.48, 17),
                gamma_grid=np.linspace(0.50, 0.70, 5)) -> Dict:
    best = None
    for tau in tau_grid:
        for gamma in gamma_grid:
            y_pred = predict_with_gating(P_val, X_val, tau, gamma)
            acc = accuracy_score(y_val, y_pred)
            f1m = f1_score(y_val, y_pred, average='macro')
            bal = balanced_accuracy_score(y_val, y_pred)
            if acc >= acc_floor:
                if (best is None) or (f1m > best['f1m']) or (abs(f1m - best['f1m']) < 1e-6 and acc > best['acc']):
                    best = {'tau': float(tau), 'gamma': float(gamma), 'acc': acc, 'bal_acc': bal, 'f1m': f1m}
    if best is None:
        # fallback: أفضل F1 ماكرو حتى لو أقل من acc_floor
        for tau in tau_grid:
            for gamma in gamma_grid:
                y_pred = predict_with_gating(P_val, X_val, tau, gamma)
                acc = accuracy_score(y_val, y_pred)
                f1m = f1_score(y_val, y_pred, average='macro')
                bal = balanced_accuracy_score(y_val, y_pred)
                if (best is None) or (f1m > best['f1m']) or (abs(f1m - best['f1m']) < 1e-6 and acc > best['acc']):
                    best = {'tau': float(tau), 'gamma': float(gamma), 'acc': acc, 'bal_acc': bal, 'f1m': f1m}
    return best

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def train():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_CSV_FILE)

    required_cols = [
        'date','competition',
        'home_team_id','away_team_id',
        'home_team_elo','away_team_elo',
        'home_form_points','home_form_gs','home_form_gc',
        'away_form_points','away_form_gs','away_form_gc',
        'result_ft_home_goals','result_ft_away_goals'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()
    df = df[(df['home_form_points'] > 0) | (df['away_form_points'] > 0)]
    if 'match_id' in df.columns:
        df = df.drop_duplicates(subset='match_id')
    if df.empty:
        print("Dataset is empty after cleaning. Cannot train model.")
        return

    # الهدف: 0 Home, 1 Draw, 2 Away
    df['target'] = np.select(
        [
            df['result_ft_home_goals'] > df['result_ft_away_goals'],
            df['result_ft_home_goals'] == df['result_ft_away_goals'],
        ],
        [0, 1],
        default=2
    ).astype(int)

    # تقسيم زمني
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df_train, df_val, df_test = time_split_by_day(df, train_frac=0.70, val_frac=0.15)

    y_train = df_train['target']
    y_val = df_val['target']
    y_test = df_test['target']

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"Train last date: {df_train['date'].max().date()} | "
          f"Val last date: {df_val['date'].max().date()} | "
          f"Test start date: {df_test['date'].min().date()}")

    # توليد الميزات
    fb = FeatureBuilder(comp_alpha=30.0).fit(df_train, y_train)
    tp = TeamPriors(alpha=50.0).fit(df_train)

    def build_X(df_block):
        return pd.concat([fb.transform(df_block), tp.transform(df_block)], axis=1)

    X_train = build_X(df_train)
    X_val = build_X(df_val)
    X_test = build_X(df_test)

    # --------- Baseline XGBoost (multiclass) + معايرة ---------
    rng = np.random.default_rng(RANDOM_STATE)
    best_base_model = None
    best_base_cfg = None
    best_base_ll = np.inf
    best_ts = None

    # أوزان متوازنة للفئات
    classes = np.array([0,1,2])
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    cw_map = {c:w for c,w in zip(classes, class_weights)}
    sw_train_base = y_train.map(cw_map).values

    for i in range(N_TRIALS_BASE):
        params = sample_params(rng)
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric=['mlogloss','merror'],
            tree_method='hist',
            random_state=RANDOM_STATE + i,
            n_jobs=-1,
            **params
        )
        xgb_fit_with_es(model, X_train, y_train, sw_train_base, X_val, y_val, es_rounds=150)
        proba_val = model.predict_proba(X_val)
        ts = TemperatureScaler().fit(proba_val, y_val)
        P_val_cal = ts.transform(proba_val)
        ll = log_loss(y_val, P_val_cal, labels=[0,1,2])
        if ll < best_base_ll:
            best_base_ll = ll
            best_base_model = model
            best_base_cfg = params
            best_ts = ts

    # baseline acc على Val بالـ argmax
    P_val_cal_base = best_ts.transform(best_base_model.predict_proba(X_val))
    base_acc_val = accuracy_score(y_val, P_val_cal_base.argmax(axis=1))

    # --------- نموذج ثنائي (Draw vs Not-Draw) ---------
    y_train_draw = (y_train == 1).astype(int)
    y_val_draw = (y_val == 1).astype(int)
    cw_bin = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train_draw)
    sw_train_draw = np.where(y_train_draw.values == 1, cw_bin[1], cw_bin[0])

    draw_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=RANDOM_STATE + 999,
        n_estimators=800, max_depth=4, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, n_jobs=-1
    )
    xgb_fit_with_es(draw_clf, X_train, y_train_draw, sw_train_draw, X_val, y_val_draw, es_rounds=150)

    # --------- ميتا-ليرنر (Stacking) ---------
    # ميزات الميتا = [base logits(3), draw_logit(1), p_skellam_logit(1), closeness(1), comp_draw_prior(1)] = 7
    EPS = 1e-12
    # Train
    P_tr_cal = best_ts.transform(best_base_model.predict_proba(X_train))
    draw_tr = draw_clf.predict_proba(X_train)[:,1]
    lam_h_tr, lam_a_tr = estimate_lambdas(df_train, X_train)
    p_sk_tr = skellam_draw_prob(lam_h_tr, lam_a_tr)
    closeness_tr = compute_closeness_score(X_train)
    comp_draw_prior_tr = X_train['comp_draw_prior'].values
    Z_train = np.column_stack([
        np.log(np.clip(P_tr_cal, EPS, 1-EPS)),
        logit(draw_tr),
        logit(p_sk_tr),
        closeness_tr,
        comp_draw_prior_tr
    ])

    # Val
    P_val_cal = best_ts.transform(best_base_model.predict_proba(X_val))
    draw_val = draw_clf.predict_proba(X_val)[:,1]
    lam_h_val, lam_a_val = estimate_lambdas(df_val, X_val)
    p_sk_val = skellam_draw_prob(lam_h_val, lam_a_val)
    closeness_val = compute_closeness_score(X_val)
    comp_draw_prior_val = X_val['comp_draw_prior'].values
    Z_val = np.column_stack([
        np.log(np.clip(P_val_cal, EPS, 1-EPS)),
        logit(draw_val),
        logit(p_sk_val),
        closeness_val,
        comp_draw_prior_val
    ])

    meta = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                              C=0.8, max_iter=300, class_weight='balanced', n_jobs=-1)
    meta.fit(Z_train, y_train)
    P_meta_val = meta.predict_proba(Z_val)
    meta_acc_val = accuracy_score(y_val, P_meta_val.argmax(axis=1))

    # acc_floor: لا نهبط عن الأفضل بين baseline/meta بأكثر من 1%
    acc_floor = max(base_acc_val, meta_acc_val) - ACC_DROP_ALLOWED

    # ضبط بوابة التعادل على مخرجات الميتا
    gating = tune_gating(P_meta_val, X_val, y_val, acc_floor)
    print(f"Val baseline argmax Acc: {base_acc_val:.2%} | Val meta argmax Acc: {meta_acc_val:.2%} | acc_floor: {acc_floor:.2%}")
    print(f"Gating tuned on Val -> tau={gating['tau']:.3f}, gamma={gating['gamma']:.2f}, "
          f"Acc={gating['acc']:.2%}, Macro-F1={gating['f1m']:.2%}")

    # --------- تقييم على الاختبار ---------
    P_te_cal = best_ts.transform(best_base_model.predict_proba(X_test))
    draw_te = draw_clf.predict_proba(X_test)[:,1]
    lam_h_te, lam_a_te = estimate_lambdas(df_test, X_test)
    p_sk_te = skellam_draw_prob(lam_h_te, lam_a_te)
    closeness_te = compute_closeness_score(X_test)
    comp_draw_prior_te = X_test['comp_draw_prior'].values
    Z_test = np.column_stack([
        np.log(np.clip(P_te_cal, EPS, 1-EPS)),
        logit(draw_te),
        logit(p_sk_te),
        closeness_te,
        comp_draw_prior_te
    ])

    P_meta_test = meta.predict_proba(Z_test)
    y_pred_test = predict_with_gating(P_meta_test, X_test, gating['tau'], gating['gamma'])
    acc, bal_acc, f1m = eval_from_preds(y_test, y_pred_test, name="Test (Meta-Stacking + gating)")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=['Home Win','Draw','Away Win'], zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred_test, labels=[0,1,2]))

    # حفظ كل شيء
    artifact = {
        'base_model': best_base_model,
        'draw_model': draw_clf,
        'meta_model': meta,
        'feature_builder': fb,
        'team_priors': tp,
        'temp_scaler_T': float(best_ts.T),
        'decision': {'tau_draw': float(gating['tau']), 'gamma_close': float(gating['gamma'])},
        'features': list(X_train.columns),
        'label_map': {0:'Home Win', 1:'Draw', 2:'Away Win'},
        'meta_info': {
            'xgboost_version': xgb.__version__,
            'train_rows': int(len(df_train)),
            'val_rows': int(len(df_val)),
            'test_rows': int(len(df_test)),
            'train_last_date': str(df_train['date'].max().date()),
            'test_start_date': str(df_test['date'].min().date()),
            'base_params': best_base_cfg,
            'best_iteration_base': get_best_iteration(best_base_model),
            'es_mode': ES_MODE,
            'acc_floor': float(acc_floor)
        }
    }
    print(f"\nSaving trained model to '{MODEL_OUTPUT_FILE}'...")
    joblib.dump(artifact, MODEL_OUTPUT_FILE)
    print("✅ Model saved successfully.")

def predict_match(example: dict, model_path: str = MODEL_OUTPUT_FILE, return_pred: bool = True) -> Dict[str, float]:
    """
    example يحتوي على الأقل:
      ['date','competition',
       'home_team_id','away_team_id',
       'home_team_elo','away_team_elo',
       'home_form_points','home_form_gs','home_form_gc',
       'away_form_points','away_form_gs','away_form_gc']
    اختياري: 'home_team_league_pos','away_team_league_pos'
    """
    art = joblib.load(model_path)
    base = art['base_model']
    draw_m = art['draw_model']
    meta = art['meta_model']
    fb = art['feature_builder']
    tp = art['team_priors']
    T = art.get('temp_scaler_T', 1.0)
    tau = art['decision']['tau_draw']
    gamma = art['decision']['gamma_close']
    label_map = art['label_map']

    df = pd.DataFrame([example])
    for opt in ['home_team_league_pos', 'away_team_league_pos']:
        if opt not in df.columns:
            df[opt] = np.nan

    X = pd.concat([fb.transform(df), tp.transform(df)], axis=1)

    # baseline + TS
    proba_base = base.predict_proba(X)
    ts = TemperatureScaler(T=T)
    P_base = ts.transform(proba_base)

    # draw binary + skellam + سياقي
    p_draw_bin = draw_m.predict_proba(X)[:,1]
    lam_h, lam_a = estimate_lambdas(df, X)
    p_sk = skellam_draw_prob(lam_h, lam_a)
    closeness = compute_closeness_score(X)
    comp_draw_prior = X['comp_draw_prior'].values

    EPS = 1e-12
    Z = np.column_stack([
        np.log(np.clip(P_base, EPS, 1-EPS)),
        logit(p_draw_bin),
        logit(p_sk),
        closeness,
        comp_draw_prior
    ])

    P = meta.predict_proba(Z)
    y_pred = predict_with_gating(P, X, tau, gamma)[0]

    out = {label_map[i]: float(P[0, i]) for i in range(3)}
    if return_pred:
        out['pred'] = label_map[int(y_pred)]
    return out

if __name__ == "__main__":
    train()
