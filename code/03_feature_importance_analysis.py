"""
feature_importance_top3.py
===========================
Permutation feature importance for the three leakage-free models compared
in Chapter 4, Section 4.1.8 ("Comparing the Top Three Models"):
  1. Random Forest  (native impurity importance + permutation)
  2. Ridge           (standardized |coefficient| + permutation)
  3. N-BEATS v2       (permutation only -- no native importance measure)

All three are trained on the same 19-feature set used in the bootstrap
validation (Table 4.5): CAL_FEATS + HIST_FEATS, with the two cross-sectional
leakage columns (pct_rank_global, zscore_municipality) excluded, exactly as
in top4_bootstrap.py.

Permutation importance: for each feature, shuffle its column across the
test period (breaking its relationship with the target) and measure the
resulting increase in test-set MAPE relative to the unpermuted baseline,
averaged over N_REPEATS shuffles. Importance is then expressed as each
feature's share of the total MAPE increase across all 19 features, so
Random Forest, Ridge, and N-BEATS are compared on identical footing
regardless of what kind of model each one is. Native importances
(RF impurity, Ridge |coefficient|) are also saved for cross-reference,
but the permutation numbers are what should populate Table
tab:feature_importance_top3 in the thesis.

For N-BEATS, whose input is a sliding 30-day window rather than one row
per prediction, permutation is applied to the val+test feature matrix
before the rolling-window inference loop replays it. Only the val+test
rows are shuffled; the training-period tail used to seed the first
window is left intact, since it is fully flushed out of the window
after SEQ_LEN steps and its effect on test-period MAPE is negligible.

Usage:
    python feature_importance_top3.py

Output (under results/feature_importance_top3/):
    permutation_importance_by_city.csv   one row per (model, city, feature)
    permutation_importance_summary.csv   mean %, one row per (model, feature)
    rf_native_importance.csv             RF impurity importance, 19-feature set
    ridge_native_importance.csv          Ridge |standardized coefficient|
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False

# ── Config ────────────────────────────────────────────────────────────────────
BASE_PATH  = Path(__file__).resolve().parents[1]
DATA_FILE  = BASE_PATH / "results" / "data" / "municipality_daily_consumption.csv"
OUT_DIR    = BASE_PATH / "results" / "feature_importance_top3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL   = "date"
TARGET_COL = "avg_kwh"
GROUP_COL  = "municipality"
GROUPS     = ["Vitoria-Gasteiz", "Donostia/San Sebastian", "Pamplona/Iruna"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED       = 42
N_REPEATS  = 10     # shuffles per feature for permutation importance

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
SEQ_LEN    = 30
EPOCHS     = 80
BATCH_SIZE = 64
PATIENCE   = 15

CAL_FEATS = [
    "is_weekend", "is_holiday_es", "is_bridge_day",
    "sin_dow", "cos_dow", "sin_month", "cos_month", "sin_week", "cos_week",
]
HIST_FEATS = [
    "lag_1d", "lag_7d", "lag_14d", "lag_30d",
    "roll7_mean", "roll7_std", "roll30_mean", "roll7_ratio",
    "wow_change", "dod_change",
]
ALL_FEATS = CAL_FEATS + HIST_FEATS   # 19 features, same order as Table 4.2 / 4.5

# ── Feature engineering (identical to top4_bootstrap.py) ──────────────────────
def build_features(df_city: pd.DataFrame) -> pd.DataFrame:
    df = df_city.sort_values(DATE_COL).copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["day_of_week"]  = df[DATE_COL].dt.dayofweek
    df["month"]        = df[DATE_COL].dt.month
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["sin_dow"]      = np.sin(2 * np.pi * df["day_of_week"]  / 7)
    df["cos_dow"]      = np.cos(2 * np.pi * df["day_of_week"]  / 7)
    df["sin_month"]    = np.sin(2 * np.pi * df["month"]        / 12)
    df["cos_month"]    = np.cos(2 * np.pi * df["month"]        / 12)
    df["sin_week"]     = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"]     = np.cos(2 * np.pi * df["week_of_year"] / 52)

    if _HAS_HOLIDAYS:
        years = sorted(df[DATE_COL].dt.year.unique())
        es_hols = set(pd.to_datetime(
            list(_holidays_lib.country_holidays("ES", years=years).keys())))
        df["is_holiday_es"] = df[DATE_COL].isin(es_hols).astype(int)
        nd  = df[DATE_COL] + pd.Timedelta(days=1)
        pd_ = df[DATE_COL] - pd.Timedelta(days=1)
        df["is_bridge_day"] = (
            ((df["day_of_week"] == 0) & nd.isin(es_hols)) |
            ((df["day_of_week"] == 4) & pd_.isin(es_hols))
        ).astype(int)
    else:
        df["is_holiday_es"] = 0
        df["is_bridge_day"] = 0

    g = df[TARGET_COL]
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}d"] = g.shift(lag)
    df["roll7_mean"]  = g.shift(1).rolling(7,  min_periods=1).mean()
    df["roll7_std"]   = g.shift(1).rolling(7,  min_periods=2).std().fillna(0)
    df["roll30_mean"] = g.shift(1).rolling(30, min_periods=1).mean()
    df["roll7_ratio"] = df["lag_1d"] / df["roll7_mean"].replace(0, np.nan)
    df["wow_change"]  = (df["lag_1d"] - df["lag_7d"])  / df["lag_7d"].replace(0, np.nan)
    df["dod_change"]  = (df["lag_1d"] - g.shift(2))    / g.shift(2).replace(0, np.nan)
    return df.fillna(0).set_index(DATE_COL)

def split_df(df):
    n    = len(df)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr+n_va], df.iloc[n_tr+n_va:]

def calc_mape(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)

# ── N-BEATS model (identical architecture to top4_bootstrap.py) ───────────────
class WinDS(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.s = seq_len
    def __len__(self):  return len(self.y) - self.s
    def __getitem__(self, i):
        return self.X[i:i+self.s], self.y[i+self.s]

class NBEATSBlock(nn.Module):
    def __init__(self, in_sz, cov_sz, hidden=64, dropout=0.2):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(in_sz + cov_sz, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),          nn.ReLU(), nn.Dropout(dropout),
        )
        self.back = nn.Linear(hidden, in_sz)
        self.fore = nn.Linear(hidden, 1)
    def forward(self, x, cov):
        h = self.net(torch.cat([x, cov], dim=-1))
        return self.back(h), self.fore(h)

class NBEATSModel(nn.Module):
    def __init__(self, seq_len, cov_sz, n_blocks=3, hidden=64, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [NBEATSBlock(seq_len, cov_sz, hidden, dropout) for _ in range(n_blocks)])
    def forward(self, x):
        # x: (B, seq_len, n_feats) -- col 0 = target, cols 1..N = features
        res = x[:, :, 0]
        cov = x[:, -1, 1:]
        fc  = torch.zeros(x.size(0), 1, device=x.device)
        for blk in self.blocks:
            back, f = blk(res, cov)
            res = res - back
            fc  = fc + f
        return fc

def _train_nbeats(model, tr_dl, va_dl):
    model = model.to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_val, bad = None, np.inf, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = [loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() for xb, yb in va_dl]
        v = float(np.mean(vl))
        if v < best_val:
            best_val = v; bad = 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model

def fit_nbeats(train, val, test, seed):
    """Train once. Returns everything needed to re-run rolling inference
    on a (possibly permuted) val+test feature matrix without retraining."""
    torch.manual_seed(seed); np.random.seed(seed)
    feats = [f for f in ALL_FEATS if f in train.columns]
    cols  = [TARGET_COL] + feats

    xs = StandardScaler(); ys = StandardScaler()
    X_tr = xs.fit_transform(train[cols].values)
    X_va = xs.transform(val[cols].values)
    X_te = xs.transform(test[cols].values)
    y_tr = ys.fit_transform(train[[TARGET_COL]]).ravel()
    y_va = ys.transform(val[[TARGET_COL]]).ravel()

    tr_dl = DataLoader(WinDS(X_tr, y_tr, SEQ_LEN), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(WinDS(X_va, y_va, SEQ_LEN), BATCH_SIZE)

    model = NBEATSModel(seq_len=SEQ_LEN, cov_sz=len(feats), n_blocks=3, hidden=64, dropout=0.2)
    model = _train_nbeats(model, tr_dl, va_dl)

    X_ctx = np.vstack([X_va, X_te])
    return model, ys, X_tr, X_ctx, feats, len(X_va)

def predict_nbeats(model, ys, X_tr, X_ctx, seq_len, n_val):
    """Rolling-window inference. X_ctx rows 0..n_val-1 are validation
    (used only to seed the buffer), rows n_val.. are the scored test period."""
    preds = []
    buf = list(X_tr[-seq_len:])
    model.eval()
    with torch.no_grad():
        for i in range(len(X_ctx)):
            win = np.array(buf[-seq_len:])
            xb  = torch.tensor(win.reshape(1, seq_len, -1), dtype=torch.float32).to(DEVICE)
            preds.append(model(xb).item())
            buf.append(X_ctx[i])
    return ys.inverse_transform(np.array(preds[n_val:]).reshape(-1, 1)).ravel().clip(0)

# ── Ridge / Random Forest ──────────────────────────────────────────────────────
def fit_ridge(train, val):
    tv = pd.concat([train, val])
    feats = [f for f in ALL_FEATS if f in tv.columns]
    sc = StandardScaler()
    X_tr = sc.fit_transform(tv[feats].values)
    model = Ridge(alpha=1.0).fit(X_tr, tv[TARGET_COL].values)
    return model, sc, feats

def fit_rf(train, val, seed):
    tv = pd.concat([train, val])
    feats = [f for f in ALL_FEATS if f in tv.columns]
    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=seed, n_jobs=-1)
    model.fit(tv[feats].values, tv[TARGET_COL].values)
    return model, feats

# ── Permutation importance ─────────────────────────────────────────────────────
def permutation_importance_tabular(predict_fn, X_test, y_test, feats, n_repeats, rng):
    baseline_mape = calc_mape(y_test, predict_fn(X_test))
    rows = []
    for j, feat in enumerate(feats):
        increases = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            rng.shuffle(Xp[:, j])
            increases.append(calc_mape(y_test, predict_fn(Xp)) - baseline_mape)
        rows.append({"feature": feat, "mape_increase": float(np.mean(increases))})
    return baseline_mape, rows

def permutation_importance_nbeats(model, ys, X_tr, X_ctx, feats, seq_len, n_val,
                                   y_test, n_repeats, rng):
    def predict_fn(X_ctx_variant):
        return predict_nbeats(model, ys, X_tr, X_ctx_variant, seq_len, n_val)

    baseline_mape = calc_mape(y_test, predict_fn(X_ctx))
    rows = []
    for j, feat in enumerate(feats):
        col = j + 1   # +1: column 0 of X_ctx is the target channel, not a feature
        increases = []
        for _ in range(n_repeats):
            Xp = X_ctx.copy()
            rng.shuffle(Xp[:, col])
            increases.append(calc_mape(y_test, predict_fn(Xp)) - baseline_mape)
        rows.append({"feature": feat, "mape_increase": float(np.mean(increases))})
    return baseline_mape, rows

def to_pct(rows):
    """Negative/near-zero increases (noise) floor at 0 before normalising to %."""
    total = sum(max(r["mape_increase"], 0.0) for r in rows)
    for r in rows:
        r["importance_pct"] = 100.0 * max(r["mape_increase"], 0.0) / total if total > 0 else 0.0
    return rows

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    rng = np.random.default_rng(SEED)

    df_raw = pd.read_csv(DATA_FILE, parse_dates=[DATE_COL])
    df_raw = df_raw[df_raw[GROUP_COL].isin(GROUPS)]

    perm_rows, rf_native_rows, ridge_native_rows = [], [], []

    for city in GROUPS:
        print(f"\n{'='*60}\n{city}\n{'='*60}")
        df = build_features(df_raw[df_raw[GROUP_COL] == city].copy())
        train, val, test = split_df(df)

        # ── Random Forest ────────────────────────────────────────────────────
        print("  Random Forest: fitting + native importance...")
        rf_model, feats = fit_rf(train, val, SEED)
        for feat, imp in zip(feats, rf_model.feature_importances_):
            rf_native_rows.append({"city": city, "feature": feat, "importance": float(imp)})

        print("  Random Forest: permutation importance...")
        X_te_rf = test[feats].values
        y_te    = test[TARGET_COL].values
        _, rows = permutation_importance_tabular(
            lambda X: rf_model.predict(X).clip(0), X_te_rf, y_te, feats, N_REPEATS, rng)
        for r in to_pct(rows):
            perm_rows.append({"model": "Random Forest", "city": city, **r})

        # ── Ridge ─────────────────────────────────────────────────────────────
        print("  Ridge: fitting + native |coefficient|...")
        ridge_model, sc, feats_r = fit_ridge(train, val)
        for feat, coef in zip(feats_r, ridge_model.coef_):
            ridge_native_rows.append({"city": city, "feature": feat, "abs_std_coef": float(abs(coef))})

        print("  Ridge: permutation importance...")
        X_te_ridge = sc.transform(test[feats_r].values)
        _, rows = permutation_importance_tabular(
            lambda X: ridge_model.predict(X).clip(0), X_te_ridge, y_te, feats_r, N_REPEATS, rng)
        for r in to_pct(rows):
            perm_rows.append({"model": "Ridge", "city": city, **r})

        # ── N-BEATS v2 ────────────────────────────────────────────────────────
        print("  N-BEATS v2: training (this is the slow part)...")
        nb_model, ys, X_tr, X_ctx, feats_nb, n_val = fit_nbeats(train, val, test, SEED)

        print("  N-BEATS v2: permutation importance (rolling re-inference per feature)...")
        _, rows = permutation_importance_nbeats(
            nb_model, ys, X_tr, X_ctx, feats_nb, SEQ_LEN, n_val, y_te, N_REPEATS, rng)
        for r in to_pct(rows):
            perm_rows.append({"model": "N-BEATS v2", "city": city, **r})

    # ── Save per-city results ────────────────────────────────────────────────
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(OUT_DIR / "permutation_importance_by_city.csv", index=False)

    pd.DataFrame(rf_native_rows).to_csv(OUT_DIR / "rf_native_importance.csv", index=False)
    pd.DataFrame(ridge_native_rows).to_csv(OUT_DIR / "ridge_native_importance.csv", index=False)

    # ── Summary: mean %% across the three cities ─────────────────────────────
    summary = (
        perm_df.groupby(["model", "feature"])["importance_pct"]
        .mean().reset_index()
        .sort_values(["model", "importance_pct"], ascending=[True, False])
    )
    summary.to_csv(OUT_DIR / "permutation_importance_summary.csv", index=False)

    print(f"\nSaved results to: {OUT_DIR}")
    print("\nMean permutation importance (%) across cities:")
    for model in ["Random Forest", "Ridge", "N-BEATS v2"]:
        print(f"\n{model}")
        print(summary[summary["model"] == model].to_string(index=False))


if __name__ == "__main__":
    main()
