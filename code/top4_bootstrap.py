"""
top4_bootstrap.py
=================
Bootstrap validation for the 4 best with-FE models:
  1. N-BEATS v2 (covariate-conditioned)
  2. LSTM v2     (dual-stream)
  3. Ridge       (calendar + history features)
  4. Random Forest (calendar + history features)

Runs each model 100 times with seeds 0..99.
After all runs, drops the 5 runs with the lowest aggregate MAPE (lucky)
and the 5 runs with the highest aggregate MAPE (unlucky) per model+city,
then reports the trimmed mean over the remaining 90 runs.

Restartable: completed runs are saved immediately and skipped on re-entry.

Usage:
    python top4_bootstrap.py

Output:
    bootstrap_top4/run_XXXX.csv         (per-run raw results)
    bootstrap_top4/summary_trimmed.csv  (trimmed-mean MAPE, MAE, RMSE)
"""

import warnings, time, gc
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
BASE_PATH  = Path(r"C:\Users\GONCA\Desktop\Iscte\MCD\Theses")
DATA_FILE  = BASE_PATH / "results" / "data" / "municipality_daily_consumption.csv"
if not DATA_FILE.exists():
    DATA_FILE = BASE_PATH / "municipality_daily_consumption.csv"
OUT_DIR    = BASE_PATH / "bootstrap_top4"
OUT_DIR.mkdir(exist_ok=True)

DATE_COL   = "date"
TARGET_COL = "avg_kwh"
GROUP_COL  = "municipality"
GROUPS     = ["Vitoria-Gasteiz", "Donostia/San Sebastian", "Pamplona/Iruna"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_RUNS     = 100
TRIM       = 5          # drop TRIM lowest + TRIM highest MAPE per model+city

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
ALL_FEATS = CAL_FEATS + HIST_FEATS   # 19 features

MODEL_NAMES = ["N-BEATS v2", "LSTM v2", "Ridge", "Random Forest"]

# ── Feature engineering ───────────────────────────────────────────────────────
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

# ── Metrics ───────────────────────────────────────────────────────────────────
def calc_metrics(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    mae  = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    mask = a != 0
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)
    return mae, rmse, mape

# ── Datasets ──────────────────────────────────────────────────────────────────
class WinDS(Dataset):
    """Single-stream window dataset for N-BEATS v2."""
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.s = seq_len
    def __len__(self):  return len(self.y) - self.s
    def __getitem__(self, i):
        return self.X[i:i+self.s], self.y[i+self.s]

class DualDS(Dataset):
    """Dual-stream window dataset for LSTM v2."""
    def __init__(self, Xs, Xc, y, seq_len):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.Xc = torch.tensor(Xc, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.float32).view(-1, 1)
        self.s  = seq_len
    def __len__(self):  return len(self.y) - self.s
    def __getitem__(self, i):
        return self.Xs[i:i+self.s], self.Xc[i+self.s], self.y[i+self.s]

# ── Models ────────────────────────────────────────────────────────────────────
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
        # x: (B, seq_len, n_feats) — col 0 = target, cols 1..N = features
        res = x[:, :, 0]            # target channel (B, seq_len)
        cov = x[:, -1, 1:]          # last-step features as covariate context
        fc  = torch.zeros(x.size(0), 1, device=x.device)
        for blk in self.blocks:
            back, f = blk(res, cov)
            res = res - back
            fc  = fc + f
        return fc

class DualLSTM(nn.Module):
    def __init__(self, seq_feats, cal_feats, hidden=48, dropout=0.3):
        super().__init__()
        self.lstm   = nn.LSTM(seq_feats, hidden, num_layers=1, batch_first=True)
        self.static = nn.Sequential(nn.Linear(cal_feats, 16), nn.ReLU(), nn.LayerNorm(16))
        self.head   = nn.Linear(hidden + 16, 1)
        self.drop   = nn.Dropout(dropout)
    def forward(self, xs, xc):
        _, (h, _) = self.lstm(xs)
        return self.head(torch.cat([self.drop(h.squeeze(0)), self.static(xc)], dim=1))

# ── Generic train loop ────────────────────────────────────────────────────────
def _train(model, tr_dl, va_dl, is_dual=False):
    model = model.to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_val, bad = None, np.inf, 0
    for ep in range(EPOCHS):
        model.train()
        for batch in tr_dl:
            if is_dual:
                xs, xc, yb = batch
                xs, xc, yb = xs.to(DEVICE), xc.to(DEVICE), yb.to(DEVICE)
                loss = loss_fn(model(xs, xc), yb)
            else:
                xb, yb = batch
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for batch in va_dl:
                if is_dual:
                    xs, xc, yb = batch
                    vl.append(loss_fn(model(xs.to(DEVICE), xc.to(DEVICE)), yb.to(DEVICE)).item())
                else:
                    xb, yb = batch
                    vl.append(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item())
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

# ── N-BEATS v2 ────────────────────────────────────────────────────────────────
def run_nbeats(train, val, test, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    feats = [f for f in ALL_FEATS if f in train.columns]
    cols  = [TARGET_COL] + feats

    xs = StandardScaler(); ys = StandardScaler()
    X_tr = xs.fit_transform(train[cols].values)
    X_va = xs.transform(val[cols].values)
    X_te = xs.transform(test[cols].values)
    y_tr = ys.fit_transform(train[[TARGET_COL]]).ravel()
    y_va = ys.transform(val[[TARGET_COL]]).ravel()

    seq_len = SEQ_LEN
    tr_dl = DataLoader(WinDS(X_tr, y_tr, seq_len), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(WinDS(X_va, y_va, seq_len), BATCH_SIZE)

    model = NBEATSModel(seq_len=seq_len, cov_sz=len(feats), n_blocks=3, hidden=64, dropout=0.2)
    model = _train(model, tr_dl, va_dl, is_dual=False)

    # Rolling prediction over val+test
    X_ctx = np.vstack([X_va, X_te])
    preds = []
    buf   = list(X_tr[-seq_len:])
    model.eval()
    with torch.no_grad():
        for i in range(len(X_ctx)):
            win = np.array(buf[-seq_len:])
            xb  = torch.tensor(win.reshape(1, seq_len, -1), dtype=torch.float32).to(DEVICE)
            preds.append(model(xb).item())
            buf.append(X_ctx[i])

    pred = ys.inverse_transform(
        np.array(preds[len(X_va):]).reshape(-1, 1)).ravel().clip(0)
    return pd.Series(pred, index=test.index)

# ── LSTM v2 ───────────────────────────────────────────────────────────────────
def run_lstm(train, val, test, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cal = [f for f in CAL_FEATS  if f in train.columns]
    seq = [f for f in HIST_FEATS if f in train.columns]

    xss = StandardScaler(); xcs = StandardScaler(); ys = StandardScaler()
    Xs_tr = xss.fit_transform(train[seq].values)
    Xs_va = xss.transform(val[seq].values)
    Xs_te = xss.transform(test[seq].values)
    Xc_tr = xcs.fit_transform(train[cal].values)
    Xc_va = xcs.transform(val[cal].values)
    Xc_te = xcs.transform(test[cal].values)
    y_tr  = ys.fit_transform(train[[TARGET_COL]]).ravel()
    y_va  = ys.transform(val[[TARGET_COL]]).ravel()

    tr_dl = DataLoader(DualDS(Xs_tr, Xc_tr, y_tr, SEQ_LEN), BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(DualDS(Xs_va, Xc_va, y_va, SEQ_LEN), BATCH_SIZE)

    model = DualLSTM(seq_feats=len(seq), cal_feats=len(cal), hidden=48, dropout=0.3)
    model = _train(model, tr_dl, va_dl, is_dual=True)

    Xs_ctx = np.vstack([Xs_va, Xs_te])
    Xc_ctx = np.vstack([Xc_va, Xc_te])
    y_ctx  = np.concatenate([y_va, ys.transform(test[[TARGET_COL]]).ravel()])
    preds  = []
    buf    = list(Xs_tr[-SEQ_LEN:])
    model.eval()
    with torch.no_grad():
        for i in range(len(Xs_ctx)):
            xs = torch.tensor(np.array(buf[-SEQ_LEN:]).reshape(1, SEQ_LEN, -1),
                               dtype=torch.float32).to(DEVICE)
            xc = torch.tensor(Xc_ctx[i].reshape(1, -1), dtype=torch.float32).to(DEVICE)
            preds.append(model(xs, xc).item())
            buf.append(Xs_ctx[i])

    pred = ys.inverse_transform(
        np.array(preds[len(y_va):]).reshape(-1, 1)).ravel().clip(0)
    return pd.Series(pred, index=test.index)

# ── Ridge ─────────────────────────────────────────────────────────────────────
def run_ridge(train, val, test, seed):
    tv   = pd.concat([train, val])
    feats = [f for f in ALL_FEATS if f in tv.columns]
    sc   = StandardScaler()
    X_tr = sc.fit_transform(tv[feats].values)
    X_te = sc.transform(test[feats].values)
    model = Ridge(alpha=1.0)          # closed form — seed unused but kept for API uniformity
    model.fit(X_tr, tv[TARGET_COL].values)
    return pd.Series(model.predict(X_te).clip(0), index=test.index)

# ── Random Forest ─────────────────────────────────────────────────────────────
def run_rf(train, val, test, seed):
    tv   = pd.concat([train, val])
    feats = [f for f in ALL_FEATS if f in tv.columns]
    X_tr = tv[feats].values
    X_te = test[feats].values
    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=seed, n_jobs=-1)          # seed varies → real variance
    model.fit(X_tr, tv[TARGET_COL].values)
    return pd.Series(model.predict(X_te).clip(0), index=test.index)

# ── One full run ──────────────────────────────────────────────────────────────
RUNNERS = {
    "N-BEATS v2":    run_nbeats,
    "LSTM v2":       run_lstm,
    "Ridge":         run_ridge,
    "Random Forest": run_rf,
}

def run_one(seed, city_data):
    rows = []
    for city, (train, val, test) in city_data.items():
        for model_name, fn in RUNNERS.items():
            t0 = time.time()
            pred = fn(train, val, test, seed)
            mae, rmse, mape = calc_metrics(test[TARGET_COL].values, pred.values)
            rows.append({
                "seed": seed, "city": city, "model": model_name,
                "MAE": mae, "RMSE": rmse, "MAPE(%)": mape,
                "elapsed_s": round(time.time() - t0, 1),
            })
            print(f"  seed={seed:03d}  {city:<28}  {model_name:<18}  MAPE={mape:.2f}%  ({time.time()-t0:.1f}s)")
    return pd.DataFrame(rows)

# ── Summary: trimmed mean ─────────────────────────────────────────────────────
def compute_trimmed_summary(all_runs: pd.DataFrame, trim: int) -> pd.DataFrame:
    """
    For each (city, model): sort 100 MAPE values, drop the `trim` lowest
    and `trim` highest, average the remaining 100 - 2*trim values.
    """
    records = []
    for (city, model), grp in all_runs.groupby(["city", "model"]):
        grp_sorted = grp.sort_values("MAPE(%)")
        kept = grp_sorted.iloc[trim : len(grp_sorted) - trim]
        records.append({
            "city":   city,
            "model":  model,
            "n_runs": len(grp_sorted),
            "n_kept": len(kept),
            "MAPE_mean":   round(kept["MAPE(%)"].mean(), 3),
            "MAPE_std":    round(kept["MAPE(%)"].std(),  3),
            "MAPE_min":    round(kept["MAPE(%)"].min(),  3),
            "MAPE_max":    round(kept["MAPE(%)"].max(),  3),
            "MAE_mean":    round(kept["MAE"].mean(),     4),
            "RMSE_mean":   round(kept["RMSE"].mean(),    4),
        })
    return pd.DataFrame(records).sort_values(["city", "MAPE_mean"])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")
    print(f"Runs:   {N_RUNS}  |  Trim: {TRIM} lowest + {TRIM} highest\n")

    # ── Load & pre-process data once ─────────────────────────────────────────
    print("Building features...")
    df_raw = pd.read_csv(DATA_FILE, parse_dates=[DATE_COL])
    df_raw = df_raw[df_raw[GROUP_COL].isin(GROUPS)]

    city_data = {}
    for city in GROUPS:
        df = build_features(df_raw[df_raw[GROUP_COL] == city].copy())
        tr, va, te = split_df(df)
        city_data[city] = (tr, va, te)
        print(f"  {city}: train={len(tr)}  val={len(va)}  test={len(te)}")
    print()

    # ── Find completed runs ───────────────────────────────────────────────────
    done_seeds = set()
    for f in OUT_DIR.glob("run_*.csv"):
        try:
            s = int(f.stem.split("_")[1])
            done_seeds.add(s)
        except ValueError:
            pass
    if done_seeds:
        print(f"Skipping {len(done_seeds)} already-completed seeds: {sorted(done_seeds)[:10]}{'...' if len(done_seeds)>10 else ''}\n")

    # ── Run seeds ─────────────────────────────────────────────────────────────
    for seed in range(N_RUNS):
        if seed in done_seeds:
            continue
        print(f"\n{'='*60}\n  Seed {seed:03d} / {N_RUNS-1}\n{'='*60}")
        run_df = run_one(seed, city_data)
        run_df.to_csv(OUT_DIR / f"run_{seed:04d}.csv", index=False)
        gc.collect()

    # ── Aggregate results ─────────────────────────────────────────────────────
    print("\nAggregating all runs...")
    all_files = sorted(OUT_DIR.glob("run_*.csv"))
    all_runs  = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    all_runs.to_csv(OUT_DIR / "all_runs.csv", index=False)

    summary = compute_trimmed_summary(all_runs, TRIM)
    summary.to_csv(OUT_DIR / "summary_trimmed.csv", index=False)

    print(f"\n{'='*70}")
    print(f"TRIMMED-MEAN RESULTS  (dropped {TRIM} best + {TRIM} worst per model/city)")
    print(f"{'='*70}")
    for city in GROUPS:
        sub = summary[summary["city"] == city][
            ["model", "MAPE_mean", "MAPE_std", "MAPE_min", "MAPE_max", "n_runs", "n_kept"]
        ]
        print(f"\n{city}")
        print(sub.to_string(index=False))

    print(f"\nSummary saved to: {OUT_DIR / 'summary_trimmed.csv'}")
    print(f"Raw runs saved to: {OUT_DIR / 'all_runs.csv'}")


if __name__ == "__main__":
    main()
