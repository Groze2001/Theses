"""
nbeats_train_vs_test_anomaly.py
================================
Extends the existing validation-vs-test anomaly comparison (Table 4.10/4.11,
produced by nbeats_val.py + nbeats_anomaly_compare.py) with a third period:
the N-BEATS v2 training set itself.

Rationale: Tables 4.10/4.11 already show the ensemble is conservative on
both validation and test residuals. Comparing those against the *training*
residuals answers a different question: is the low anomaly rate a property
of the residual signal in general, or specific to data the model has not
seen? In-sample (training) residuals are expected to be small almost by
construction, since AdamW/MSE training directly minimises them; a very low
training anomaly rate is therefore NOT evidence the model generalises well
on its own. What is informative is the *shape* of the gap: if training
residuals are near-zero and uniformly small while test residuals are larger
and cluster around specific calendar/weather events (as in Table 4.11),
that is consistent with a model that fits the training signal cleanly
without memorising test-period noise -- i.e. it is not simply overfit to
whatever falls in the training window. If training and test anomaly rates
were similar, that would instead suggest under-fitting (the model is
noisy everywhere, not just out-of-sample).

Pipeline (must match nbeats_val.py / nbeats_anomaly_compare.py exactly so
train, val, and test numbers stay directly comparable):
  1. Train N-BEATS v2 per city with the same architecture, features, and
     70/15/15 chronological split as the rest of Chapter 4.
  2. Generate in-sample "predictions" for the training period itself, using
     the same rolling-window inference as val/test (each day's forecast is
     conditioned on the preceding 30 real days, exactly as during training).
     The first 30 training days have no valid preceding window and are
     dropped, exactly as they are during model training.
  3. Run the identical 5-detector majority-vote ensemble (rolling z-score,
     Isolation Forest, LOF, One-Class SVM, K-Means, >=3/5 vote) used for
     Table 4.10/4.11, separately on the training and test residuals.
  4. Save rates and flagged days in the same format as the existing
     nbeats_anomaly_rates_val_vs_test.csv / nbeats_anomaly_flagged_days.csv,
     so the two can be joined directly.

Usage:
    python nbeats_train_vs_test_anomaly.py

Output (results/anomaly/):
    nbeats_anomaly_rates_train_vs_test.csv
    nbeats_anomaly_flagged_days_train_vs_test.csv
"""
import warnings, random, gc
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
BASE_PATH  = Path(__file__).resolve().parents[1]
DATA_PATH  = BASE_PATH / "results" / "data" / "municipality_daily_consumption.csv"
OUT_DIR    = BASE_PATH / "results" / "anomaly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUPS     = ["Vitoria-Gasteiz", "Donostia/San Sebastian", "Pamplona/Iruna"]
TARGET_COL = "avg_kwh"
DATE_COL   = "date"
EPOCHS     = 80
BATCH_SIZE = 64
PATIENCE   = 15
SEQ_LENGTH = 30

CALENDAR_FEATURES = ["is_weekend", "is_holiday_es", "is_bridge_day",
                      "sin_dow", "cos_dow", "sin_month", "cos_month", "sin_week", "cos_week"]
HISTORY_FEATURES  = ["lag_1d", "lag_7d", "lag_14d", "lag_30d",
                      "roll7_mean", "roll7_std", "roll30_mean", "roll7_ratio", "wow_change", "dod_change"]
NBEATS_COLS = [TARGET_COL] + CALENDAR_FEATURES + HISTORY_FEATURES

# ── Feature engineering (identical to nbeats_val.py) ───────────────────────────
def add_calendar(df):
    df = df.copy()
    df["day_of_week"]  = df[DATE_COL].dt.dayofweek
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"]        = df[DATE_COL].dt.month
    df["week_of_year"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["sin_dow"]   = np.sin(2 * np.pi * df["day_of_week"]  / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * df["day_of_week"]  / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"]        / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"]        / 12)
    df["sin_week"]  = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["cos_week"]  = np.cos(2 * np.pi * df["week_of_year"] / 52)
    return df

def add_holidays(df):
    df = df.copy()
    try:
        import holidays
        years  = sorted(df[DATE_COL].dt.year.unique())
        es_h   = holidays.country_holidays("ES", years=years)
        hdates = set(pd.to_datetime(list(es_h.keys())))
        df["is_holiday_es"] = df[DATE_COL].isin(hdates).astype(int)
        nd  = df[DATE_COL] + pd.Timedelta(days=1)
        pd_ = df[DATE_COL] - pd.Timedelta(days=1)
        df["is_bridge_day"] = (
            ((df[DATE_COL].dt.dayofweek == 0) & nd.isin(hdates)) |
            ((df[DATE_COL].dt.dayofweek == 4) & pd_.isin(hdates))
        ).astype(int)
    except Exception:
        df["is_holiday_es"] = 0
        df["is_bridge_day"] = 0
    return df

def add_lags(df):
    df = df.sort_values(["municipality", DATE_COL]).copy()
    g  = df.groupby("municipality", group_keys=False)[TARGET_COL]
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}d"] = g.shift(lag)
    df["roll7_mean"]  = g.shift(1).rolling(7,  min_periods=1).mean().reset_index(level=0, drop=True)
    df["roll7_std"]   = g.shift(1).rolling(7,  min_periods=2).std().reset_index(level=0, drop=True)
    df["roll30_mean"] = g.shift(1).rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)
    df["roll7_ratio"] = df["lag_1d"] / df["roll7_mean"].replace(0, np.nan)
    df["wow_change"]  = (df["lag_1d"] - df["lag_7d"]) / df["lag_7d"].replace(0, np.nan)
    lag2 = g.shift(2)
    df["dod_change"]  = (df["lag_1d"] - lag2) / lag2.replace(0, np.nan)
    return df

# ── N-BEATS v2 (identical architecture to nbeats_val.py) ───────────────────────
class WindowDataset(Dataset):
    def __init__(self, X, y, seq_length=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.seq_length = seq_length
    def __len__(self): return len(self.y) - self.seq_length
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_length], self.y[idx+self.seq_length]

class NBeatsBlock(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, covariate_size=0, dropout=0.2):
        super().__init__()
        in_dim = input_size + covariate_size
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
        )
        self.backcast = nn.Linear(hidden_size, input_size)
        self.forecast = nn.Linear(hidden_size, 1)
    def forward(self, x, cov=None):
        B, T, F = x.shape
        flat = x.reshape(B, T * F)
        if cov is not None:
            flat = torch.cat([flat, cov], dim=1)
        h  = self.fc(flat)
        bc = self.backcast(h).unsqueeze(-1)
        fc = self.forecast(h)
        return bc, fc

class NBeatsModel(nn.Module):
    def __init__(self, input_size=30, n_blocks=3, hidden_size=64, covariate_size=0, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, covariate_size, dropout)
            for _ in range(n_blocks)
        ])
    def forward(self, x):
        cov   = x[:, -1, :]
        resid = x[:, :, :1]
        total = torch.zeros(x.size(0), 1, device=x.device)
        for blk in self.blocks:
            bc, fc = blk(resid, cov)
            resid  = resid - bc
            total  = total + fc
        return total

def train_model(model, train_loader, val_loader):
    model = model.to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_val, bad = None, np.inf, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = [loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item() for xb, yb in val_loader]
        vl = float(np.mean(vloss)) if vloss else np.inf
        if vl < best_val:
            best_val = vl; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  Early stop epoch {epoch}. Best val={best_val:.5f}"); break
    if best_state: model.load_state_dict(best_state)
    return model

def predict_windows(model, X_context, y_scaler, index, seq_length=30):
    model.eval()
    ds     = WindowDataset(X_context, np.zeros(len(X_context)), seq_length)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    preds = y_scaler.inverse_transform(np.vstack(preds).ravel().reshape(-1, 1)).ravel()
    return pd.Series(preds, index=index[seq_length:])

# ── Ensemble anomaly detector (identical to nbeats_anomaly_compare.py) ────────
def rolling_zscore(series, window=30, min_periods=10):
    r = series.rolling(window=window, min_periods=min_periods, center=True)
    return (series - r.mean()) / (r.std() + 1e-8)

def calibrate_contamination(z_series, fallback=0.022):
    rate = float(np.mean(np.abs(z_series) > 3))
    return float(np.clip(rate, 0.01, 0.10)) if rate > 0 else fallback

def build_ensemble_features(df):
    X = pd.DataFrame({
        "residual":        df["residual"],
        "abs_residual":    df["residual"].abs(),
        "z_residual":      df["z_residual"],
        "lag_residual_1d": df["residual"].shift(1).fillna(0),
        "lag_residual_7d": df["residual"].shift(7).fillna(0),
        "rolling_mean_7d": df["residual"].rolling(7, min_periods=1).mean(),
        "rolling_std_7d":  df["residual"].rolling(7, min_periods=1).std().fillna(0),
        "dow":             df["date"].dt.dayofweek,
        "month":           df["date"].dt.month,
    })
    return X.fillna(0).values

def run_ensemble(df):
    df = df.copy().reset_index(drop=True)
    df["z_residual"] = rolling_zscore(df["residual"])
    X = build_ensemble_features(df)
    c = calibrate_contamination(df["z_residual"])

    zscore = (np.abs(df["z_residual"].values) > 3).astype(int)
    ifo    = (IsolationForest(contamination=c, random_state=42, n_estimators=200, n_jobs=-1)
              .fit_predict(X) == -1).astype(int)
    lof    = (LocalOutlierFactor(contamination=c, n_neighbors=20)
              .fit_predict(X) == -1).astype(int)
    ocsvm  = (OneClassSVM(nu=c, kernel="rbf", gamma="scale")
              .fit_predict(X) == -1).astype(int)
    km     = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)
    c0_abs = np.abs(df["residual"].values[km == 0]).mean()
    c1_abs = np.abs(df["residual"].values[km == 1]).mean()
    km_flag = (km == (0 if c0_abs > c1_abs else 1)).astype(int)

    votes = zscore + ifo + lof + ocsvm + km_flag
    df["zscore_flag"]   = zscore
    df["iforest_flag"]  = ifo
    df["lof_flag"]      = lof
    df["ocsvm_flag"]    = ocsvm
    df["kmeans_flag"]   = km_flag
    df["votes"]         = votes
    df["ensemble_flag"] = (votes >= 3).astype(int)
    return df

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df = add_calendar(df)
    df = add_holidays(df)
    df = add_lags(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    rate_rows, flagged_dfs = [], {}

    for city in GROUPS:
        print(f"\n{'='*60}\nN-BEATS v2 -- {city}\n{'='*60}")

        part = df[df["municipality"] == city].sort_values(DATE_COL).copy()
        part = part.dropna(subset=NBEATS_COLS).reset_index(drop=True)
        part = part.set_index(DATE_COL)

        n = len(part)
        i_tr = int(n * 0.70)
        i_va = int(n * 0.85)
        train, val, test = part.iloc[:i_tr], part.iloc[i_tr:i_va], part.iloc[i_va:]
        print(f"  train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} days)")
        print(f"  test:  {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)")

        cols = [c for c in NBEATS_COLS if c in part.columns]
        x_sc, y_sc = StandardScaler(), StandardScaler()
        X_tr = x_sc.fit_transform(train[cols].fillna(0).astype(float))
        X_va = x_sc.transform(val[cols].fillna(0).astype(float))
        X_te = x_sc.transform(test[cols].fillna(0).astype(float))
        y_tr = y_sc.fit_transform(train[[TARGET_COL]].astype(float)).ravel()
        y_va = y_sc.transform(val[[TARGET_COL]].astype(float)).ravel()

        tr_loader = DataLoader(WindowDataset(X_tr, y_tr, SEQ_LENGTH), BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(WindowDataset(X_va, y_va, SEQ_LENGTH), BATCH_SIZE, shuffle=False)

        model = NBeatsModel(input_size=SEQ_LENGTH, n_blocks=3, hidden_size=64,
                             covariate_size=len(cols), dropout=0.2)
        model = train_model(model, tr_loader, va_loader)

        # In-sample predictions on the training window itself: each day is
        # still conditioned on the preceding 30 REAL days, exactly as during
        # training (teacher forcing) and exactly as val/test are scored.
        pred_train = predict_windows(model, X_tr, y_sc, train.index, SEQ_LENGTH)
        pred_test  = predict_windows(model, np.vstack([X_va, X_te]), y_sc,
                                      val.index.append(test.index), SEQ_LENGTH)
        pred_test  = pred_test.reindex(test.index).dropna()

        for period, part_df, pred in (("train", train, pred_train), ("test", test, pred_test)):
            raw = pd.DataFrame({
                "date":                              pred.index,
                "actual":                            part_df[TARGET_COL].reindex(pred.index).values,
                "N-BEATS v2 (covariate-conditioned)": pred.values,
            }).dropna()
            raw["residual"] = raw["actual"] - raw["N-BEATS v2 (covariate-conditioned)"]

            res = run_ensemble(raw)
            flagged = res[res["ensemble_flag"] == 1].copy()
            flagged["city"] = city
            flagged["period"] = period
            flagged_dfs[(city, period)] = flagged

            for method, col in [("Z-score", "zscore_flag"), ("IForest", "iforest_flag"),
                                 ("LOF", "lof_flag"), ("OC-SVM", "ocsvm_flag"),
                                 ("K-Means", "kmeans_flag"), ("Ensemble", "ensemble_flag")]:
                rate_rows.append({
                    "city": city, "period": period, "method": method,
                    "n_days": len(res), "n_flagged": int(res[col].sum()),
                    "rate_%": round(res[col].mean() * 100, 2),
                })
            print(f"  {period}: {len(res)} days scored, "
                  f"{int(res['ensemble_flag'].sum())} ensemble-flagged "
                  f"({res['ensemble_flag'].mean()*100:.2f}%)")

        gc.collect()

    rates = pd.DataFrame(rate_rows)
    rates.to_csv(OUT_DIR / "nbeats_anomaly_rates_train_vs_test.csv", index=False)

    all_flagged = pd.concat([
        d[["city", "period", "date", "actual", "N-BEATS v2 (covariate-conditioned)", "residual", "votes"]]
        for d in flagged_dfs.values() if len(d)
    ]).sort_values(["city", "period", "date"])
    all_flagged.to_csv(OUT_DIR / "nbeats_anomaly_flagged_days_train_vs_test.csv", index=False)

    print("\n" + "=" * 70)
    print("TRAIN vs TEST ENSEMBLE ANOMALY RATES (%)")
    print("=" * 70)
    pivot = rates[rates["method"] == "Ensemble"].pivot_table(
        index="city", columns="period", values="rate_%")
    pivot.columns.name = None
    pivot = pivot[["train", "test"]]
    pivot["delta (pp)"] = (pivot["test"] - pivot["train"]).round(2)
    print(pivot.to_string())

    print(f"\nSaved:\n  {OUT_DIR / 'nbeats_anomaly_rates_train_vs_test.csv'}")
    print(f"  {OUT_DIR / 'nbeats_anomaly_flagged_days_train_vs_test.csv'}")


if __name__ == "__main__":
    main()
