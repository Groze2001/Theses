"""
Run N-BEATS v2 only — saves validation predictions per city.
Identical setup to 4_forecasting_models.ipynb.
"""
import warnings, random, gc
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_PATH  = Path(r'C:/Users/GONCA/Desktop/Iscte/MCD/Theses')
DATA_PATH  = BASE_PATH / 'results' / 'data' / 'municipality_daily_consumption.csv'
OUT_DIR    = BASE_PATH / 'results' / 'forecasting' / 'nbeats_val'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUPS     = ['Vitoria-Gasteiz', 'Donostia/San Sebastian', 'Pamplona/Iruna']
TARGET_COL = 'avg_kwh'
DATE_COL   = 'date'
EPOCHS     = 80
BATCH_SIZE = 64
PATIENCE   = 15
SEQ_LENGTH = 30

CALENDAR_FEATURES = ['is_weekend','is_holiday_es','is_bridge_day',
                     'sin_dow','cos_dow','sin_month','cos_month','sin_week','cos_week']
HISTORY_FEATURES  = ['lag_1d','lag_7d','lag_14d','lag_30d',
                     'roll7_mean','roll7_std','roll30_mean','roll7_ratio','wow_change','dod_change']

# ── Feature engineering ────────────────────────────────────────────────────────
def add_calendar(df):
    df = df.copy()
    df['day_of_week']  = df[DATE_COL].dt.dayofweek
    df['is_weekend']   = df['day_of_week'].isin([5, 6]).astype(int)
    df['month']        = df[DATE_COL].dt.month
    df['week_of_year'] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df['sin_dow']   = np.sin(2*np.pi*df['day_of_week']/7)
    df['cos_dow']   = np.cos(2*np.pi*df['day_of_week']/7)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12)
    df['sin_week']  = np.sin(2*np.pi*df['week_of_year']/52)
    df['cos_week']  = np.cos(2*np.pi*df['week_of_year']/52)
    return df

def add_holidays(df):
    df = df.copy()
    try:
        import holidays
        years = sorted(df[DATE_COL].dt.year.unique())
        es_h  = holidays.country_holidays('ES', years=years)
        hdates = set(pd.to_datetime(list(es_h.keys())))
        df['is_holiday_es'] = df[DATE_COL].isin(hdates).astype(int)
        nd = df[DATE_COL] + pd.Timedelta(days=1)
        pd_ = df[DATE_COL] - pd.Timedelta(days=1)
        df['is_bridge_day'] = (
            ((df[DATE_COL].dt.dayofweek == 0) & nd.isin(hdates)) |
            ((df[DATE_COL].dt.dayofweek == 4) & pd_.isin(hdates))
        ).astype(int)
    except Exception:
        df['is_holiday_es'] = 0
        df['is_bridge_day'] = 0
    return df

def add_lags(df):
    df = df.sort_values(['municipality', DATE_COL]).copy()
    g  = df.groupby('municipality', group_keys=False)[TARGET_COL]
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}d'] = g.shift(lag)
    df['roll7_mean']  = g.shift(1).rolling(7,  min_periods=1).mean().reset_index(level=0, drop=True)
    df['roll7_std']   = g.shift(1).rolling(7,  min_periods=2).std().reset_index(level=0, drop=True)
    df['roll30_mean'] = g.shift(1).rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)
    df['roll7_ratio'] = df['lag_1d'] / df['roll7_mean'].replace(0, np.nan)
    df['wow_change']  = (df['lag_1d'] - df['lag_7d']) / df['lag_7d'].replace(0, np.nan)
    lag2 = g.shift(2)
    df['dod_change']  = (df['lag_1d'] - lag2) / lag2.replace(0, np.nan)
    return df

# ── PyTorch utilities ──────────────────────────────────────────────────────────
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
        # x: (batch, seq_len, features) — flatten last two dims
        B, T, F = x.shape
        flat = x.reshape(B, T * F)
        if cov is not None:
            flat = torch.cat([flat, cov], dim=1)
        h  = self.fc(flat)
        bc = self.backcast(h).unsqueeze(-1)  # (B, T, 1)
        fc = self.forecast(h)                # (B, 1)
        return bc, fc

class NBeatsModel(nn.Module):
    def __init__(self, input_size=30, n_blocks=3, hidden_size=64, covariate_size=0, dropout=0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, covariate_size, dropout)
            for _ in range(n_blocks)
        ])
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        # last column is the target channel; covariate = last time step of all features
        cov   = x[:, -1, :]          # (B, n_features)
        resid = x[:, :, :1]          # (B, seq_len, 1) — target channel
        total = torch.zeros(x.size(0), 1, device=x.device)
        for blk in self.blocks:
            bc, fc = blk(resid, cov)
            resid  = resid - bc
            total  = total + fc
        return total

def train_model(model, train_loader, val_loader):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state, best_val, bad = None, np.inf, 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        model.eval(); vloss = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vloss.append(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item())
        vl = float(np.mean(vloss)) if vloss else np.inf
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f'  Epoch {epoch:03d} | train={np.mean(losses):.5f} | val={vl:.5f}')
        if bad >= PATIENCE:
            print(f'  Early stop epoch {epoch}. Best val={best_val:.5f}'); break
    if best_state: model.load_state_dict(best_state)
    return model

def predict_windows(model, X_context, y_scaler, index, seq_length=30):
    model.eval()
    ds     = WindowDataset(X_context, np.zeros(len(X_context)), seq_length)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    preds  = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    preds = y_scaler.inverse_transform(
        np.vstack(preds).ravel().reshape(-1, 1)
    ).ravel()
    return pd.Series(preds, index=index[seq_length:])

def mape(actual, pred):
    a, p = np.array(actual, float), np.array(pred, float)
    mask = a != 0
    return np.nanmean(np.abs((a[mask] - p[mask]) / a[mask])) * 100

# ── Load & prep data ───────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = add_calendar(df)
df = add_holidays(df)
df = add_lags(df)
df = df.replace([np.inf, -np.inf], np.nan)

NBEATS_COLS = [TARGET_COL] + CALENDAR_FEATURES + HISTORY_FEATURES

# ── Per-city loop ──────────────────────────────────────────────────────────────
for city in GROUPS:
    print(f'\n{"="*60}\nN-BEATS v2 — {city}\n{"="*60}')

    part = df[df['municipality'] == city].sort_values(DATE_COL).copy()
    part = part.dropna(subset=NBEATS_COLS).reset_index(drop=True)
    part = part.set_index(DATE_COL)

    n = len(part)
    i_tr = int(n * 0.70)
    i_va = int(n * 0.85)
    train, val, test = part.iloc[:i_tr], part.iloc[i_tr:i_va], part.iloc[i_va:]
    print(f'  train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)')
    print(f'  val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} days)')
    print(f'  test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days)')

    cols = [c for c in NBEATS_COLS if c in part.columns]
    x_sc, y_sc = StandardScaler(), StandardScaler()
    X_tr = x_sc.fit_transform(train[cols].fillna(0).astype(float))
    X_va = x_sc.transform(val[cols].fillna(0).astype(float))
    X_te = x_sc.transform(test[cols].fillna(0).astype(float))
    y_tr = y_sc.fit_transform(train[[TARGET_COL]].astype(float)).ravel()
    y_va = y_sc.transform(val[[TARGET_COL]].astype(float)).ravel()

    tr_loader = DataLoader(WindowDataset(X_tr, y_tr, SEQ_LENGTH), BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(WindowDataset(X_va, y_va, SEQ_LENGTH), BATCH_SIZE, shuffle=False)

    model = NBeatsModel(
        input_size=SEQ_LENGTH,
        n_blocks=3,
        hidden_size=64,
        covariate_size=len(cols),
        dropout=0.2,
    )
    model = train_model(model, tr_loader, va_loader)

    # Predict: seed with val then test so the window is warm
    X_ctx   = np.vstack([X_va, X_te])
    ctx_idx = val.index.append(test.index)
    pred_all = predict_windows(model, X_ctx, y_sc, ctx_idx, SEQ_LENGTH)

    pred_val  = pred_all.reindex(val.index).dropna()
    pred_test = pred_all.reindex(test.index).dropna()

    val_mape  = mape(val[TARGET_COL].reindex(pred_val.index),  pred_val)
    test_mape = mape(test[TARGET_COL].reindex(pred_test.index), pred_test)
    print(f'  Val MAPE:  {val_mape:.2f}%')
    print(f'  Test MAPE: {test_mape:.2f}%')

    # Save validation predictions
    safe = city.replace('/', '_').replace(' ', '_')
    out = pd.DataFrame({
        'actual':                           val[TARGET_COL].reindex(pred_val.index),
        'N-BEATS v2 (covariate-conditioned)': pred_val,
        'residual': val[TARGET_COL].reindex(pred_val.index) - pred_val,
    })
    out.index.name = 'date'
    out.to_csv(OUT_DIR / f'{safe}_val_predictions.csv')
    print(f'  Saved: {OUT_DIR / (safe + "_val_predictions.csv")}')

    # Also save test predictions (for reference)
    out_test = pd.DataFrame({
        'actual':                           test[TARGET_COL].reindex(pred_test.index),
        'N-BEATS v2 (covariate-conditioned)': pred_test,
        'residual': test[TARGET_COL].reindex(pred_test.index) - pred_test,
    })
    out_test.index.name = 'date'
    out_test.to_csv(OUT_DIR / f'{safe}_test_predictions.csv')

    gc.collect()

print('\nDone. Files saved to:', OUT_DIR)
