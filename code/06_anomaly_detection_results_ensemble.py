"""
Compare anomaly detection on validation vs test period using N-BEATS residuals.
Runs the same improved 5-detector ensemble (z-score, IForest, LOF, OC-SVM, K-Means)
on each period separately, then prints a side-by-side summary.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans

PRED_DIR = r'C:/Users/GONCA/Desktop/Iscte/MCD/Theses/results/forecasting/nbeats_val'
OUT_DIR  = r'C:/Users/GONCA/Desktop/Iscte/MCD/Theses/results/anomaly'

CITIES = {
    'Vitoria-Gasteiz':         'Vitoria-Gasteiz',
    'Donostia/San Sebastian':  'Donostia_San_Sebastian',
    'Pamplona/Iruna':          'Pamplona_Iruna',
}

# ── helpers ───────────────────────────────────────────────────────────────────
def rolling_zscore(series, window=30, min_periods=10):
    r = series.rolling(window=window, min_periods=min_periods, center=True)
    return (series - r.mean()) / (r.std() + 1e-8)

def calibrate_contamination(z_series, fallback=0.022):
    rate = float(np.mean(np.abs(z_series) > 3))
    return float(np.clip(rate, 0.01, 0.10)) if rate > 0 else fallback

def build_features(df):
    X = pd.DataFrame({
        'residual':        df['residual'],
        'abs_residual':    df['residual'].abs(),
        'z_residual':      df['z_residual'],
        'lag_residual_1d': df['residual'].shift(1).fillna(0),
        'lag_residual_7d': df['residual'].shift(7).fillna(0),
        'rolling_mean_7d': df['residual'].rolling(7, min_periods=1).mean(),
        'rolling_std_7d':  df['residual'].rolling(7, min_periods=1).std().fillna(0),
        'dow':             df['date'].dt.dayofweek,
        'month':           df['date'].dt.month,
    })
    return X.fillna(0).values

def run_ensemble(df):
    """Add anomaly flags to df (must have 'residual', 'date' columns)."""
    df = df.copy().reset_index(drop=True)
    df['z_residual'] = rolling_zscore(df['residual'])
    X = build_features(df)
    c = calibrate_contamination(df['z_residual'])

    zscore = (np.abs(df['z_residual'].values) > 3).astype(int)
    ifo    = (IsolationForest(contamination=c, random_state=42, n_estimators=200, n_jobs=-1)
              .fit_predict(X) == -1).astype(int)
    lof    = (LocalOutlierFactor(contamination=c, n_neighbors=20)
              .fit_predict(X) == -1).astype(int)
    ocsvm  = (OneClassSVM(nu=c, kernel='rbf', gamma='scale')
              .fit_predict(X) == -1).astype(int)
    km     = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)
    c0_abs = np.abs(df['residual'].values[km == 0]).mean()
    c1_abs = np.abs(df['residual'].values[km == 1]).mean()
    km_flag = (km == (0 if c0_abs > c1_abs else 1)).astype(int)

    votes = zscore + ifo + lof + ocsvm + km_flag
    df['zscore_flag']    = zscore
    df['iforest_flag']   = ifo
    df['lof_flag']       = lof
    df['ocsvm_flag']     = ocsvm
    df['kmeans_flag']    = km_flag
    df['votes']          = votes
    df['ensemble_flag']  = (votes >= 3).astype(int)
    return df

# ── main ──────────────────────────────────────────────────────────────────────
rate_rows   = []
flagged_dfs = {}

for display_name, file_stem in CITIES.items():
    for period in ('val', 'test'):
        path = f'{PRED_DIR}/{file_stem}_{period}_predictions.csv'
        raw  = pd.read_csv(path, parse_dates=['date'])
        raw  = raw.dropna(subset=['actual', 'N-BEATS v2 (covariate-conditioned)'])
        raw['residual'] = raw['actual'] - raw['N-BEATS v2 (covariate-conditioned)']

        res = run_ensemble(raw)
        flagged = res[res['ensemble_flag'] == 1].copy()
        flagged['city']   = display_name
        flagged['period'] = period
        flagged_dfs[(display_name, period)] = flagged

        for method, col in [('Z-score','zscore_flag'),('IForest','iforest_flag'),
                             ('LOF','lof_flag'),('OC-SVM','ocsvm_flag'),
                             ('K-Means','kmeans_flag'),('Ensemble','ensemble_flag')]:
            rate_rows.append({
                'city': display_name, 'period': period, 'method': method,
                'n_days': len(res), 'n_flagged': int(res[col].sum()),
                'rate_%': round(res[col].mean() * 100, 2),
            })

rates = pd.DataFrame(rate_rows)

# ── print rate comparison ─────────────────────────────────────────────────────
print('\n' + '='*70)
print('ANOMALY RATES (%) — N-BEATS RESIDUALS')
print('='*70)
pivot = rates[rates['method'] == 'Ensemble'].pivot_table(
    index='city', columns='period', values='rate_%')
pivot.columns.name = None
pivot = pivot[['val', 'test']]
pivot['delta (pp)'] = (pivot['test'] - pivot['val']).round(2)
print(pivot.to_string())

print('\n\n--- ALL METHODS ---')
all_m = rates.pivot_table(index=['city','method'], columns='period', values='rate_%')
all_m.columns.name = None
all_m = all_m[['val','test']]
print(all_m.to_string())

# ── print flagged days ────────────────────────────────────────────────────────
print('\n\n' + '='*70)
print('FLAGGED DAYS')
print('='*70)
cols_show = ['date','actual','N-BEATS v2 (covariate-conditioned)','residual','votes']

for (city, period), df in flagged_dfs.items():
    df_show = df[['date','actual','N-BEATS v2 (covariate-conditioned)','residual','votes']].copy()
    df_show.columns = ['date','actual','predicted','residual','votes']
    df_show = df_show.sort_values('date')
    print(f'\n{city} — {period} ({len(df_show)} days flagged)')
    if len(df_show):
        print(df_show.to_string(index=False))
    else:
        print('  (none)')

# ── save outputs ──────────────────────────────────────────────────────────────
import os; os.makedirs(OUT_DIR, exist_ok=True)

rates.to_csv(f'{OUT_DIR}/nbeats_anomaly_rates_val_vs_test.csv', index=False)

all_flagged = pd.concat([
    df[['city','period','date','actual','N-BEATS v2 (covariate-conditioned)','residual','votes']]
    for df in flagged_dfs.values() if len(df)
]).sort_values(['city','period','date'])
all_flagged.to_csv(f'{OUT_DIR}/nbeats_anomaly_flagged_days.csv', index=False)

print(f'\nSaved:\n  {OUT_DIR}/nbeats_anomaly_rates_val_vs_test.csv')
print(f'  {OUT_DIR}/nbeats_anomaly_flagged_days.csv')
