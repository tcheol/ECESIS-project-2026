import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
PREDICTIONS_DIR = Path('predictions')
PREDICTIONS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
    mask_mape = mask & (y_true > 1.0)
    mape  = np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100 if mask_mape.sum() > 0 else np.nan
    mae   = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    rmse  = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    wmape = np.sum(np.abs(y_true[mask] - y_pred[mask])) / max(np.sum(y_true[mask]), 1e-6) * 100
    if label:
        print(f"\n{label}:")
        print(f"  WMAPE: {wmape:6.2f}%   <- primary")
        print(f"  MAPE:  {mape:6.2f}%")
        print(f"  MAE:   {mae:.3f} MW")
        print(f"  RMSE:  {rmse:.3f} MW")
    return {"mape": mape, "wmape": wmape, "mae": mae, "rmse": rmse}


def scan_load(year):
    path = DATA_DIR / f'bus_load_{year}.parquet'
    if not path.exists():
        return None
    return (
        pl.scan_parquet(path)
        .filter(pl.col('bus_type') == 'LOAD')
        .filter(pl.col('pd').is_not_null())
    )


def main():
    # ── Step 1: historical average (pre-2025 only) ────────────────────────────
    print("Computing historical averages from 2022-2024 (pre-2025 only)...")
    hist_frames = [scan_load(y) for y in [2022, 2023, 2024]]
    hist_frames = [f for f in hist_frames if f is not None]
    hist_avg = (
        pl.concat(hist_frames)
        .with_columns(
            (pl.col('date').cast(pl.Date).cast(pl.Datetime('us')) +
             pl.duration(hours=pl.col('he') - 1)).alias('timestamp')
        )
        .with_columns(pl.col('timestamp').dt.weekday().alias('day_of_week'))
        .group_by(['bus_unique_id', 'he', 'day_of_week'])
        .agg(pl.col('pd').mean().alias('hist_avg'))
        .collect()
    )
    print(f"  {hist_avg.height:,} (bus, he, dow) averages")

    # ── Step 2: year-ago baseline — join 2024 actuals onto 2025 by calendar date ─
    # For any 2025-MM-DD HE-H, the year-ago value is the 2024-MM-DD HE-H actual.
    # This uses ZERO 2025 data as input features.
    print("Building year-ago (lag_8760h) baseline from 2024 actuals...")
    lag_8760 = (
        scan_load(2024)
        .with_columns([
            pl.col('date').cast(pl.Date).dt.month().alias('month'),
            pl.col('date').cast(pl.Date).dt.day().alias('day'),
        ])
        .select(['bus_unique_id', 'zone_name', 'month', 'day', 'he',
                 pl.col('pd').alias('lag_8760h')])
        .collect()
    )
    print(f"  {lag_8760.height:,} year-ago reference rows")

    # ── Step 3: load 2025 actuals for evaluation ONLY (not used as features) ───
    print("Loading 2025 actuals for evaluation...")
    lf_2025 = scan_load(2025)
    if lf_2025 is None:
        print("ERROR: bus_load_2025.parquet not found")
        return
    test = (
        lf_2025
        .with_columns([
            pl.col('date').cast(pl.Date).alias('date'),
            (pl.col('date').cast(pl.Date).cast(pl.Datetime('us')) +
             pl.duration(hours=pl.col('he') - 1)).alias('timestamp'),
            pl.col('date').cast(pl.Date).dt.month().alias('month'),
            pl.col('date').cast(pl.Date).dt.day().alias('day'),
            pl.col('date').cast(pl.Date).dt.weekday().alias('day_of_week'),
        ])
        .select(['bus_unique_id', 'zone_name', 'date', 'he', 'timestamp',
                 'pd', 'month', 'day', 'day_of_week'])
        .collect()
    )
    print(f"  {test.height:,} 2025 rows")

    # ── Step 4: attach baseline columns ──────────────────────────────────────
    test = (
        test
        .join(lag_8760, on=['bus_unique_id', 'zone_name', 'month', 'day', 'he'], how='left')
        .join(hist_avg,  on=['bus_unique_id', 'he', 'day_of_week'],              how='left')
    )

    test_pd = test.to_pandas()
    actual  = test_pd['pd'].values

    # ── Task 1 baselines ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TASK 1 (Next-Day) — Baselines  [all built from pre-2025 data]")
    print("="*70)
    compute_metrics(actual, test_pd['lag_8760h'].values, "Baseline: same hour prev year (lag_8760h)")
    compute_metrics(actual, test_pd['hist_avg'].values,  "Baseline: historical avg by bus/hour/dow")

    print("\nPer-zone WMAPE (lag_8760h):")
    for zone, g in test_pd.groupby('zone_name'):
        valid = np.isfinite(g['lag_8760h']) & (g['pd'] > 0)
        if valid.sum() == 0:
            continue
        m = np.sum(np.abs(g.loc[valid,'pd'] - g.loc[valid,'lag_8760h'])) / max(g.loc[valid,'pd'].sum(), 1e-6) * 100
        print(f"  {zone}: {m:.2f}%")

    t1 = pd.DataFrame({
        'model_name': 'baseline_lag8760',
        'forecast_created_at': pd.to_datetime(test_pd['date']) - pd.Timedelta(days=1) + pd.Timedelta(minutes=1),
        'target_date': test_pd['date'],
        'he': test_pd['he'].astype(int),
        'bus_id': test_pd['bus_unique_id'],
        'zone_id': test_pd['zone_name'].astype(str),
        'predict_pd': test_pd['lag_8760h'].clip(lower=0),
    })
    t1.to_csv(PREDICTIONS_DIR / 'task1_baseline.csv', index=False)
    print(f"\nSaved predictions/task1_baseline.csv ({len(t1):,} rows)")

    # ── Task 2 baselines ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TASK 2 (Next-Month) — Baselines  [all built from pre-2025 data]")
    print("="*70)
    compute_metrics(actual, test_pd['lag_8760h'].values, "Baseline: same hour prev year (lag_8760h)")
    compute_metrics(actual, test_pd['hist_avg'].values,  "Baseline: historical avg by bus/hour/dow")

    print("\nPer-zone WMAPE (lag_8760h):")
    for zone, g in test_pd.groupby('zone_name'):
        valid = np.isfinite(g['lag_8760h']) & (g['pd'] > 0)
        if valid.sum() == 0:
            continue
        m = np.sum(np.abs(g.loc[valid,'pd'] - g.loc[valid,'lag_8760h'])) / max(g.loc[valid,'pd'].sum(), 1e-6) * 100
        print(f"  {zone}: {m:.2f}%")

    t2 = pd.DataFrame({
        'model_name': 'baseline_lag8760',
        'forecast_created_at': (pd.to_datetime(test_pd['date']) - pd.offsets.MonthBegin(1)).dt.normalize(),
        'target_date': test_pd['date'],
        'he': test_pd['he'].astype(int),
        'bus_id': test_pd['bus_unique_id'],
        'zone_id': test_pd['zone_name'].astype(str),
        'predict_pd': test_pd['lag_8760h'].clip(lower=0),
    })
    t2.to_csv(PREDICTIONS_DIR / 'task2_baseline.csv', index=False)
    print(f"Saved predictions/task2_baseline.csv ({len(t2):,} rows)")


if __name__ == '__main__':
    main()
