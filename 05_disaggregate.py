import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
PREDICTIONS_DIR = Path('predictions')
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Fix 1 — Misclassified buses dropped here so they never enter the disaggregation
# output. Previously only 08_Evaluate.py excluded these, which meant HELIOSCR
# polluted problem-bus selection in 06 (it was the #1 over-prediction).
EXCLUDE_BUSES = ['HELIOSCR_345KV_1']

def compute_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = np.isfinite(y_pred) & (y_true > 0)
    mask_mape = valid & (y_true > 1.0)
    mape  = (np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100
             if mask_mape.sum() > 0 else np.nan)
    mae   = np.mean(np.abs(y_true[valid] - y_pred[valid]))
    rmse  = np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))
    wmape = np.sum(np.abs(y_true[valid] - y_pred[valid])) / max(np.sum(y_true[valid]), 1e-6) * 100

    if label:
        print(f"\n{label}:")
        print(f"  WMAPE: {wmape:6.2f}%   <- primary")
        print(f"  MAPE:  {mape:6.2f}%")
        print(f"  MAE:   {mae:.3f} MW")
        print(f"  RMSE:  {rmse:.3f} MW")

    return {'mape': mape, 'wmape': wmape, 'mae': mae, 'rmse': rmse}

def disaggregate_one_quarter(zone_forecasts,bus_data_path,year,quarter):
    print(f"\nDisaggregating {year} Q{quarter}...")

    shares_path = DATA_DIR / "bus_shares_by_target_quarter" / f"shares_for_{year}_Q{quarter}.parquet"
    if not shares_path.exists():
        print(f"No shares for {year} Q{quarter}, skipping")
        return None
    
    shares = pl.read_parquet(shares_path)
    if 'share_smoothe' in shares.columns:
        shares = shares.rename({'share_smoothe': 'share_smoothed'})

    # ── Fix 2 — Fallback hierarchy for buses whose primary share lookup fails ──
    # Tier A:  matched (bus, dow, hod) share_smoothed   (the normal path)
    # Tier B:  bus's overall fallback_share (any dow/hod, this same bus)
    # Tier C:  zone median share (bus is brand new in 2025, no history at all)
    # Tier D:  0.0 (only if even the zone has no shares — should never happen)
    #
    # Previously share_smoothed defaulted to 0 on lookup failure, which produced
    # zero forecasts for ~750 buses worth ~2.5 WMAPE points. Reconciliation
    # below will scale these fallback shares to keep zone totals consistent.
    bus_overall_fallback = (
        shares.select(['bus_unique_id', 'fallback_share'])
        .unique(subset=['bus_unique_id'])
    )
    zone_median_fallback = (
        shares.filter(pl.col('share_smoothed') > 1e-9)
        .group_by('zone_name')
        .agg(pl.col('share_smoothed').median().alias('zone_median_share'))
    )

    q_start = pd.Timestamp(year=year, month=(quarter-1)*3+1, day=1)
    q_end = (q_start + pd.offsets.QuarterEnd(0)).replace(hour=23)

    zf = zone_forecasts.filter(
        (pl.col('timestamp') >= q_start) & (pl.col('timestamp') <= q_end)
    )

    if zf.height == 0:
        return None

    bus_q = (
        pl.scan_parquet(bus_data_path).with_columns(
            pl.col('date').cast(pl.Date),
            pl.col('zone_name').cast(pl.Utf8),
        ).filter(
            (pl.col('date') >= q_start.date()) &
            (pl.col('date') <= q_end.date()) &
            (pl.col('bus_type') == 'LOAD') &
            pl.col('pd').is_not_null() &
            ~pl.col('bus_unique_id').is_in(EXCLUDE_BUSES)   # Fix 1 — drop misclassified buses
        ).with_columns([
            (pl.col('date').cast(pl.Datetime) + pl.duration(hours=pl.col('he') - 1)).alias('timestamp'),
        ]).select(['bus_unique_id','zone_name','date','he','timestamp','pd']).collect()
    )

    print(f"  Zone forecasts: {zf.height:,}")
    print(f"  Bus actuals:    {bus_q.height:,}")

    bus_q = bus_q.with_columns([
        pl.col('timestamp').dt.weekday().alias('dow'), pl.col('he').alias('hod'),
    ])

    bus_with_zone = bus_q.join(
        zf.with_columns(
            pl.col('zone_name').cast(pl.Utf8),
            pl.col('date').cast(pl.Date),
        ).select(['zone_name','date','he','zone_forecast']),
        on=['zone_name','date','he'], how='left',
    )

    bus_with_shares = (
        bus_with_zone
        .join(
            shares.select(['bus_unique_id', 'dow', 'hod', 'share_smoothed']),
            on=['bus_unique_id', 'dow', 'hod'], how='left',
        )
        .join(bus_overall_fallback, on='bus_unique_id', how='left')
        .join(zone_median_fallback, on='zone_name', how='left')
        .with_columns(
            # Apply the fallback hierarchy described above.
            pl.when((pl.col('share_smoothed').is_not_null())
                    & (pl.col('share_smoothed') > 1e-9))
            .then(pl.col('share_smoothed'))
            .when((pl.col('fallback_share').is_not_null())
                  & (pl.col('fallback_share') > 1e-9))
            .then(pl.col('fallback_share'))
            .when(pl.col('zone_median_share').is_not_null())
            .then(pl.col('zone_median_share'))
            .otherwise(0.0)
            .alias('share_smoothed')
        )
        .drop('fallback_share', 'zone_median_share')
    )

    # Diagnostic: how many rows landed on each tier
    n_zero_after = bus_with_shares.filter(pl.col('share_smoothed') < 1e-9).height
    if n_zero_after > 0:
        print(f"  WARN: {n_zero_after:,} rows still have share=0 after all fallbacks")

    bus_forecasts = bus_with_shares.with_columns([
        (pl.col('zone_forecast') * pl.col('share_smoothed')).alias('bus_forecast'),
    ])

    return bus_forecasts.select([
        'bus_unique_id','zone_name','date','he','timestamp','pd','bus_forecast','zone_forecast','share_smoothed',
    ])

def reconcile_to_zone(bus_forecasts,zone_forecasts):
    print("Reconciling so sum(buses) = zone_forecast...")

    bus_sum = (
        bus_forecasts.group_by(['zone_name','date','he']).agg(pl.col('bus_forecast').sum().alias('bus_sum'))
    )

    scale = (
        bus_sum.join(
            zone_forecasts.with_columns(
                pl.col('zone_name').cast(pl.Utf8),
                pl.col('date').cast(pl.Date),
            ).select(["zone_name",'date','he','zone_forecast']),
            on=['zone_name','date','he'], how='left',
        ).with_columns([
            pl.when(pl.col('bus_sum') > 0).then(pl.col('zone_forecast') / pl.col('bus_sum'))
            .otherwise(1.0).alias('scale_factor'),
        ]).select(['zone_name','date','he','scale_factor'])
    )

    reconciled = bus_forecasts.join(
        scale, on=['zone_name','date','he'],how='left',
    ).with_columns([
        (pl.col('bus_forecast') * pl.col('scale_factor')).alias('bus_forecast_reconciled'),
    ])

    return reconciled

def disaggregate_task(zone_forecasts_path, bus_data_path, task_name, target_year):
    print("\n" + "="*70)
    print(f"DISAGGREGATING {task_name.upper()} ({target_year})")
    print("="*70)
    
    zone_fc = pl.read_parquet(zone_forecasts_path)

    output_files = []

    for q in [1,2,3,4]:
        bus_fc = disaggregate_one_quarter(zone_fc,bus_data_path, target_year, q)

        if bus_fc is None:
            continue

        bus_fc = reconcile_to_zone(bus_fc, zone_fc)
        out_path = PREDICTIONS_DIR / f"{task_name}_{target_year}_Q{q}.parquet"
        bus_fc.write_parquet(out_path)
        output_files.append(out_path)
        print(f"Saved -> {out_path.name}")

        compute_metrics(
            bus_fc['pd'].to_numpy(),
            bus_fc['bus_forecast_reconciled'].to_numpy(),
            label = f"{task_name} {target_year} Q{q}"
        )

    if output_files:
        combined = pl.concat([pl.read_parquet(f) for f in output_files])
        compute_metrics(
            combined['pd'].to_numpy(),
            combined['bus_forecast_reconciled'].to_numpy(),
            label=f"{task_name} {target_year} FULL YEAR"
        )

def main():
    bus_data_path = DATA_DIR / 'bus_load.parquet'

    if not bus_data_path.exists():
        print('Missing data/bus_load.parquet')
        return
    
    task1_zone_fc = DATA_DIR / 'task1_zone_forecast.parquet'

    if task1_zone_fc.exists():
        disaggregate_task(task1_zone_fc,bus_data_path,'task1',2025)

    # Bus-level val forecasts (2024 H2) — consumed by 06_idenitfy_problem_buses.py
    # to pick problem buses on a fold that isn't the test year. Required to avoid
    # selection bias on 2025.
    task1_val_zone_fc = DATA_DIR / 'task1_val_zone_forecast.parquet'
    if task1_val_zone_fc.exists():
        disaggregate_task(task1_val_zone_fc, bus_data_path, 'task1_val', 2024)

    task2_zone_fc = DATA_DIR / 'task2_zone_forecasts.parquet'

    if task2_zone_fc.exists():
        disaggregate_task(task2_zone_fc,bus_data_path,'task2',2025)

    print("\n" + "="*70)
    print("STAGE 4 COMPLETE — next step: python 05_identify_problem_buses.py")
    print("="*70)
 
 
if __name__ == "__main__":
    main()