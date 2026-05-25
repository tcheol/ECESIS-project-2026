import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
PREDICTIONS_DIR = Path('predictions')

PROBLEM_THRESHOLD_WMAPE = 15.0

# Fix 4 — Split the load threshold by purpose:
#   * MIN_LOAD_FOR_WMAPE: below this, WMAPE is undefined (tiny denominator → noise)
#   * MIN_LOAD_FOR_SELECTION: below this, a bus contributes <0.01% of aggregate
#     load and isn't worth a specialized model — even if its WMAPE is high.
#     Previously a single 0.1 MW threshold let sub-MW noise buses (DRIVER,
#     JCHRSTL) crowd out real problem buses (data centers) from the slot list.
MIN_LOAD_FOR_WMAPE = 0.1
MIN_LOAD_FOR_SELECTION = 1.0

# Empirically tested 500 / 700 / 1500. 1500 wins on Task 1 (~2.5 pp better than
# 700) and ties on Task 2. The val-WMAPE ranking is informative all the way
# through ~1500 — the earlier hypothesis that it went noisy past ~500 was wrong.
MAX_PROBLEM_BUSES = 1500

def evaluate_per_bus(prediction_files):
    print('Loading predictions...')
    df = pl.concat([pl.read_parquet(f) for f in prediction_files])
    print(f'Total rows: {df.height:,}')

    per_bus = (
        df.group_by(['bus_unique_id','zone_name'])
        .agg([
            pl.col('pd').sum().alias('total_actual'),
            pl.col('bus_forecast_reconciled').sum().alias('total_forecast'),
            (pl.col('pd') - pl.col('bus_forecast_reconciled')).abs().sum().alias('abs_err_sum'),
            pl.col('pd').mean().alias('mean_load'),
            pl.col('pd').max().alias('max_load'),
            pl.col('pd').count().alias('n_obs'),
        ]).with_columns([
            pl.when(pl.col('total_actual') > MIN_LOAD_FOR_WMAPE)
            .then(pl.col('abs_err_sum') / pl.col('total_actual') * 100)
            .otherwise(None)
            .alias('wmape'),
        ]).sort('wmape',descending=True,nulls_last=True)
    )

    return per_bus

def main():
    print("="*70)
    print("STAGE 5: IDENTIFY PROBLEM BUSES")
    print("="*70)

    # Selection MUST use a fold that isn't the test year. We use the OOS val
    # forecasts (2024 H2) produced by 03_training_model.py → 05_disaggregate.py.
    # Picking problem buses from 2025 errors would be selection bias on the test
    # set: the specialized models we then train for those buses are evaluated on
    # the same 2025 — biasing their reported WMAPE downward.
    pred_files = sorted(PREDICTIONS_DIR.glob('task1_val_2024_Q*.parquet'))
    if not pred_files:
        print('ERROR: no validation-fold predictions found at '
              'predictions/task1_val_2024_Q*.parquet.')
        print('Run 03_training_model.py (emits task1_val_zone_forecast.parquet) and '
              'then 05_disaggregate.py (which disaggregates it to bus level).')
        return

    print(f"Selecting problem buses from {len(pred_files)} val-fold file(s): "
          f"{', '.join(f.name for f in pred_files)}")

    per_bus = evaluate_per_bus(pred_files)

    print("\nPer-bus WMAPE distribution:")
    pdf = per_bus.to_pandas()
    valid = pdf.dropna(subset=['wmape'])

    for q in [0.5,0.75,0.9,0.95,0.99]:
        pct = valid['wmape'].quantile(q)
        print(f'{int(q*100)}th pct:{pct:.2f}%')

    print(f"\nBuses with WMAPE > {PROBLEM_THRESHOLD_WMAPE}%: {(valid['wmape'] > PROBLEM_THRESHOLD_WMAPE).sum():,}")
    print(f"Buses with WMAPE > 25%: {(valid['wmape'] > 25).sum():,}")
    print(f"Buses with WMAPE > 50%: {(valid['wmape'] > 50).sum():,}")

    # Selection rule:
    #   1. WMAPE above the problem threshold (default 15%)
    #   2. mean_load above MIN_LOAD_FOR_SELECTION (excludes noise buses)
    #   3. Rank by mean_load DESC then WMAPE DESC: spend specialized-model budget
    #      on the buses whose absolute error contribution is largest, not on
    #      tiny buses with sky-high percentage errors.
    pre_filter = (valid['wmape'] > PROBLEM_THRESHOLD_WMAPE)
    print(f"  Candidates with WMAPE > {PROBLEM_THRESHOLD_WMAPE}%: {pre_filter.sum():,}")
    print(f"  ... of which mean_load >= {MIN_LOAD_FOR_SELECTION} MW: "
          f"{(pre_filter & (valid['mean_load'] >= MIN_LOAD_FOR_SELECTION)).sum():,}")
    print(f"  ... of which mean_load <  {MIN_LOAD_FOR_SELECTION} MW (excluded as noise): "
          f"{(pre_filter & (valid['mean_load'] <  MIN_LOAD_FOR_SELECTION)).sum():,}")

    problem = (
        valid[(valid['wmape'] > PROBLEM_THRESHOLD_WMAPE)
              & (valid['mean_load'] >= MIN_LOAD_FOR_SELECTION)]
        .sort_values(['mean_load', 'wmape'], ascending=[False, False])
        .head(MAX_PROBLEM_BUSES)
    )

    print(f"\nSelected {len(problem):,} problem buses for specialized models "
          f"(cap = {MAX_PROBLEM_BUSES})")
    print(f"  Total mean load: {problem['mean_load'].sum():.1f} MW")

    pl.DataFrame(problem).write_parquet(DATA_DIR / "problem_buses.parquet")
    per_bus.write_parquet(DATA_DIR / 'per_bus_analysis.parquet')

    print(f"\nSaved -> data/problem_buses.parquet")

    print("\nTop 10 problem buses:")
    print(problem[["bus_unique_id", "zone_name", "mean_load", "wmape"]].head(10).to_string(index=False))

    print("\n" + "="*70)
    print("STAGE 5 COMPLETE — next step: python 06_specialized_models.py")
    print("="*70)
 
 
if __name__ == "__main__":
    main()