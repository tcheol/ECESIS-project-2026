import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')
PREDICTION_DIR = Path('predictions')

def compute_metrics(y_true,y_pred,label=""):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred,dtype=np.float64)
    mask = y_true > 1.0
    mape  = (np.nanmean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan)
    mae   = np.nanmean(np.abs(y_true - y_pred))
    rmse  = np.sqrt(np.nanmean((y_true - y_pred) **2))
    wmape = np.nansum(np.abs(y_true - y_pred)) / max(np.nansum(y_true), 1e-6) * 100

    if label:
        print(f"\n{label}:")
        print(f"  WMAPE: {wmape:6.2f}%   ← primary")
        print(f"  MAPE:  {mape:6.2f}%")
        print(f"  MAE:   {mae:.3f} MW")
        print(f"  RMSE:  {rmse:.3f} MW")
    return {"mape": mape, "wmape": wmape, "mae": mae, "rmse": rmse}

def blend_forecasts(task):
    print(f"\n{'='*70}")
    print(f"BLENDING {task.upper()}")
    print("="*70)

    hier_files = sorted(PREDICTION_DIR.glob(f"{task}_2025_Q*.parquet"))
    hier = pl.concat([pl.read_parquet(f) for f in hier_files])
    print(f"Hierarchical: {hier.height:,} rows")

    spec_path = PREDICTION_DIR / f"{task}_specialized.parquet"
    problem_path = DATA_DIR / 'problem_buses.parquet'

    if spec_path.exists():
        spec = pl.read_parquet(spec_path)
        # Treat problem_buses.parquet as the source of truth for *who* gets
        # specialized — the spec parquet is just a cache of available
        # per-bus predictions. This lets us shrink MAX_PROBLEM_BUSES in 06
        # and re-blend without retraining 07. (If the parquet ever grew
        # past what 07 last trained, those missing buses fall through to
        # hierarchical, no error.)
        if problem_path.exists():
            pb = pl.read_parquet(problem_path)['bus_unique_id'].unique().to_list()
            before = spec.height
            spec = spec.filter(pl.col('bus_unique_id').is_in(pb))
            print(f"Specialized: {spec.height:,} rows "
                  f"(filtered from {before:,} via problem_buses.parquet, "
                  f"{len(pb):,} buses)")
        else:
            print(f"Specialized: {spec.height:,} rows "
                  f"(no problem_buses.parquet — using all rows from cache)")
        problem_buses = spec['bus_unique_id'].unique().to_list()
    else:
        spec = None
        problem_buses = []
    
    final = hier.with_columns([
        pl.col('bus_forecast_reconciled').alias('forecast'),
        pl.lit('hierarchical').alias('method'),
    ])

    if spec is not None and len(problem_buses) > 0:
        spec_lookup = spec.with_columns(
            pl.col('date').cast(pl.Date)
        ).select([
            'bus_unique_id','date','he',
            pl.col('bus_forecast_specialized').alias('specialized_fc'),
        ])

        final = final.join(
            spec_lookup, on=['bus_unique_id','date','he'], how='left',
        )

        final = final.with_columns([
            pl.when(pl.col('specialized_fc').is_not_null())
            .then(pl.col('specialized_fc'))
            .otherwise(pl.col('forecast'))
            .alias('forecast'),
            pl.when(pl.col('specialized_fc').is_not_null())
            .then(pl.lit('specialized'))
            .otherwise(pl.lit('hierarchical'))
            .alias('method')
        ])

        n_spec = (final['method'] == 'specialized').sum()
        print(f'Rows using specialized: {n_spec:,}')

    return final.select([
        'bus_unique_id','zone_name','date','he','timestamp','pd','forecast','method',
    ]).rename({'pd':'actual'})

def evaluate_final(final,task):
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION: {task.upper()}")
    print("="*70)

    actual = final['actual'].to_numpy()
    forecast = final['forecast'].to_numpy()
    compute_metrics(actual,forecast,label=f'{task.upper()} OVERALL (2025)')

    pdf = final.to_pandas()

    print('\nPer-zone WMAPE:')
    by_zone = pdf.groupby('zone_name',observed=True).apply(
        lambda g: np.sum(np.abs(g['actual'] - g['forecast'])) / max(g['actual'].sum(),1e-6) * 100
    ).sort_values()

    for zone,wmape in by_zone.items():
        print(f'{zone}: {wmape:6.2f}%')

    print('\nPer_month WMAPE')
    pdf['month'] = pd.to_datetime(pdf['timestamp']).dt.month
    by_month = pdf.groupby('month').apply(
        lambda g: np.sum(np.abs(g['actual'] - g['forecast'])) / max(g['actual'].sum(), 1e-6) * 100
    )

    for m,wmape in by_month.items():
        print(f"Month {m:2d}: {wmape:6.2f}%")

    print('\nWMAPE by method:')
    by_method = pdf.groupby('method').apply(
        lambda g: np.sum(np.abs(g['actual'] - g['forecast'])) / max(g['actual'].sum(), 1e-6) * 100
    )

    for method, wmape in by_method.items():
        n = (pdf['method'] == method).sum()
        print(f"{method}: {wmape:.2f}% ({n:,} rows)")

    print("\nTop 10 worst-predicted buses (by total absolute error):")
    bus_err = pdf.groupby(['bus_unique_id','zone_name']).apply(
        lambda g: pd.Series({
            'total_actual': g['actual'].sum(),
            'total_abs_err': np.sum(np.abs(g['actual'] - g['forecast'])),
            'wmape': np.sum(np.abs(g['actual'] - g['forecast'])) / max(g['actual'].sum(),1e-6) * 100,

        })
    ).reset_index().sort_values('total_abs_err',ascending=False)
    print(bus_err.head(10).to_string(index=False))

def write_submission_csv(final_df, task, model_name='my_model'):
    pdf = final_df.to_pandas()
    pdf['target_date'] = pd.to_datetime(pdf['date']).dt.date

    if task == 'task1':
        pdf['forecast_created_at'] = (
            pd.to_datetime(pdf['target_date']) - pd.Timedelta(days=1) + pd.Timedelta(minutes=1)
        )
    else:
        # Spec: forecast_created_at = first day of the *previous* month (for ALL
        # target days within target month M, FCA is the first day of month M-1).
        #
        # Bug being fixed: `target_ts - pd.offsets.MonthBegin(1)` returns the
        # most recent month start, which for target Feb 5 is Feb 1 — not Jan 1.
        # Only target dates that ARE the first day of a month happened to land
        # on the right value before. For all other 95% of rows, FCA was off by
        # a full month.
        target_ts = pd.to_datetime(pdf['target_date'])
        pdf['forecast_created_at'] = (
            (target_ts.dt.to_period('M') - 1).dt.to_timestamp()
        )

    submission = pd.DataFrame({
        'model_name': model_name,
        'forecast_created_at': pdf['forecast_created_at'],
        'target_date': pdf['target_date'],
        'he': pdf['he'].astype(int),
        'bus_id': pdf['bus_unique_id'],
        'zone_id': pdf['zone_name'].astype(str),
        'predict_pd': pdf['forecast'].clip(lower=0),
    })

    out = PREDICTION_DIR / f'{task}_submission.csv'
    submission.to_csv(out, index=False)
    print(f'  Saved submission → {out}  ({len(submission):,} rows)')


EXCLUDE_BUSES = ['HELIOSCR_345KV_1']

def main():
    for task in ['task1','task2']:
        final = blend_forecasts(task)
        final = final.filter(~pl.col('bus_unique_id').is_in(EXCLUDE_BUSES))
        out_path = PREDICTION_DIR / f"{task}_final.parquet"
        final.write_parquet(out_path)
        print(f"\n Saved final -> {out_path}")
        evaluate_final(final, task)
        write_submission_csv(final, task)

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nFinal outputs:")
    print("  predictions/task1_final.parquet")
    print("  predictions/task2_final.parquet")
    print("  predictions/task1_submission.csv")
    print("  predictions/task2_submission.csv")
 
 
if __name__ == "__main__":
    main()