import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')

def reconcile_bus_vs_zone():
    print('='*70)
    print("RECONCILIATION: sum(bus_pd) vs provided zone_pd")
    print("="*70)

    print("\nAggregating buses for comparsion...")
    bus_agg = (
        pl.scan_parquet(DATA_DIR / "bus_load.parquet")
        .filter(pl.col('bus_type') == "LOAD")
        .filter(pl.col('pd').is_not_null())
        .filter(pl.col('zone_name').is_not_null())
        .group_by(["zone_name","date","he"])
        .agg([
            pl.col('pd').sum().alias('bus_sum_pd'),
            pl.col("pd").count().alias('n_buses'),
        ]).collect()
    )
    print(f"Aggregated to {bus_agg.height:,} zone-hours")
    print("\nLoading provided zone data...")
    zone = (
        pl.scan_parquet(DATA_DIR / "zone_load.parquet")
        .select(['zone_name','date','he',
                 pl.col('pd').alias('zone_pd_provided')
        ]).collect()
    )
    print(f"Loaded {zone.height:,} zone_hours")

    compare = bus_agg.join(zone, on=["zone_name","date",'he'], how='full', coalesce=True)

    compare = compare.with_columns([
        pl.when(pl.col('zone_pd_provided') > 0)
        .then(
            (pl.col('bus_sum_pd') - pl.col('zone_pd_provided')).abs()
            / pl.col('zone_pd_provided') * 100
        ).otherwise(None).alias('pct_diff')
    ])

    print("\nPer-zone reconciliation:")
    report_lines = ["Per-zone reconciliation", '='*40]

    by_zone = (
        compare.group_by('zone_name')
        .agg([
            pl.col('bus_sum_pd').mean().alias('bus_mean'),
            pl.col('zone_pd_provided').mean().alias('zone_mean'),
            pl.col('pct_diff').mean().alias('mean_diff'),
            pl.col('pct_diff').max().alias('max_diff'),
            pl.col('n_buses').mean().alias('avg_n_buses')
        ]).sort('zone_name')
    )

    _f = lambda v: v if v is not None else float('nan')
    for row in by_zone.iter_rows(named=True):
        line = (
            f"  {row['zone_name']:<10s} | "
            f"bus_sum={_f(row['bus_mean']):9.1f} | "
            f"zone={_f(row['zone_mean']):9.1f} | "
            f"diff={_f(row['mean_diff']):5.2f}% | "
            f"max={_f(row['max_diff']):5.2f}% | "
            f"avg_buses={_f(row['avg_n_buses']):.0f}"
        )
        print(line)
        report_lines.append(line)

    only_bus = compare.filter(pl.col('zone_pd_provided').is_null()).height
    only_zone = compare.filter(pl.col('bus_sum_pd').is_null()).height

    if only_bus > 0:
        msg = f"{only_bus:,} hours in bus data but not zone data"
        print(f"\n  {msg}")
        report_lines.append(msg)

    if only_zone > 0:
        msg = f"{only_zone:,} hours in zone data but not bus data"
        print(f"  {msg}")
        report_lines.append(msg)
    
    mean_diff = compare['pct_diff'].drop_nulls().mean()

    if mean_diff and mean_diff < 2:
         print(f"\nMean discrepancy {mean_diff:.2f}% — files agree well")
 
    (DATA_DIR / "reconciliation_report.txt").write_text("\n".join(report_lines))
    return mean_diff if mean_diff else 0


def compute_bus_shares():
    print('\n' + '='*70)
    print("STAGE 2B: COMPUTE QUARTERLY BUS LOAD FILES")
    print("="*70)

    out_dir = DATA_DIR / 'bus_shares'
    out_dir.mkdir(exist_ok=True)

    # Handle zone_pd column name (may be 'pd' or 'zone_pd')
    zone_pd_name = 'zone_pd' if 'zone_pd' in pl.scan_parquet(DATA_DIR / "zone_load.parquet").collect_schema().names() else 'pd'

    zone = (
        pl.scan_parquet(DATA_DIR / "zone_load.parquet")
        .select(['zone_name', 'date', 'he', pl.col(zone_pd_name).alias('zone_pd')])
    )

    bus = (
        pl.scan_parquet(DATA_DIR / "bus_load.parquet")
        .filter(pl.col('bus_type') == "LOAD")
        .filter(pl.col('pd').is_not_null())
    )

    joined = (
        bus.join(zone, on=['zone_name', 'date', 'he'], how='left')
        .with_columns([
            pl.col('date').cast(pl.Date).alias('_d'),
        ])
        .with_columns([
            (pl.col('_d').cast(pl.Datetime) +
             pl.duration(hours=pl.col('he') - 1)).alias('timestamp'),
            pl.col('_d').dt.year().alias('_year'),
            pl.col('_d').dt.quarter().alias('_quarter'),
        ])
        .drop('_d')
    )

    df = joined.collect()
    print(f"  Loaded {df.height:,} bus-hours across {df['bus_unique_id'].n_unique():,} buses")

    quarters = (
        df.select(['_year', '_quarter'])
        .filter(pl.col('_year').is_not_null() & pl.col('_quarter').is_not_null())
        .unique()
        .sort(['_year', '_quarter'])
    )
    print(f"  Splitting into {quarters.height} quarterly files...\n")

    for row in quarters.iter_rows(named=True):
        year, q = row['_year'], row['_quarter']
        subset = (
            df.filter((pl.col('_year') == year) & (pl.col('_quarter') == q))
            .drop(['_year', '_quarter'])
        )
        out_path = out_dir / f"shares_{year}_Q{q}.parquet"
        subset.write_parquet(out_path)
        print(f"  Q{q} {year}: {subset.height:,} rows → {out_path.name}")

    print(f"\n  Done — {quarters.height} files written to {out_dir}/")


def main():
    reconcile_bus_vs_zone()
    compute_bus_shares()


if __name__ == "__main__":
    main()
