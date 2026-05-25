import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')

# Fix 3 — share growth multiplier bounds.
# Real buses can grow/shrink up to these factors year-on-year. Clipping prevents
# tiny-load buses from producing runaway multipliers from numerator/denominator noise.
#
# Tightening rationale (after first end-to-end run with [0.5, 3.0]):
#   The wider [0.5, 3.0] bounds let weather noise dominate the signal — a mild-Q1
#   year over a harsh-Q1 baseline produced spurious "shrinkage" of 15-40% on
#   ~1,500 stable buses, regressing hierarchical WMAPE by 4-6 pp on both tasks.
#   The 108 real fast-growers were helped, but the 1,500 noisy "shrinkers" hurt
#   the overall ~10x more.
#
#   New bounds [0.85, 1.75]:
#     • shrinkage capped at -15%   (filters Q1 weather noise; real shrinkage rare)
#     • growth   capped at +75%    (still captures most data-center growth;
#                                    extreme growers >1.75× will be under-helped
#                                    but their volume is too small to matter
#                                    vs. the noise we're removing)
GROWTH_CLIP_LO = 0.85  # was 0.5
GROWTH_CLIP_HI = 1.75  # was 3.0


def _compute_overall_fallback(files):
    """Bus's overall share-of-zone aggregated across the provided files.
    One row per bus."""
    return (
        pl.concat([pl.scan_parquet(f) for f in files])
        .group_by(['bus_unique_id', 'zone_name'])
        .agg([
            pl.col('pd').sum().alias('bus_total_all'),
            pl.col('zone_pd').sum().alias('zone_total_all'),
        ])
        .with_columns([
            pl.when(pl.col('zone_total_all') > 0)
            .then(pl.col('bus_total_all') / pl.col('zone_total_all'))
            .otherwise(0.0)
            .alias('fallback_share'),
        ])
        .collect()
    )


def compute_shares_for_quarter(year, quarter):
    print(f"\nComputing shares for {year} Q{quarter} using same-quarter historical data...")

    share_files = sorted((DATA_DIR / 'bus_shares').glob('shares_*.parquet'))

    # No-leakage rule: a forecast for (year, quarter) may only use share data
    # observed STRICTLY BEFORE that quarter. Combined with the same-quarter
    # restriction (preserves seasonal AC/heating load patterns), we keep files
    # where file_q == quarter AND file_year < year.
    #
    # Prior bug: `file_year < 2025` was hardcoded, so 2024 Q1 was being built
    # from 2022/2023/2024 Q1 — leaking the target quarter into itself.
    keep_files = []
    for f in share_files:
        parts = f.stem.split('_')                 # ['shares', 'YYYY', 'Q{n}']
        file_year = int(parts[1])
        file_q    = int(parts[2][1:])
        if file_q == quarter and file_year < year:
            keep_files.append(f)

    if not keep_files:
        print(f"  WARNING: no prior same-quarter data available for {year} Q{quarter} — skipping")
        return None

    # Chronological so keep_files[-1] is the most recent historical year.
    keep_files.sort(key=lambda f: int(f.stem.split('_')[1]))
    print(f"  Using {len(keep_files)} historical Q{quarter} file(s): "
          f"{', '.join(f.stem for f in keep_files)}")

    df = pl.concat([pl.scan_parquet(f) for f in keep_files])

    df = df.with_columns([
        pl.col('timestamp').dt.weekday().alias('dow'),
        pl.col('he').alias('hod')
    ])

    shares = (
        df.group_by([
            'bus_unique_id','zone_name','dow','hod'
        ]).agg([
            pl.col('pd').sum().alias('bus_total'),
            pl.col('zone_pd').sum().alias('zone_total'),
            pl.col('pd').count().alias('n_obs'),
        ]).with_columns([
            pl.when(pl.col('zone_total') > 0).then(pl.col('bus_total') / pl.col('zone_total'))
            .otherwise(0.0).alias('share'),
        ]).collect()
    )

    fallback = _compute_overall_fallback(keep_files)

    shares = shares.join(
        fallback.select(['bus_unique_id','fallback_share']),
        on='bus_unique_id', how='left',
    )

    shares = shares.with_columns([
        ((pl.col('share')*pl.col('n_obs') + pl.col('fallback_share')*20)
         / (pl.col('n_obs')+20)).alias('share_smoothed'),
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # Fix 3 — Growth multiplier.
    # A bus that grew (or shrank) materially between the older window and the
    # most recent historical year is unlikely to settle back to its long-run
    # mean. The Bayesian-smoothed share above is centered on the long-run mean
    # and will systematically under-predict growing buses (data centers, new
    # industrial sites) and over-predict shrinking ones.
    #
    # Multiplier = (share averaged over the most recent historical year)
    #              / (share averaged over the older historical years).
    # Clipped to [GROWTH_CLIP_LO, GROWTH_CLIP_HI] so noise on small-load buses
    # can't blow up the share. Applied multiplicatively to share_smoothed.
    # ─────────────────────────────────────────────────────────────────────────
    if len(keep_files) >= 2:
        recent_share = _compute_overall_fallback([keep_files[-1]]).select([
            'bus_unique_id',
            pl.col('fallback_share').alias('recent_share'),
        ])
        older_share  = _compute_overall_fallback(keep_files[:-1]).select([
            'bus_unique_id',
            pl.col('fallback_share').alias('older_share'),
        ])

        growth = (
            recent_share.join(older_share, on='bus_unique_id', how='inner')
            .with_columns([
                # Guard against div-by-zero on the older window
                pl.when(pl.col('older_share') > 1e-6)
                .then((pl.col('recent_share') / pl.col('older_share'))
                      .clip(GROWTH_CLIP_LO, GROWTH_CLIP_HI))
                .otherwise(1.0)
                .alias('growth_mult'),
            ])
            .select(['bus_unique_id', 'growth_mult'])
        )

        shares = (
            shares.join(growth, on='bus_unique_id', how='left')
            .with_columns(pl.col('growth_mult').fill_null(1.0))
            .with_columns([
                # Both share_smoothed and fallback_share are growth-adjusted so
                # downstream consumers (05_disaggregate.py Tier B fallback) get
                # a consistent, growth-aware estimate regardless of which tier
                # they land on.
                (pl.col('share_smoothed') * pl.col('growth_mult')).alias('share_smoothed'),
                (pl.col('fallback_share')  * pl.col('growth_mult')).alias('fallback_share'),
            ])
        )

        # Diagnostic: how many buses moved materially?
        bus_growth = shares.unique(subset=['bus_unique_id'])
        n_growing  = bus_growth.filter(pl.col('growth_mult') > 1.10).height
        n_shrinking = bus_growth.filter(pl.col('growth_mult') < 0.90).height
        n_stable   = bus_growth.height - n_growing - n_shrinking
        print(f"  Growth multiplier — growing >+10%: {n_growing:,}  "
              f"shrinking <-10%: {n_shrinking:,}  stable: {n_stable:,}")
    else:
        shares = shares.with_columns(pl.lit(1.0).alias('growth_mult'))
        print(f"  Only 1 historical year — growth multiplier set to 1.0 for all buses")

    print(f"  Computed for {shares['bus_unique_id'].n_unique():,} buses")
    print(f"  Total share rows: {shares.height:,}")

    return shares

def main():
    print('='*70)
    print("STAGE 3: COMPUTE QUARTERLY TARGET SHARES")
    print("="*70)
    
    out_dir = DATA_DIR / 'bus_shares_by_target_quarter'
    out_dir.mkdir(exist_ok=True)

    targets = [(2024,q) for q in [1,2,3,4]] + [(2025,q) for q in [1,2,3,4]]

    for year,q in targets:
        shares = compute_shares_for_quarter(year,q)
        if shares is None:
            continue
        out_path = out_dir / f"shares_for_{year}_Q{q}.parquet"
        shares.write_parquet(out_path)
        print(f"  Saved → {out_path.name}")
 
    print("\n" + "="*70)
    print("STAGE 3 COMPLETE — next step: python 04_disaggregate.py")
    print("="*70)
 
 
if __name__ == "__main__":
    main()