import polars as pl
from pathlib import Path

BUS_Files = [
    "data/bus_load_2022.parquet",
    "data/bus_load_2023.parquet",
    "data/bus_load_2024.parquet",
    "data/bus_load_2025.parquet",
]

ZONE_Files = [
    "data/zone_load_2022.parquet",
    "data/zone_load_2023.parquet",
    "data/zone_load_2024.parquet",
    "data/zone_load_2025.parquet",
]

def combine_streaming(input_files, output_path, label):
    
    print(f"\n{'='*60}")
    print(f"Combining {label}")
    print(f"{'='*60}")

    for f in input_files:
        size_mb = Path(f).stat().st_size / 1e6
        print(f"Input: {f} ({size_mb:.0f} MB)")
    
    combined = pl.concat(
        [pl.scan_parquet(f) for f in input_files],
        how='vertical_relaxed'
    )

    print(f"\n Streaming write to {output_path}...")
    combined.sink_parquet(output_path, compression='snappy', statistics=True)
    out_size = Path(output_path).stat().st_size / 1e6

    print(f"Output: {output_path} ({out_size:.0f} MB)")

def verify(path,label):
    print(f"\n Verifying {label}...")
    lf = pl.scan_parquet(path)
    stats = lf.select([
        pl.len().alias('n_rows'),
        pl.col('date').min().alias('min_date'),
        pl.col('date').max().alias('max_date'),
        pl.col("date").n_unique().alias("unique_dates"),
    ]).collect()

    print(f"Rows:{stats['n_rows'][0]:,}")
    print(f"Date range:{stats['min_date'][0]} → {stats['max_date'][0]}")
    print(f"Unique dates:{stats['unique_dates'][0]:,}")


# def check_duplicates(path, label, group_cols):
#     print(f"\n  Checking {label} for duplicates...")
#     n_total = pl.scan_parquet(path).select(pl.len()).collect().item()
#     n_unique = (
#         pl.scan_parquet(path)
#         .select(group_cols)
#         .unique()
#         .select(pl.len())
#         .collect()
#         .item()
#     )
#     duplicates = n_total - n_unique
#     print(f"    Total: {n_total:,}  Unique: {n_unique:,}  Duplicates: {duplicates:,}")

def main():
    for f in BUS_Files + ZONE_Files:
        if not Path(f).exists():
            print(f"Missing: {f}")
            print("Edit BUS_FILES / ZONE_FILES at the top of this script.")
            return
        
    Path('data').mkdir(exist_ok=True)

    combine_streaming(BUS_Files, "data/bus_load.parquet", "BUS LOAD")
    verify("data/bus_load.parquet", "bus load")
    # check_duplicates("data/bus_load.parquet", "bus load",
    #                  ["bus_unique_id", "date", "he"])
    
    combine_streaming(ZONE_Files, "data/zone_load.parquet", "ZONE LOAD")
    verify("data/zone_load.parquet", "zone load")
    # check_duplicates("data/zone_load.parquet", "zone load",
    #                  ["zone_name", "date", "he"])
 
    print("\n" + "="*60)
    print("DONE — next step: python 00_pull_weather.py")
    print("="*60)
 
 
if __name__ == "__main__":
    main()

