"""
Stream-convert NEON CSVs to parquet: one-by-one, deleting CSVs as we go.
Constant memory usage (~100 MB per CSV). Groups by product prefix.
"""
import pandas as pd
from pathlib import Path
import json, time, sys

neon_dir = Path('data/raw/neon_aquatic')
LOG_EVERY = 100  # Print progress every N files

products = ['DP1.20288.001', 'DP1.20042.001', 'DP1.20264.001', 'DP1.20016.001']

total_rows = 0
total_freed_mb = 0

for prod in products:
    csvs = sorted(neon_dir.glob(f'{prod}*.csv'))
    if not csvs:
        print(f'{prod}: no CSVs found', flush=True)
        continue

    prod_dir = neon_dir / f'shards_{prod}'
    prod_dir.mkdir(exist_ok=True)

    already_done = set(f.stem for f in prod_dir.glob('*.parquet'))
    todo = [f for f in csvs if f.stem not in already_done]
    print(f'{prod}: {len(csvs)} CSVs, {len(todo)} to convert', flush=True)

    prod_rows = 0
    prod_freed = 0
    for i, csv_path in enumerate(todo):
        try:
            df = pd.read_csv(csv_path, low_memory=False, dtype=str)
            if len(df) == 0:
                csv_path.unlink()
                continue
            out = prod_dir / f'{csv_path.stem}.parquet'
            df.to_parquet(out, index=False, compression='snappy')
            freed = csv_path.stat().st_size / 1e6
            prod_freed += freed
            prod_rows += len(df)
            csv_path.unlink()
        except Exception as e:
            print(f'  Error {csv_path.name}: {e}', flush=True)

        if (i + 1) % LOG_EVERY == 0:
            print(f'  {prod}: {i+1}/{len(todo)} done, {prod_rows:,} rows, {prod_freed:.0f} MB freed', flush=True)
            sys.stdout.flush()

    total_rows += prod_rows
    total_freed_mb += prod_freed
    print(f'{prod} DONE: {prod_rows:,} rows, {prod_freed:.0f} MB freed', flush=True)

print(f'\nAll done: {total_rows:,} rows, {total_freed_mb/1024:.1f} GB freed', flush=True)
json.dump({'rows': total_rows, 'freed_gb': total_freed_mb/1024},
          open('data/raw/neon_compress_status.json', 'w'), indent=2)
