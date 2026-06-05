"""
Download HydroLAKES + LakeATLAS global lake database.
HydroLAKES: 1.4M lakes with polygon geometry and basic stats
LakeATLAS: Extended WQ attributes for lakes
Source: https://www.hydrosheds.org/products/hydrolakes
"""
import os, sys, json, time, requests, zipfile
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('hydrolakes')

OUT_DIR = Path('data/raw/hydrolakes')
OUT_DIR.mkdir(parents=True, exist_ok=True)


DOWNLOAD_URLS = {
    # HydroLAKES - global lake polygon database
    'hydrolakes_polys_v10': 'https://97de0c9d-0-0-www-hydrosheds-org.a.run.app/uploads/product-files/HydroLAKES_polys_v10_shp.zip',
    # LakeATLAS - extended attributes
    'lakeatlas_global': 'https://zenodo.org/record/6386212/files/LakeATLAS_v10_csv.zip',
    # Hydrobasins
    'hydrobasins_global': 'https://97de0c9d-0-0-www-hydrosheds-org.a.run.app/uploads/product-files/hybas_lake_ar_lev08_v1c_shp.zip',
}

ZENODO_IDS = {
    'lakeatlas': '6386212',
    'hydrolakes_data': '6407091',
    'glowabo': '6481897',  # Global Water Bodies database
}

ALTERNATIVE_URLS = [
    # HydroSHEDS official
    'https://www.hydrosheds.org/geoserver/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=hydrosheds:hydrolakes_polys&outputFormat=csv&maxFeatures=100000',
    # Figshare HydroLAKES data table
    'https://figshare.com/ndownloader/files/9622510',
    # Zenodo - global lake database
    'https://zenodo.org/record/6386212/files/LakeATLAS_v10_csv.zip',
]


def download_zenodo(record_id, out_dir):
    """Download files from a Zenodo record"""
    log.info(f'Zenodo record {record_id}...')
    try:
        r = requests.get(f'https://zenodo.org/api/records/{record_id}', timeout=30)
        if r.status_code != 200:
            log.warning(f'  Record not found: {r.status_code}')
            return 0
        meta = r.json()
        title = meta.get('metadata', {}).get('title', '')
        log.info(f'  Title: {title}')
        files = meta.get('files', [])
        downloaded = 0
        for f in files:
            fname = f['key']
            fsize = f.get('size', 0)
            furl = f['links']['self']
            log.info(f'  File: {fname} ({fsize/1e6:.1f} MB)')
            if fsize > 3e9:
                log.info(f'  Skipping {fname} — too large')
                continue
            dest = out_dir / fname
            if dest.exists() and dest.stat().st_size == fsize:
                log.info(f'  Already complete: {fname}')
                downloaded += 1
                continue
            with requests.get(furl, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with open(dest, 'wb') as fp:
                    for chunk in resp.iter_content(chunk_size=2*1024*1024):
                        fp.write(chunk)
            log.info(f'  Saved: {fname}')
            downloaded += 1
        return downloaded
    except Exception as e:
        log.warning(f'  Failed: {e}')
        return 0


def process_lake_data():
    """Convert downloaded files to parquet for easy loading"""
    csvs = list(OUT_DIR.glob('**/*.csv'))
    zips = list(OUT_DIR.glob('**/*.zip'))

    records = []
    for zpath in zips:
        log.info(f'Processing zip: {zpath.name}')
        try:
            with zipfile.ZipFile(zpath) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith('.csv') or n.endswith('.dbf')]
                log.info(f'  Contents: {csv_names[:5]}')
                for name in csv_names:
                    try:
                        with zf.open(name) as f:
                            df = pd.read_csv(f, low_memory=False)
                            records.append({'file': name, 'rows': len(df), 'cols': list(df.columns[:8])})
                            log.info(f'  {name}: {len(df):,} rows, cols: {list(df.columns[:5])}')
                            out = OUT_DIR / f'{Path(name).stem}.parquet'
                            df.to_parquet(out, index=False)
                            log.info(f'  → {out.name} ({out.stat().st_size/1e6:.1f} MB)')
                    except Exception as e:
                        log.warning(f'  Could not read {name}: {e}')
        except Exception as e:
            log.warning(f'Could not open {zpath}: {e}')

    return records


def try_hydrosheds_direct():
    """Try direct download from HydroSHEDS website"""
    log.info('Trying HydroSHEDS direct URLs...')
    urls = [
        # HydroLAKES data tables (CSV attributes without geometry)
        ('hydrolakes_table.csv',
         'https://www.hydrosheds.org/images/inpages/HydroLAKES_polys_v10_csv.zip'),
        # LakeATLAS
        ('lakeatlas.zip',
         'https://zenodo.org/record/6386212/files/LakeATLAS_v10_csv.zip'),
    ]
    for fname, url in urls:
        dest = OUT_DIR / fname
        if dest.exists() and dest.stat().st_size > 1e6:
            log.info(f'  Already have: {fname}')
            continue
        try:
            log.info(f'  Trying: {url}')
            r = requests.get(url, stream=True, timeout=120,
                           headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=2*1024*1024):
                        f.write(chunk)
                log.info(f'  Saved: {fname} ({dest.stat().st_size/1e6:.1f} MB)')
            else:
                log.warning(f'  HTTP {r.status_code}: {url}')
        except Exception as e:
            log.warning(f'  Failed {url}: {e}')


def search_zenodo_lakes():
    """Search Zenodo for additional lake databases"""
    log.info('Searching Zenodo for global lake data...')
    queries = ['global lake water quality', 'HydroLAKES', 'lake database global']
    for q in queries:
        try:
            r = requests.get('https://zenodo.org/api/records', timeout=30, params={
                'q': q, 'sort': 'mostrecent', 'size': 5, 'type': 'dataset',
                'access_right': 'open'
            })
            if r.status_code == 200:
                for h in r.json().get('hits', {}).get('hits', []):
                    title = h.get('metadata', {}).get('title', '')
                    recid = h.get('id', '')
                    files = h.get('files', [])
                    sz = sum(f.get('size', 0) for f in files) / 1e6
                    log.info(f'  [{recid}] {title[:70]} ({sz:.0f} MB)')
        except Exception as e:
            log.warning(f'  Search failed: {e}')


if __name__ == '__main__':
    log.info('=== HydroLAKES / LakeATLAS Download ===')
    log.info(f'Output: {OUT_DIR.absolute()}')

    # Try Zenodo records
    total_downloaded = 0
    for name, rid in ZENODO_IDS.items():
        n = download_zenodo(rid, OUT_DIR)
        total_downloaded += n

    # Try direct HydroSHEDS download
    try_hydrosheds_direct()

    # Search for more
    search_zenodo_lakes()

    # Process downloaded files
    records = process_lake_data()

    # Status
    files = [f for f in OUT_DIR.iterdir() if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    status = {
        'files': [f.name for f in files],
        'total_mb': total_size / 1e6,
        'processed_tables': records,
    }
    json.dump(status, open(OUT_DIR / 'download_status.json', 'w'), indent=2)
    log.info(f'Total: {total_size/1e6:.1f} MB across {len(files)} files')
    log.info('Done.')
