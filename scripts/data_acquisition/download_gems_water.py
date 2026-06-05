"""
Download GEMS/Water (GEMStat) global water quality data.
Primary: Zenodo bulk download (CC BY license)
Fallback: GEMStat REST API by country
"""
import os, sys, json, time, requests, zipfile, io
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('gems_water')

OUT_DIR = Path('data/raw/gems_water')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORDS = [
    # GEMS/Water Global Water Quality Database - try multiple known record IDs
    '7547235',
    '7547234',
    '5778959',
    '6350573',
]

GEMSTAT_API = 'https://gemstat.org/api/getdata/'

def try_zenodo(record_id):
    url = f'https://zenodo.org/api/records/{record_id}'
    log.info(f'Trying Zenodo record {record_id}...')
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False
        meta = r.json()
        title = meta.get('metadata', {}).get('title', '')
        log.info(f'  Title: {title}')
        if 'water' not in title.lower() and 'gems' not in title.lower() and 'gem' not in title.lower():
            log.info(f'  Skipping — not a water quality record')
            return False
        files = meta.get('files', [])
        log.info(f'  Files: {len(files)}')
        for f in files:
            fname = f['key']
            fsize = f.get('size', 0)
            furl = f['links']['self']
            log.info(f'  File: {fname} ({fsize/1e6:.1f} MB)')
            if fsize > 5e9:  # skip files >5 GB
                log.info(f'  Skipping {fname} — too large ({fsize/1e9:.1f} GB)')
                continue
            dest = OUT_DIR / fname
            if dest.exists() and dest.stat().st_size > 0:
                log.info(f'  Already exists: {fname}')
                continue
            log.info(f'  Downloading {fname}...')
            with requests.get(furl, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(dest, 'wb') as out:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        out.write(chunk)
            log.info(f'  Saved {fname}')
        return True
    except Exception as e:
        log.warning(f'  Zenodo {record_id} failed: {e}')
        return False


def try_gemstat_api():
    """Try the GEMStat REST API - sample by station/country"""
    log.info('Trying GEMStat REST API...')
    # GEMStat API: https://gemstat.org/api/getdata/?StationNb=...
    # Station list endpoint
    try:
        # Try to get station list
        r = requests.get('https://gemstat.org/api/getStations/', timeout=30,
                        params={'format': 'json'})
        if r.status_code == 200:
            stations = r.json()
            log.info(f'GEMStat stations: {len(stations)}')
            return stations
    except Exception as e:
        log.warning(f'GEMStat API failed: {e}')
    return []


def try_gemstat_zenodo_search():
    """Search Zenodo for GEMS Water datasets"""
    log.info('Searching Zenodo for GEMS Water...')
    try:
        r = requests.get('https://zenodo.org/api/records', timeout=30, params={
            'q': 'GEMS Water quality global',
            'sort': 'mostrecent',
            'size': 10,
            'type': 'dataset'
        })
        if r.status_code == 200:
            hits = r.json().get('hits', {}).get('hits', [])
            for h in hits:
                title = h.get('metadata', {}).get('title', '')
                recid = h.get('id', '')
                log.info(f'  Found: [{recid}] {title}')
            return hits
    except Exception as e:
        log.warning(f'Zenodo search failed: {e}')
    return []


def try_unep_gems_download():
    """Try direct download from UNEP GEMS/Water portal"""
    log.info('Trying UNEP GEMS/Water portal...')
    # The GEMS/Water data is accessible at:
    # https://www.unep.org/explore-topics/freshwater/what-we-do/monitoring-and-assessment/gems-water
    # But bulk download requires registration. Try the open data portal.
    urls_to_try = [
        'https://gemstat.org/wp-content/uploads/2023/01/GEMStat_DataDownload.zip',
        'https://data.unep.org/api/3/action/package_show?id=gems-water',
    ]
    for url in urls_to_try:
        try:
            r = requests.head(url, timeout=20, allow_redirects=True)
            log.info(f'  {url}: {r.status_code}')
            if r.status_code == 200:
                size = int(r.headers.get('content-length', 0))
                log.info(f'  Size: {size/1e6:.1f} MB')
                if size > 0 and size < 10e9:
                    dest = OUT_DIR / 'gemstat_bulk.zip'
                    log.info(f'  Downloading to {dest}...')
                    with requests.get(url, stream=True, timeout=300) as resp:
                        with open(dest, 'wb') as f:
                            for chunk in resp.iter_content(chunk_size=1024*1024):
                                f.write(chunk)
                    log.info(f'  Downloaded: {dest}')
                    return True
        except Exception as e:
            log.warning(f'  Failed {url}: {e}')
    return False


def try_hydrosheds_gems():
    """Try HydroSHEDS/GloFAS water quality associated data"""
    log.info('Trying alternative global WQ sources on Zenodo...')
    # Search for other large global water quality datasets
    queries = [
        'global river water quality monitoring stations',
        'global lake water quality database',
        'GEMStat water quality observations',
    ]
    found = []
    for q in queries:
        try:
            r = requests.get('https://zenodo.org/api/records', timeout=30, params={
                'q': q, 'sort': 'mostrecent', 'size': 5, 'type': 'dataset',
                'access_right': 'open'
            })
            if r.status_code == 200:
                hits = r.json().get('hits', {}).get('hits', [])
                for h in hits:
                    title = h.get('metadata', {}).get('title', '')
                    recid = h.get('id', '')
                    nfiles = len(h.get('files', []))
                    log.info(f'  [{recid}] {title[:80]} ({nfiles} files)')
                    found.append({'id': recid, 'title': title})
        except Exception as e:
            log.warning(f'  Search failed: {e}')
    return found


def process_downloaded():
    """Process any downloaded CSV/zip files into unified parquet"""
    csvs = list(OUT_DIR.glob('**/*.csv'))
    zips = list(OUT_DIR.glob('**/*.zip'))
    log.info(f'Processing {len(csvs)} CSVs, {len(zips)} ZIPs...')
    dfs = []
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                log.info(f'  {z.name}: {names[:5]}')
                for name in names:
                    if name.endswith('.csv'):
                        with zf.open(name) as f:
                            df = pd.read_csv(f, low_memory=False, nrows=100)
                            log.info(f'    {name}: {len(df)} rows, cols: {list(df.columns[:5])}')
        except Exception as e:
            log.warning(f'  Failed to process {z}: {e}')


if __name__ == '__main__':
    log.info('=== GEMS/Water Download ===')
    log.info(f'Output: {OUT_DIR.absolute()}')

    success = False

    # 1. Try known Zenodo record IDs
    for rid in ZENODO_RECORDS:
        if try_zenodo(rid):
            success = True
            break

    # 2. Search Zenodo
    if not success:
        results = try_gemstat_zenodo_search()
        # Try downloading any promising hits
        for hit in results[:3]:
            rid = str(hit.get('id', ''))
            if rid and try_zenodo(rid):
                success = True
                break

    # 3. Try direct download
    if not success:
        success = try_unep_gems_download()

    # 4. Try GEMStat API
    if not success:
        try_gemstat_api()

    # 5. Search for alternatives
    try_hydrosheds_gems()

    # Process whatever we got
    process_downloaded()

    # Save status
    files = list(OUT_DIR.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    status = {
        'success': success,
        'files': [f.name for f in files],
        'total_mb': total_size / 1e6
    }
    json.dump(status, open(OUT_DIR / 'download_status.json', 'w'), indent=2)
    log.info(f'Status: {status}')
    log.info('Done.')
