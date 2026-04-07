"""
Download EU Waterbase water quality data via EEA DISCODATA API.
Covers 39 EU countries, ~60M+ physico-chemical measurements.
Table: Waterbase - Water Quality ICM (WISE_SOE_W)
API: https://discodata.eea.europa.eu/
"""
import os, sys, json, time, requests
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('eu_waterbase')

OUT_DIR = Path('data/raw/eu_waterbase')
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISCODATA_BASE = 'https://discodata.eea.europa.eu/sql'
# Max rows per request
PAGE_SIZE = 10000
# Target: up to 10M rows (manageable ~2 GB)
MAX_ROWS = 10_000_000


def query_discodata(sql, offset=0):
    """Execute a SQL query against DISCODATA API"""
    params = {
        'query': sql,
        'p': offset // PAGE_SIZE + 1,
        'nrOfHits': PAGE_SIZE,
    }
    try:
        r = requests.get(DISCODATA_BASE, params=params, timeout=60)
        if r.status_code == 200:
            data = r.json()
            return data.get('results', [])
        else:
            log.warning(f'DISCODATA {r.status_code}: {r.text[:200]}')
            return []
    except Exception as e:
        log.warning(f'DISCODATA error: {e}')
        return []


def get_table_list():
    """List available Waterbase tables"""
    sql = """SELECT table_name FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name LIKE '%WISE%'
             ORDER BY table_name"""
    results = query_discodata(sql)
    log.info(f'Tables: {results[:10]}')
    return results


def get_waterbase_physchemdata():
    """Download Waterbase physico-chemical data in chunks"""
    # Main physico-chemical observations table
    tables_to_try = [
        'WISE_SOE_W_PhysChemData',
        'Waterbase_v2021_1_T_WISE4_AggregatedData',
        'WISE_SOE_W_Observations',
        'v_waterbase_wise4_aggregated',
    ]

    for table in tables_to_try:
        log.info(f'Trying table: {table}')
        # Get count first
        count_sql = f'SELECT COUNT(*) as cnt FROM "{table}"'
        count_result = query_discodata(count_sql)
        if count_result:
            log.info(f'  Count result: {count_result}')
            break

    # Try the direct Waterbase download endpoint
    log.info('Trying EEA direct download API...')

    # EEA Data Service for Waterbase
    eea_endpoints = [
        # Waterbase Water Quality ICM 2021
        'https://www.eea.europa.eu/ds_resolveuid/waterbase-water-quality-icm-2',
        # WISE WFD aggregated data
        'https://cdr.eionet.europa.eu/help/wise/index_html',
    ]

    # Try downloading via the EEA bulk download
    # The Waterbase v2021.1 dataset is available as zip
    bulk_urls = [
        'https://sdi.eea.europa.eu/catalogue/srv/api/records/789c4b81-d839-4b0d-bd47-15f6d0a62a4d/attachments/Waterbase_v2021_1_csv.zip',
        'https://www.eea.europa.eu/data-and-maps/data/waterbase-water-quality-icm-2/download',
        'https://discodata.eea.europa.eu/download?query=SELECT+*+FROM+%22WISE_SOE_W_PhysChemData%22+LIMIT+1000000&filename=waterbase_wq.csv',
    ]

    for url in bulk_urls:
        try:
            log.info(f'Trying: {url}')
            r = requests.head(url, timeout=30, allow_redirects=True)
            log.info(f'  Status: {r.status_code}, Content-Type: {r.headers.get("content-type", "")}')
            if r.status_code == 200:
                size = int(r.headers.get('content-length', 0))
                log.info(f'  Size: {size/1e6:.1f} MB')
                if 0 < size < 5e9:  # < 5 GB
                    dest = OUT_DIR / 'waterbase_bulk.zip'
                    download_file(url, dest)
                    return True
        except Exception as e:
            log.warning(f'  Failed: {e}')

    return False


def download_discodata_paginated():
    """Download via paginated DISCODATA SQL API"""
    log.info('Downloading via paginated DISCODATA API...')

    # Try different query approaches
    queries = [
        ('physchem', 'SELECT * FROM "WISE_SOE_W_PhysChemData" ORDER BY 1'),
        ('aggregated', 'SELECT * FROM "Waterbase_v2021_1_T_WISE4_AggregatedData" ORDER BY 1'),
        ('obs', 'SELECT * FROM "WISE_SWB_Observations" ORDER BY 1'),
    ]

    for name, sql_base in queries:
        log.info(f'  Trying {name}...')
        all_rows = []
        offset = 0
        max_pages = MAX_ROWS // PAGE_SIZE

        for page in range(max_pages):
            sql = f'{sql_base} OFFSET {offset} ROWS FETCH NEXT {PAGE_SIZE} ROWS ONLY'
            rows = query_discodata(sql, offset)
            if not rows:
                log.info(f'  No more rows at offset {offset}')
                break
            all_rows.extend(rows)
            offset += len(rows)
            if page % 10 == 0:
                log.info(f'  {name}: {offset:,} rows downloaded')
            if len(rows) < PAGE_SIZE:
                break  # last page
            time.sleep(0.5)

        if all_rows:
            log.info(f'  Saving {len(all_rows):,} rows for {name}')
            df = pd.DataFrame(all_rows)
            out_path = OUT_DIR / f'waterbase_{name}.parquet'
            df.to_parquet(out_path, index=False)
            log.info(f'  Saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)')
            return len(all_rows)

    return 0


def try_eea_sparql():
    """Try EEA SPARQL endpoint for water quality data"""
    log.info('Trying EEA SPARQL...')
    sparql_url = 'https://semantic.eea.europa.eu/sparql'
    query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?dataset ?title WHERE {
        ?dataset a <http://www.w3.org/ns/dcat#Dataset> ;
                 <http://purl.org/dc/terms/title> ?title .
        FILTER(CONTAINS(LCASE(STR(?title)), "waterbase"))
    } LIMIT 20
    """
    try:
        r = requests.post(sparql_url, data={'query': query},
                         headers={'Accept': 'application/json'}, timeout=30)
        log.info(f'SPARQL: {r.status_code}')
        if r.status_code == 200:
            results = r.json()
            bindings = results.get('results', {}).get('bindings', [])
            for b in bindings:
                log.info(f"  {b.get('title', {}).get('value', '')}")
    except Exception as e:
        log.warning(f'SPARQL failed: {e}')


def download_file(url, dest, chunk_mb=4):
    """Download a file with progress logging"""
    log.info(f'Downloading {url} → {dest}')
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_mb*1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50*1024*1024) < chunk_mb*1024*1024:
                    log.info(f'  {downloaded/1e6:.0f} / {total/1e6:.0f} MB')
    log.info(f'  Complete: {dest} ({dest.stat().st_size/1e6:.1f} MB)')


def try_zenodo_eu_water():
    """Search Zenodo for EU water quality datasets"""
    log.info('Searching Zenodo for EU water datasets...')
    queries = ['EU Waterbase water quality', 'European water quality monitoring WISE']
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
                    total_size = sum(f.get('size', 0) for f in files)
                    log.info(f'  [{recid}] {title[:70]} ({total_size/1e6:.0f} MB)')
        except Exception as e:
            log.warning(f'Search failed: {e}')


if __name__ == '__main__':
    log.info('=== EU Waterbase Download ===')
    log.info(f'Output: {OUT_DIR.absolute()}')

    # Check what we already have
    existing = list(OUT_DIR.glob('*.parquet')) + list(OUT_DIR.glob('*.zip'))
    existing_rows = sum(len(pd.read_parquet(f)) for f in OUT_DIR.glob('*.parquet'))
    log.info(f'Existing: {len(existing)} files, ~{existing_rows:,} rows')

    # Try bulk download first
    success = get_waterbase_physchemdata()

    # Try paginated API
    if not success:
        rows = download_discodata_paginated()
        success = rows > 0

    # Try Zenodo alternatives
    try_zenodo_eu_water()

    # Try SPARQL
    try_eea_sparql()

    # Save status
    files = list(f for f in OUT_DIR.iterdir() if f.is_file())
    total_size = sum(f.stat().st_size for f in files)
    row_counts = {}
    for f in OUT_DIR.glob('*.parquet'):
        try:
            row_counts[f.name] = len(pd.read_parquet(f))
        except:
            pass

    status = {
        'success': success,
        'files': [f.name for f in files],
        'total_mb': total_size / 1e6,
        'row_counts': row_counts,
    }
    json.dump(status, open(OUT_DIR / 'download_status.json', 'w'), indent=2)
    log.info(f'Status: {json.dumps(status, indent=2)}')
    log.info('Done.')
