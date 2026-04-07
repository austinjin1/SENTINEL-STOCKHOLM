"""
Expand NEON to all 34 aquatic sites + all WQ products.
Also: Canada ECCC, USGS discrete WQ, more WQP, GBIF aquatic species,
HydroLAKES via HydroSHEDS, and global river databases.
"""
import os, sys, json, time, requests, zipfile, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('expand')

BASE = Path('data/raw')
BASE.mkdir(exist_ok=True)

NEON_BASE = 'https://data.neonscience.org/api/v0'
WQP_BASE = 'https://www.waterqualitydata.us/data'
NWIS_BASE = 'https://waterservices.usgs.gov/nwis'


# ─────────────────────────────────────────────────────
# NEON - all 34 sites, 6 WQ products
# ─────────────────────────────────────────────────────
NEON_PRODUCTS = [
    'DP1.20288.001',  # Chemical properties of surface water (grab samples)
    'DP1.20093.001',  # Nitrate in surface water (continuous)
    'DP1.20190.001',  # Water quality (sonde - continuous)
    'DP1.20264.001',  # Temperature in surface water
    'DP1.20042.001',  # Stream discharge
    'DP1.20016.001',  # Reaeration
    'DP1.20048.001',  # Stream morphology
    'DP1.20033.001',  # Groundwater levels
]

def neon_get_sites_for_product(prod_code):
    try:
        r = requests.get(f'{NEON_BASE}/products/{prod_code}', timeout=20)
        if r.status_code == 200:
            return r.json().get('data', {}).get('siteCodes', [])
    except:
        pass
    return []

def neon_download_site_month(prod_code, site_code, month, out_dir):
    """Download one NEON site-month, return rows saved"""
    url = f'{NEON_BASE}/data/{prod_code}/{site_code}/{month}'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return 0
        files_data = r.json().get('data', {}).get('files', [])
        rows = 0
        for finfo in files_data:
            name = finfo.get('name', '')
            # Only grab sensor/chemistry CSV files (skip readme, variables, etc.)
            if not name.endswith('.csv'):
                continue
            if any(skip in name for skip in ['readme', 'variables', 'validation', 'sensor_positions']):
                continue
            furl = finfo.get('url', '')
            dest = out_dir / f'{prod_code}_{site_code}_{month}_{name}'
            if dest.exists() and dest.stat().st_size > 0:
                return -1  # Already done
            r2 = requests.get(furl, timeout=60)
            if r2.status_code == 200 and len(r2.content) > 100:
                with open(dest, 'wb') as f:
                    f.write(r2.content)
                try:
                    df = pd.read_csv(io.BytesIO(r2.content), low_memory=False)
                    rows += len(df)
                except:
                    pass
        return rows
    except Exception as e:
        return 0

def expand_neon():
    out_dir = BASE / 'neon_aquatic'
    out_dir.mkdir(exist_ok=True)
    log.info('=== NEON Expansion ===')

    total_rows = 0
    tasks = []

    for prod_code in NEON_PRODUCTS[:6]:
        sites = neon_get_sites_for_product(prod_code)
        log.info(f'  {prod_code}: {len(sites)} sites')
        for site in sites:
            site_code = site.get('siteCode', '')
            months = site.get('availableMonths', [])
            # Get last 24 months (most recent data)
            for month in months[-24:]:
                tasks.append((prod_code, site_code, month))

    log.info(f'  Total tasks: {len(tasks)} site-month downloads')

    # Run with thread pool (network-bound)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(neon_download_site_month, prod, site, month, out_dir): (prod, site, month)
            for prod, site, month in tasks
        }
        completed = 0
        for future in as_completed(futures):
            rows = future.result()
            if rows > 0:
                total_rows += rows
            completed += 1
            if completed % 50 == 0:
                log.info(f'  NEON: {completed}/{len(tasks)} tasks, ~{total_rows:,} rows so far')

    # Consolidate all CSVs to parquet by product
    log.info('  Consolidating NEON CSVs...')
    for prod_code in NEON_PRODUCTS[:6]:
        csvs = list(out_dir.glob(f'{prod_code}*.csv'))
        if not csvs:
            continue
        dfs = []
        for csv in csvs:
            try:
                df = pd.read_csv(csv, low_memory=False)
                df['neon_product'] = prod_code
                df['source_file'] = csv.name
                dfs.append(df)
            except:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            out_path = out_dir / f'neon_{prod_code}_all.parquet'
            combined.to_parquet(out_path, index=False)
            log.info(f'  Consolidated {prod_code}: {len(combined):,} rows → {out_path.name}')

    log.info(f'  NEON total: ~{total_rows:,} rows')
    return total_rows


# ─────────────────────────────────────────────────────
# Canada ECCC water quality (via WQP - they're a partner)
# ─────────────────────────────────────────────────────
def download_canada_wq():
    out_dir = BASE / 'canada_wq'
    out_dir.mkdir(exist_ok=True)
    log.info('=== Canada Water Quality (ECCC via WQP) ===')

    # ECCC submits to WQP as provider
    # Also available via open.canada.ca
    canada_urls = {
        # Canadian Open Data water quality
        'eccc_freshwater': 'https://dd.weather.gc.ca/hydrometric/csv/',
        # Water quality monitoring stations
        'eccc_stations_csv': 'https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/WQM_stations_list_e.csv',
        # Federal open data - water quality data
        'canada_wq_opendata': 'https://open.canada.ca/data/api/action/package_show?id=canadian-aquatic-monitoring-survey',
    }

    total = 0
    # Try Environment Canada water quality via REMS
    rems_url = 'https://data-donnees.ec.gc.ca/data/substances/monitor/canada-aquatic-ecosystems-water-quality-monitoring-data-collected-by-environment-and-climate-change-canada/'
    log.info(f'  Checking ECCC REMS: {rems_url}')
    try:
        r = requests.head(rems_url, timeout=15, allow_redirects=True)
        log.info(f'  REMS: HTTP {r.status_code}')
    except Exception as e:
        log.warning(f'  REMS failed: {e}')

    # WQP direct query for Canadian data
    log.info('  WQP query for Canada (provider=ECCC)...')
    try:
        r = requests.get(f'{WQP_BASE}/Result/search', timeout=60, params={
            'countrycode': 'CA',
            'mimeType': 'csv',
            'dataProfile': 'narrowResult',
            'sorted': 'no',
        }, stream=True)
        count = r.headers.get('Total-Result-Count', '0')
        log.info(f'  Canada WQP: {count} records available')
        if int(count) > 0:
            content = b''
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                content += chunk
                if len(content) > 500e6:  # cap 500 MB
                    break
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
            df.to_parquet(out_dir / 'canada_wqp.parquet', index=False)
            log.info(f'  Canada WQP saved: {len(df):,} rows')
            total += len(df)
    except Exception as e:
        log.warning(f'  Canada WQP failed: {e}')

    # Try Australian BOM via WQP (they also contribute)
    log.info('  WQP for Mexico/PR/VI...')
    for country in ['MX', 'PR', 'VI']:
        try:
            r = requests.get(f'{WQP_BASE}/Result/search', timeout=30, params={
                'countrycode': country,
                'mimeType': 'csv',
                'dataProfile': 'narrowResult',
                'sorted': 'no',
            })
            count = r.headers.get('Total-Result-Count', '0')
            log.info(f'  {country} WQP: {count} records')
        except Exception as e:
            log.warning(f'  {country}: {e}')

    return total


# ─────────────────────────────────────────────────────
# USGS NWIS discrete water quality (separate from sensor time series)
# ─────────────────────────────────────────────────────
def download_usgs_discrete_wq():
    out_dir = BASE / 'usgs_nwis_expanded'
    out_dir.mkdir(exist_ok=True)
    log.info('=== USGS Discrete Water Quality ===')

    total = 0
    # USGS water quality discrete samples via WQP (USGS is a partner)
    # These are grab samples, different from our continuous sensor data
    param_groups = {
        'nutrients': '00600,00605,00608,00613,00618,00631,62855',
        'metals': '01046,01049,01051,01056,01060,01065,01080,01090,01095',
        'organics': '32210,32211,50050,32209',
        'microbial': '31501,31625,50468,61213',
        'physical': '00010,00095,00300,00400,00600,63680',
    }

    for group, pcodes in param_groups.items():
        out_path = out_dir / f'usgs_wq_{group}.parquet'
        if out_path.exists():
            existing = len(pd.read_parquet(out_path))
            log.info(f'  Already have {group}: {existing:,} rows')
            total += existing
            continue

        log.info(f'  USGS discrete {group} (pCodes: {pcodes[:30]}...)')
        try:
            r = requests.get(f'{WQP_BASE}/Result/search', timeout=120, params={
                'organization': 'USGS',
                'pCode': pcodes,
                'startDateLo': '01-01-2010',
                'startDateHi': '12-31-2023',
                'mimeType': 'csv',
                'dataProfile': 'narrowResult',
                'sorted': 'no',
            }, stream=True)
            count = r.headers.get('Total-Result-Count', '0')
            log.info(f'    Available: {count} records')

            content = b''
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                content += chunk
                if len(content) > 300e6:
                    log.warning(f'    Capping {group} at 300 MB')
                    break

            if len(content) > 1000:
                df = pd.read_csv(io.BytesIO(content), low_memory=False)
                df.to_parquet(out_path, index=False)
                log.info(f'    Saved: {group} → {len(df):,} rows')
                total += len(df)
        except Exception as e:
            log.warning(f'    {group} failed: {e}')
        time.sleep(3)

    return total


# ─────────────────────────────────────────────────────
# HydroLAKES via direct HydroSHEDS website
# ─────────────────────────────────────────────────────
def download_hydrolakes():
    out_dir = BASE / 'hydrolakes'
    out_dir.mkdir(exist_ok=True)
    log.info('=== HydroLAKES ===')

    # HydroSHEDS provides direct downloads but requires form submission
    # Try alternative: HydroATLAS river network attributes (related dataset)
    urls = [
        # HydroRIVERS CSV attributes
        ('hydrorivers_attr.zip',
         'https://data.hydrosheds.org/file/hydrorivers/HydroRIVERS_v10_world_shp.zip'),
        # LakeATLAS - lake attributes
        ('lakeatlas.zip',
         'https://data.hydrosheds.org/file/hydrolakes/LakeATLAS_v10_csv.zip'),
        # HydroATLAS - basin attributes
        ('hydroatlas.zip',
         'https://data.hydrosheds.org/file/hydroatlas/BasinATLAS_v10_csv.zip'),
    ]

    for fname, url in urls:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 1e6:
            log.info(f'  Already have: {fname}')
            continue
        try:
            log.info(f'  Trying: {url}')
            r = requests.get(url, timeout=120, stream=True,
                           headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'})
            log.info(f'  HTTP {r.status_code}')
            if r.status_code == 200:
                size = 0
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=4*1024*1024):
                        f.write(chunk)
                        size += len(chunk)
                log.info(f'  Saved: {fname} ({size/1e6:.1f} MB)')
                # Extract CSV
                if fname.endswith('.zip') and size > 1e6:
                    try:
                        with zipfile.ZipFile(dest) as zf:
                            for name in zf.namelist():
                                if name.endswith('.csv'):
                                    zf.extract(name, out_dir)
                                    log.info(f'  Extracted: {name}')
                    except:
                        pass
        except Exception as e:
            log.warning(f'  {url} failed: {e}')

    # Try Figshare article API
    for article_id in ['6163614', '6194095', '7282430']:
        try:
            r = requests.get(f'https://api.figshare.com/v2/articles/{article_id}', timeout=15)
            if r.status_code == 200:
                data = r.json()
                title = data.get('title', '')
                files = data.get('files', [])
                log.info(f'  Figshare {article_id}: {title[:60]}')
                for f in files[:3]:
                    log.info(f'    File: {f.get("name","")} {f.get("size",0)/1e6:.1f} MB')
        except Exception as e:
            log.warning(f'  Figshare {article_id}: {e}')

    # Count parquet files
    csvs = list(out_dir.glob('**/*.csv'))
    total_rows = 0
    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
            out = out_dir / (csv.stem + '.parquet')
            df.to_parquet(out, index=False)
            total_rows += len(df)
            log.info(f'  CSV→parquet: {csv.name} → {len(df):,} rows')
        except:
            pass
    return total_rows


# ─────────────────────────────────────────────────────
# GBIF - freshwater species occurrences (bio signal)
# ─────────────────────────────────────────────────────
def download_gbif_freshwater():
    out_dir = BASE / 'gbif_freshwater'
    out_dir.mkdir(exist_ok=True)
    log.info('=== GBIF Freshwater Species ===')

    # GBIF API - search for aquatic taxa
    # Freshwater indicator species: macroinvertebrates, fish, algae
    taxa_keys = {
        # Ephemeroptera (mayflies - water quality bioindicators)
        'mayflies': '936',
        # Plecoptera (stoneflies)
        'stoneflies': '937',
        # Trichoptera (caddisflies)
        'caddisflies': '935',
        # Daphnia
        'daphnia': '2234882',
        # Freshwater fish (Actinopterygii in freshwater)
        'freshwater_fish': '204',
    }

    total = 0
    for taxon_name, taxon_key in list(taxa_keys.items())[:4]:
        out_path = out_dir / f'gbif_{taxon_name}.parquet'
        if out_path.exists():
            existing = len(pd.read_parquet(out_path))
            log.info(f'  Already have {taxon_name}: {existing:,}')
            total += existing
            continue

        log.info(f'  GBIF {taxon_name} (taxonKey={taxon_key})')
        all_records = []
        offset = 0
        limit = 300  # GBIF max per request
        max_records = 50000

        while offset < max_records:
            try:
                r = requests.get('https://api.gbif.org/v1/occurrence/search', timeout=30, params={
                    'taxonKey': taxon_key,
                    'hasCoordinate': 'true',
                    'hasGeospatialIssue': 'false',
                    'limit': limit,
                    'offset': offset,
                    'fields': 'key,species,decimalLatitude,decimalLongitude,eventDate,country,stateProvince,elevation,depth,waterBody',
                })
                if r.status_code != 200:
                    break
                data = r.json()
                results = data.get('results', [])
                if not results:
                    break
                all_records.extend(results)
                offset += len(results)
                if data.get('endOfRecords', True):
                    break
                time.sleep(0.3)
            except Exception as e:
                log.warning(f'  GBIF {taxon_name} offset {offset}: {e}')
                break

        if all_records:
            df = pd.DataFrame(all_records)
            df.to_parquet(out_path, index=False)
            log.info(f'  Saved: {taxon_name} → {len(df):,} records')
            total += len(df)

    return total


# ─────────────────────────────────────────────────────
# OpenAQ - air quality near water monitoring (proxy sensor)
# Actually: Let's get more WQP data - by state, more history
# ─────────────────────────────────────────────────────
def download_more_wqp_by_state():
    out_dir = BASE / 'epa_wqp_states'
    out_dir.mkdir(exist_ok=True)
    log.info('=== WQP by State (additional parameters) ===')

    # Parameters not in original download
    extra_char = [
        'Chlorophyll a',
        'Escherichia coli',
        'Fecal Coliform',
        'Specific conductance',
        'Turbidity',
        'Nitrate',
        'Ammonia',
        'Total dissolved solids',
        'Arsenic',
        'Lead',
        'Mercury',
    ]

    total = 0
    for char in extra_char[:6]:
        safe = char.replace(' ', '_').replace('-', '')[:20]
        out_path = out_dir / f'wqp_{safe}.parquet'
        if out_path.exists():
            existing = len(pd.read_parquet(out_path))
            log.info(f'  Already have {char}: {existing:,}')
            total += existing
            continue
        try:
            log.info(f'  WQP characteristic: {char}')
            r = requests.get(f'{WQP_BASE}/Result/search', timeout=120,
                           stream=True, params={
                               'characteristicName': char,
                               'startDateLo': '01-01-2015',
                               'mimeType': 'csv',
                               'dataProfile': 'narrowResult',
                               'sorted': 'no',
                           })
            count = r.headers.get('Total-Result-Count', '0')
            log.info(f'    {char}: {count} available')

            content = b''
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                content += chunk
                if len(content) > 200e6:
                    break

            if len(content) > 10000:
                df = pd.read_csv(io.BytesIO(content), low_memory=False)
                df.to_parquet(out_path, index=False)
                log.info(f'    Saved: {char} → {len(df):,} rows')
                total += len(df)
        except Exception as e:
            log.warning(f'    {char} failed: {e}')
        time.sleep(5)

    return total


if __name__ == '__main__':
    log.info('=' * 60)
    log.info('SENTINEL-DB Expansion Round 2')
    log.info('=' * 60)

    results = {}

    log.info('\n--- NEON Aquatic (all sites) ---')
    results['neon_expanded'] = expand_neon()

    log.info('\n--- HydroLAKES ---')
    results['hydrolakes'] = download_hydrolakes()

    log.info('\n--- GBIF Freshwater Species ---')
    results['gbif'] = download_gbif_freshwater()

    log.info('\n--- Canada ECCC Water Quality ---')
    results['canada'] = download_canada_wq()

    log.info('\n--- USGS Discrete WQ ---')
    results['usgs_discrete'] = download_usgs_discrete_wq()

    log.info('\n--- WQP Additional Parameters ---')
    results['wqp_extra'] = download_more_wqp_by_state()

    log.info('=' * 60)
    log.info('FINAL SUMMARY:')
    for src, n in results.items():
        log.info(f'  {src}: {n:,} records')
    total = sum(results.values())
    log.info(f'  TOTAL NEW RECORDS: {total:,}')

    status = {'results': results, 'total': total}
    json.dump(status, open('data/raw/expansion_r2_status.json', 'w'), indent=2)
    log.info('Done.')
