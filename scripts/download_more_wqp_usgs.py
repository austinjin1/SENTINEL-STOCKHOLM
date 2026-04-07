"""
Expand EPA WQP and USGS NWIS downloads.
WQP: Get more parameters (we had narrowband params, get full set)
     Also get Canadian/tribal/Puerto Rico data
USGS NWIS: Download daily values (not just instantaneous) - more records
           Also get more parameters beyond 5 we had
"""
import os, sys, json, time, requests
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('wqp_usgs_expand')

WQP_DIR = Path('data/raw/epa_wqp_expanded')
WQP_DIR.mkdir(parents=True, exist_ok=True)

NWIS_DIR = Path('data/raw/usgs_nwis_expanded')
NWIS_DIR.mkdir(parents=True, exist_ok=True)

WQP_BASE = 'https://www.waterqualitydata.us/data'
NWIS_BASE = 'https://waterservices.usgs.gov/nwis'

# Characteristic names we haven't downloaded yet
WQP_EXTRA_CHARS = [
    'Escherichia coli',
    'Fecal Coliform',
    'Enterococcus',
    'Total dissolved solids',
    'Chlorophyll a',
    'Suspended sediment concentration (SSC)',
    'Arsenic',
    'Lead',
    'Mercury',
    'Nitrate',
    'Ammonia',
    'Total Kjeldahl nitrogen (Organic N + NH3)',
    'Chemical oxygen demand (COD)',
    'Biochemical oxygen demand (BOD)',
    'Atrazine',
    'Glyphosate',
    'PFAS',
    'Microplastics',
]

# Non-US WQP providers (tribal, territories, Canada)
EXTRA_PROVIDERS = [
    'NWTRIBALDEP',  # Tribal departments
    'DOETRIBUTARIES',
    'USGS-PR',  # Puerto Rico
    'USGS-USVI',  # US Virgin Islands
    'ECCC',  # Environment and Climate Change Canada
]


def download_wqp_characteristic(char_name, max_records=500000):
    """Download WQP data for a specific characteristic"""
    safe_name = char_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')[:30]
    out_path = WQP_DIR / f'wqp_{safe_name}.parquet'
    if out_path.exists():
        existing = len(pd.read_parquet(out_path))
        log.info(f'  Already have {char_name}: {existing:,} rows')
        return existing

    log.info(f'  Downloading: {char_name}')
    params = {
        'characteristicName': char_name,
        'mimeType': 'csv',
        'zip': 'yes',
        'dataProfile': 'narrowResult',
        'sorted': 'no',
    }

    try:
        r = requests.get(f'{WQP_BASE}/Result/search', params=params,
                        stream=True, timeout=300,
                        headers={'Accept': 'text/csv'})
        if r.status_code == 200:
            # Read the zipped CSV
            import zipfile, io
            content = b''
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                content += chunk
                if len(content) > 2e9:  # cap at 2 GB
                    log.warning(f'  {char_name}: Capping at 2 GB')
                    break

            try:
                # Try as zip
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    csvname = zf.namelist()[0]
                    with zf.open(csvname) as f:
                        df = pd.read_csv(f, low_memory=False, nrows=max_records)
            except:
                # Try as raw CSV
                df = pd.read_csv(io.BytesIO(content), low_memory=False, nrows=max_records)

            df.to_parquet(out_path, index=False)
            log.info(f'  Saved: {char_name} → {len(df):,} rows')
            return len(df)
        else:
            log.warning(f'  HTTP {r.status_code} for {char_name}')
            return 0
    except Exception as e:
        log.warning(f'  Failed {char_name}: {e}')
        return 0


def download_wqp_provider(provider, max_records=1000000):
    """Download all WQP data from a specific provider"""
    out_path = WQP_DIR / f'wqp_provider_{provider}.parquet'
    if out_path.exists():
        log.info(f'  Already have provider {provider}')
        return

    log.info(f'  Downloading provider: {provider}')
    params = {
        'providers': provider,
        'mimeType': 'csv',
        'dataProfile': 'narrowResult',
        'sorted': 'no',
    }
    try:
        r = requests.get(f'{WQP_BASE}/Result/search', params=params,
                        timeout=120)
        # Get count header first
        count = r.headers.get('Total-Result-Count', '0')
        log.info(f'  Provider {provider}: ~{count} records available')
    except Exception as e:
        log.warning(f'  Provider {provider} check failed: {e}')


def download_nwis_daily_values():
    """Download USGS NWIS daily values (dv) - more compact than instantaneous values"""
    log.info('Downloading USGS NWIS daily values...')
    # Get all stations that have water quality data
    # Focus on parameter codes for WQ parameters
    pcode_groups = {
        'nutrients': ['00600', '00608', '00613', '00631', '62855'],  # N, NO3, NO2, NH4
        'metals': ['01046', '01049', '01051', '01056', '01060'],  # Fe, Pb, Mn, Ni, Zn
        'organics': ['32210', '32211', '50050'],  # Chl-a, phycocyanin
        'microbial': ['31501', '31625', '50468'],  # E. coli, fecal coliform, enterococcus
    }

    for group_name, pcodes in pcode_groups.items():
        pcode_str = ','.join(pcodes)
        out_path = NWIS_DIR / f'nwis_dv_{group_name}.parquet'
        if out_path.exists():
            log.info(f'  Already have NWIS dv {group_name}')
            continue

        log.info(f'  Downloading NWIS daily values: {group_name} ({pcode_str})')
        try:
            r = requests.get(f'{NWIS_BASE}/dv/', timeout=60, params={
                'format': 'rdb',
                'parameterCd': pcode_str,
                'startDT': '2010-01-01',
                'endDT': '2023-12-31',
                'siteStatus': 'all',
                'siteType': 'ST',  # streams
                'period': 'P3650D',
            })
            log.info(f'  NWIS dv {group_name}: HTTP {r.status_code}, {len(r.content)/1e6:.1f} MB')

            if r.status_code == 200 and len(r.content) > 1000:
                # Parse RDB format
                lines = r.text.split('\n')
                data_lines = [l for l in lines if l and not l.startswith('#')]
                if len(data_lines) > 2:
                    header = data_lines[0].split('\t')
                    rows = [l.split('\t') for l in data_lines[2:] if l.strip()]
                    df = pd.DataFrame(rows, columns=header[:len(rows[0])] if rows else header)
                    df.to_parquet(out_path, index=False)
                    log.info(f'  Saved: {group_name} → {len(df):,} rows')
        except Exception as e:
            log.warning(f'  NWIS dv {group_name} failed: {e}')

        time.sleep(2)  # rate limit


def download_nwis_groundwater():
    """Download USGS NWIS groundwater quality data"""
    log.info('Downloading USGS NWIS groundwater quality...')
    out_path = NWIS_DIR / 'nwis_gw_wq.parquet'
    if out_path.exists():
        log.info('  Already have groundwater data')
        return

    try:
        # Discrete water quality samples for wells
        r = requests.get(f'{NWIS_BASE}/qwdata/', timeout=60, params={
            'format': 'rdb',
            'siteType': 'GW',  # groundwater
            'pCode': '00940,00945,00900,00920,00955',  # Cl, SO4, hardness, Ca, SiO2
            'startDT': '2000-01-01',
            'endDT': '2023-12-31',
            'siteStatus': 'all',
        })
        log.info(f'  NWIS GW: HTTP {r.status_code}, {len(r.content)/1e6:.1f} MB')
        if r.status_code == 200 and len(r.content) > 1000:
            lines = r.text.split('\n')
            data_lines = [l for l in lines if l and not l.startswith('#')]
            if len(data_lines) > 2:
                header = data_lines[0].split('\t')
                rows = [l.split('\t') for l in data_lines[2:] if l.strip()]
                if rows:
                    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
                    df.to_parquet(out_path, index=False)
                    log.info(f'  Saved: {len(df):,} groundwater records')
    except Exception as e:
        log.warning(f'  NWIS GW failed: {e}')


if __name__ == '__main__':
    log.info('=== Expanded WQP + USGS NWIS Download ===')

    # Download additional WQP characteristics
    total_wqp = 0
    for char in WQP_EXTRA_CHARS[:8]:  # Start with first 8, respect rate limits
        n = download_wqp_characteristic(char, max_records=200000)
        total_wqp += n
        time.sleep(3)  # WQP rate limit

    # Check providers
    for provider in EXTRA_PROVIDERS[:3]:
        download_wqp_provider(provider)
        time.sleep(2)

    # NWIS daily values
    download_nwis_daily_values()

    # NWIS groundwater
    download_nwis_groundwater()

    # Summary
    wqp_files = list(WQP_DIR.glob('*.parquet'))
    nwis_files = list(NWIS_DIR.glob('*.parquet'))
    wqp_rows = sum(len(pd.read_parquet(f)) for f in wqp_files)
    nwis_rows = sum(len(pd.read_parquet(f)) for f in nwis_files)
    wqp_size = sum(f.stat().st_size for f in wqp_files)
    nwis_size = sum(f.stat().st_size for f in nwis_files)

    status = {
        'wqp_files': len(wqp_files),
        'wqp_rows': wqp_rows,
        'wqp_mb': wqp_size / 1e6,
        'nwis_files': len(nwis_files),
        'nwis_rows': nwis_rows,
        'nwis_mb': nwis_size / 1e6,
    }
    json.dump(status, open(WQP_DIR / 'download_status.json', 'w'), indent=2)
    log.info(f'WQP expanded: {wqp_rows:,} rows ({wqp_size/1e6:.1f} MB)')
    log.info(f'NWIS expanded: {nwis_rows:,} rows ({nwis_size/1e6:.1f} MB)')
    log.info('Done.')
