"""
Download GBIF freshwater species, Canada WQP, and USGS discrete water quality.
Runs after NEON compression completes.
"""
import os, sys, json, time, requests, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('gbif_canada_usgs')

BASE = Path('data/raw')
WQP_BASE = 'https://www.waterqualitydata.us/data'


# ─── GBIF freshwater bioindicator species ───────────────
def download_gbif():
    out_dir = BASE / 'gbif_freshwater'
    out_dir.mkdir(exist_ok=True)
    log.info('=== GBIF Freshwater Bioindicators ===')

    # Aquatic taxa used as water quality bioindicators
    taxa = {
        'mayflies_ephemeroptera': '936',      # Water quality bioindicators
        'stoneflies_plecoptera': '937',
        'caddisflies_trichoptera': '935',
        'daphnia': '2234882',
        'chironomidae': '9153',               # Midges - pollution tolerant/sensitive
        'freshwater_mussels': '2287',         # Unionida
        'freshwater_fish': '204',             # Actinopterygii (subset by habitat)
        'algae_chlorophyta': '3041',          # Green algae water quality indicators
    }

    total = 0
    for taxon_name, taxon_key in taxa.items():
        out_path = out_dir / f'gbif_{taxon_name}.parquet'
        if out_path.exists():
            n = len(pd.read_parquet(out_path))
            log.info(f'  Already have {taxon_name}: {n:,}')
            total += n
            continue

        log.info(f'  GBIF {taxon_name} (taxonKey={taxon_key})')
        all_records = []
        offset = 0
        max_records = 100000

        while offset < max_records:
            try:
                params = {
                    'taxonKey': taxon_key,
                    'hasCoordinate': 'true',
                    'hasGeospatialIssue': 'false',
                    'occurrenceStatus': 'PRESENT',
                    'limit': 300,
                    'offset': offset,
                }
                r = requests.get('https://api.gbif.org/v1/occurrence/search',
                               timeout=30, params=params)
                if r.status_code != 200:
                    break
                data = r.json()
                results = data.get('results', [])
                if not results:
                    break
                all_records.extend([{
                    'species': rec.get('species', ''),
                    'lat': rec.get('decimalLatitude'),
                    'lon': rec.get('decimalLongitude'),
                    'date': rec.get('eventDate', ''),
                    'country': rec.get('countryCode', ''),
                    'water_body': rec.get('waterBody', ''),
                    'taxon': taxon_name,
                } for rec in results])
                offset += len(results)
                if data.get('endOfRecords', True):
                    break
                time.sleep(0.25)
            except Exception as e:
                log.warning(f'  {taxon_name} offset {offset}: {e}')
                break

        if all_records:
            df = pd.DataFrame(all_records)
            df.to_parquet(out_path, index=False, compression='snappy')
            log.info(f'  Saved: {taxon_name} → {len(df):,}')
            total += len(df)

    log.info(f'  GBIF total: {total:,}')
    return total


# ─── Canada WQP ─────────────────────────────────────────
def download_canada():
    out_dir = BASE / 'canada_wq'
    out_dir.mkdir(exist_ok=True)
    log.info('=== Canada Water Quality (WQP) ===')

    out_path = out_dir / 'canada_wqp.parquet'
    if out_path.exists():
        n = len(pd.read_parquet(out_path))
        log.info(f'  Already have Canada: {n:,}')
        return n

    try:
        r = requests.get(f'{WQP_BASE}/Result/search', timeout=30, params={
            'countrycode': 'CA',
            'mimeType': 'csv',
            'dataProfile': 'narrowResult',
            'sorted': 'no',
        })
        count = int(r.headers.get('Total-Result-Count', '0'))
        log.info(f'  Canada WQP: {count:,} records available')
    except Exception as e:
        log.warning(f'  Canada count check failed: {e}')
        count = 0

    if count == 0:
        return 0

    # Download with streaming, cap at 300 MB
    try:
        r = requests.get(f'{WQP_BASE}/Result/search', timeout=180,
                        stream=True, params={
                            'countrycode': 'CA',
                            'mimeType': 'csv',
                            'dataProfile': 'narrowResult',
                            'sorted': 'no',
                        })
        content = b''
        for chunk in r.iter_content(chunk_size=4*1024*1024):
            content += chunk
            if len(content) > 300e6:
                log.warning('  Capped Canada at 300 MB')
                break

        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        df.to_parquet(out_path, index=False, compression='snappy')
        log.info(f'  Canada WQP saved: {len(df):,}')
        return len(df)
    except Exception as e:
        log.warning(f'  Canada download failed: {e}')
        return 0


# ─── USGS discrete water quality (grab samples) ──────────
def download_usgs_discrete():
    out_dir = BASE / 'usgs_discrete_wq'
    out_dir.mkdir(exist_ok=True)
    log.info('=== USGS Discrete Water Quality (grab samples) ===')

    # Key parameter codes for WQ grab samples
    param_groups = {
        'nutrients': '00600,00605,00608,00613,00618,00631,62855',
        'metals': '01046,01049,01051,01056,01060,01065,01080,01090',
        'physical': '00010,00095,00300,00400,63680',
        'microbial': '31501,31625,50468',
        'organics': '32210,32211',
    }

    total = 0
    for group, pcodes in param_groups.items():
        out_path = out_dir / f'usgs_wq_{group}.parquet'
        if out_path.exists():
            n = len(pd.read_parquet(out_path))
            log.info(f'  Already have {group}: {n:,}')
            total += n
            continue

        log.info(f'  USGS discrete {group}...')
        try:
            r = requests.get(f'{WQP_BASE}/Result/search', timeout=30, params={
                'organization': 'USGS',
                'pCode': pcodes,
                'startDateLo': '01-01-2000',
                'mimeType': 'csv',
                'dataProfile': 'narrowResult',
                'sorted': 'no',
            })
            count = int(r.headers.get('Total-Result-Count', '0'))
            log.info(f'  {group}: {count:,} available')
        except Exception as e:
            log.warning(f'  {group} count failed: {e}')
            time.sleep(3)
            continue

        if count == 0:
            time.sleep(3)
            continue

        try:
            r = requests.get(f'{WQP_BASE}/Result/search', timeout=300,
                           stream=True, params={
                               'organization': 'USGS',
                               'pCode': pcodes,
                               'startDateLo': '01-01-2000',
                               'mimeType': 'csv',
                               'dataProfile': 'narrowResult',
                               'sorted': 'no',
                           })
            content = b''
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                content += chunk
                if len(content) > 300e6:
                    log.warning(f'  {group}: capped at 300 MB')
                    break

            df = pd.read_csv(io.BytesIO(content), low_memory=False)
            df.to_parquet(out_path, index=False, compression='snappy')
            log.info(f'  Saved: {group} → {len(df):,}')
            total += len(df)
        except Exception as e:
            log.warning(f'  {group} download failed: {e}')
        time.sleep(5)

    return total


# ─── WQP additional characteristics ─────────────────────
def download_wqp_extra():
    out_dir = BASE / 'wqp_extra_chars'
    out_dir.mkdir(exist_ok=True)
    log.info('=== WQP Additional Characteristics ===')

    chars = [
        'Chlorophyll a',
        'Nitrate',
        'Ammonia',
        'Turbidity',
        'Specific conductance',
        'Escherichia coli',
        'Total dissolved solids',
        'Phosphorus',
    ]

    total = 0
    for char in chars:
        safe = char.replace(' ', '_').replace('-', '').replace('.', '')[:25]
        out_path = out_dir / f'wqp_{safe}.parquet'
        if out_path.exists():
            n = len(pd.read_parquet(out_path))
            log.info(f'  Already have {char}: {n:,}')
            total += n
            continue

        log.info(f'  WQP: {char}')
        try:
            r = requests.get(f'{WQP_BASE}/Result/search', timeout=60, params={
                'characteristicName': char,
                'startDateLo': '01-01-2010',
                'mimeType': 'csv',
                'dataProfile': 'narrowResult',
                'sorted': 'no',
            })
            count = int(r.headers.get('Total-Result-Count', '0'))
            log.info(f'  {char}: {count:,} available')

            if count == 0:
                time.sleep(3)
                continue

            r2 = requests.get(f'{WQP_BASE}/Result/search', timeout=300,
                            stream=True, params={
                                'characteristicName': char,
                                'startDateLo': '01-01-2010',
                                'mimeType': 'csv',
                                'dataProfile': 'narrowResult',
                                'sorted': 'no',
                            })
            content = b''
            for chunk in r2.iter_content(chunk_size=4*1024*1024):
                content += chunk
                if len(content) > 200e6:
                    break

            df = pd.read_csv(io.BytesIO(content), low_memory=False)
            df.to_parquet(out_path, index=False, compression='snappy')
            log.info(f'  Saved: {char} → {len(df):,}')
            total += len(df)
        except Exception as e:
            log.warning(f'  {char} failed: {e}')
        time.sleep(5)

    return total


if __name__ == '__main__':
    log.info('=' * 60)
    log.info('Post-NEON Downloads: GBIF + Canada + USGS Discrete + WQP Extra')
    log.info('=' * 60)

    results = {}
    results['gbif'] = download_gbif()
    results['canada'] = download_canada()
    results['usgs_discrete'] = download_usgs_discrete()
    results['wqp_extra'] = download_wqp_extra()

    log.info('=' * 60)
    log.info('SUMMARY:')
    for src, n in results.items():
        log.info(f'  {src}: {n:,}')
    log.info(f'  TOTAL: {sum(results.values()):,}')

    json.dump(results, open('data/raw/post_neon_status.json', 'w'), indent=2)
    log.info('Done.')
