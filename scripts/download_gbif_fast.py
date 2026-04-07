"""
Fast GBIF download: restrict to North America, post-2010, smaller taxa.
Then download Canada WQP and USGS discrete WQ.
"""
import os, sys, json, time, requests, io
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('gbif_fast')

BASE = Path('data/raw')
WQP_BASE = 'https://www.waterqualitydata.us/data'


def download_gbif():
    out_dir = BASE / 'gbif_freshwater'
    out_dir.mkdir(exist_ok=True)
    log.info('=== GBIF Freshwater Bioindicators (NA, 2010+) ===')

    # More specific keys to avoid huge global queries
    # Restricted to North America + year >=2010 for speed
    taxa = {
        'mayflies_ephemeroptera': '936',
        'stoneflies_plecoptera': '937',
        'caddisflies_trichoptera': '935',
        'daphnia': '2234882',
        'chironomidae': '9153',
        'freshwater_mussels_unionida': '2287',
        'algae_chlorophyta': '3041',
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
        max_records = 50000  # Smaller cap for speed

        while offset < max_records:
            try:
                params = {
                    'taxonKey': taxon_key,
                    'hasCoordinate': 'true',
                    'hasGeospatialIssue': 'false',
                    'occurrenceStatus': 'PRESENT',
                    'continent': 'NORTH_AMERICA',
                    'year': '2010,2026',
                    'limit': 300,
                    'offset': offset,
                }
                r = requests.get('https://api.gbif.org/v1/occurrence/search',
                               timeout=20, params=params)
                if r.status_code != 200:
                    log.warning(f'  {taxon_name}: HTTP {r.status_code}')
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
                    'state': rec.get('stateProvince', ''),
                    'water_body': rec.get('waterBody', ''),
                    'taxon': taxon_name,
                } for rec in results])
                offset += len(results)
                if data.get('endOfRecords', True):
                    break
                time.sleep(0.1)
            except Exception as e:
                log.warning(f'  {taxon_name} offset {offset}: {e}')
                break

        if all_records:
            df = pd.DataFrame(all_records)
            df.to_parquet(out_path, index=False, compression='snappy')
            log.info(f'  Saved: {taxon_name} → {len(df):,}')
            total += len(df)
        else:
            log.info(f'  {taxon_name}: no records found')

    log.info(f'  GBIF total: {total:,}')
    return total


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


def download_usgs_discrete():
    out_dir = BASE / 'usgs_discrete_wq'
    out_dir.mkdir(exist_ok=True)
    log.info('=== USGS Discrete Water Quality ===')

    param_groups = {
        'nutrients': '00600,00605,00608,00613,00618,00631,62855',
        'metals': '01046,01049,01051,01056,01060,01065,01080,01090',
        'physical': '00010,00095,00300,00400,63680',
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


if __name__ == '__main__':
    log.info('=' * 60)
    log.info('Fast GBIF + Canada WQP + USGS Discrete WQ')
    log.info('=' * 60)

    results = {}
    results['gbif'] = download_gbif()
    results['canada'] = download_canada()
    results['usgs_discrete'] = download_usgs_discrete()

    log.info('=' * 60)
    log.info('SUMMARY:')
    for src, n in results.items():
        log.info(f'  {src}: {n:,}')
    log.info(f'  TOTAL: {sum(results.values()):,}')

    json.dump(results, open('data/raw/gbif_fast_status.json', 'w'), indent=2)
    log.info('Done.')
