"""
Download WHO/UNICEF JMP WASH data and FreshWater Watch data.
JMP: https://washdata.org/data/downloads
Coverage: 200+ countries, WASH indicators
"""
import os, sys, json, time, requests, zipfile, io
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('who_jmp')

OUT_DIR = Path('data/raw/who_jmp')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FWW_DIR = Path('data/raw/freshwater_watch')
FWW_DIR.mkdir(parents=True, exist_ok=True)


# WHO/UNICEF JMP - direct download endpoints
JMP_URLS = {
    # National data - drinking water and sanitation
    'jmp_national_wats': 'https://washdata.org/sites/default/files/wld_2023_jmpwat_en.xlsx',
    'jmp_national_san': 'https://washdata.org/sites/default/files/wld_2023_jmpsan_en.xlsx',
    # CSV versions
    'jmp_water_csv': 'https://washdata.org/data/household#!/table?geo0=region&geo1=WLD&tab=download',
    # Open data portal
    'jmp_opendata': 'https://api.washdata.org/download?indicator=W-P-LMD&area_type=country&start_year=2000&end_year=2022&format=csv',
}

# Alternative: World Bank Water data
WB_URLS = {
    'wb_water_quality': 'https://api.worldbank.org/v2/country/all/indicator/SH.H2O.SMDW.ZS?format=json&per_page=10000&mrv=1',
    'wb_sanitation': 'https://api.worldbank.org/v2/country/all/indicator/SH.STA.SMSS.ZS?format=json&per_page=10000&mrv=1',
}

# FreshWater Watch
FWW_URLS = {
    'fww_zenodo': 'https://zenodo.org/record/5076926',  # FreshWater Watch data
    'fww_api': 'https://freshwaterwatch.thewaterhub.org/api/v1/results',
}


def download_jmp():
    """Download WHO/UNICEF JMP WASH indicators"""
    log.info('Downloading JMP WASH data...')
    records = []

    # Try direct XLSX downloads
    for name, url in JMP_URLS.items():
        dest = OUT_DIR / f'{name}.xlsx'
        if dest.exists() and dest.stat().st_size > 10000:
            log.info(f'  Already have: {name}')
            continue
        try:
            r = requests.get(url, timeout=60, allow_redirects=True,
                           headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dest, 'wb') as f:
                    f.write(r.content)
                log.info(f'  Saved: {name} ({dest.stat().st_size/1e3:.1f} KB)')
                records.append(name)
            else:
                log.warning(f'  Failed {name}: HTTP {r.status_code}')
        except Exception as e:
            log.warning(f'  Failed {name}: {e}')

    # Try World Bank API (open, no auth needed)
    log.info('Downloading World Bank water/sanitation indicators...')
    wb_records = []
    for name, url in WB_URLS.items():
        try:
            # Get total pages first
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 1:
                    meta = data[0]
                    values = data[1]
                    total_pages = meta.get('pages', 1)
                    log.info(f'  {name}: {meta.get("total", 0)} records, {total_pages} pages')
                    all_values = list(values)

                    for page in range(2, min(total_pages + 1, 20)):
                        r2 = requests.get(url + f'&page={page}', timeout=30)
                        if r2.status_code == 200:
                            d2 = r2.json()
                            if isinstance(d2, list) and len(d2) > 1:
                                all_values.extend(d2[1])
                        time.sleep(0.2)

                    if all_values:
                        df = pd.DataFrame(all_values)
                        out_path = OUT_DIR / f'{name}.parquet'
                        df.to_parquet(out_path, index=False)
                        log.info(f'  Saved: {name} ({len(df):,} records)')
                        wb_records.append({'name': name, 'rows': len(df)})
        except Exception as e:
            log.warning(f'  World Bank {name} failed: {e}')

    # Search Zenodo for JMP data
    log.info('Searching Zenodo for JMP/WASH data...')
    try:
        r = requests.get('https://zenodo.org/api/records', timeout=30, params={
            'q': 'WHO UNICEF JMP WASH water sanitation',
            'sort': 'mostrecent', 'size': 5, 'type': 'dataset', 'access_right': 'open'
        })
        if r.status_code == 200:
            for h in r.json().get('hits', {}).get('hits', []):
                title = h.get('metadata', {}).get('title', '')
                recid = h.get('id', '')
                log.info(f'  [{recid}] {title[:70]}')
    except Exception as e:
        log.warning(f'Zenodo search failed: {e}')

    return records


def download_freshwater_watch():
    """Download FreshWater Watch citizen science data"""
    log.info('Downloading FreshWater Watch data...')

    # Try Zenodo - FreshWater Watch has published datasets there
    zenodo_queries = [
        'FreshWater Watch water quality',
        'Earthwatch freshwater quality citizen science',
    ]

    for q in zenodo_queries:
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
                    log.info(f'  [{recid}] {title[:70]} ({sz:.0f} MB, {len(files)} files)')

                    # Download if promising
                    if 'freshwater' in title.lower() or 'water quality' in title.lower():
                        for f in files[:3]:
                            furl = f['links']['self']
                            fname = f['key']
                            fsize = f.get('size', 0)
                            if fsize < 500e6:  # < 500 MB
                                dest = FWW_DIR / fname
                                if not dest.exists():
                                    log.info(f'  Downloading: {fname} ({fsize/1e6:.1f} MB)')
                                    with requests.get(furl, stream=True, timeout=120) as resp:
                                        if resp.status_code == 200:
                                            with open(dest, 'wb') as fp:
                                                for chunk in resp.iter_content(chunk_size=2*1024*1024):
                                                    fp.write(chunk)
                                            log.info(f'  Saved: {fname}')
        except Exception as e:
            log.warning(f'  Search failed: {e}')

    # Try FWW API directly
    log.info('Trying FreshWater Watch API...')
    try:
        r = requests.get('https://freshwaterwatch.thewaterhub.org/api/v1/results',
                        timeout=30, params={'limit': 1000, 'offset': 0,
                                           'format': 'json'})
        log.info(f'FWW API: {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            log.info(f'FWW data: {json.dumps(data)[:200]}')
    except Exception as e:
        log.warning(f'FWW API failed: {e}')


def download_gleon():
    """Download GLEON data from EDI repository"""
    log.info('Downloading GLEON data from EDI...')
    gleon_dir = Path('data/raw/gleon')
    gleon_dir.mkdir(parents=True, exist_ok=True)

    # EDI search for GLEON data
    edi_search = 'https://pasta.lternet.edu/package/search/eml?q=GLEON+lake+water+quality&fl=packageid,title&rows=20'
    try:
        r = requests.get(edi_search, timeout=30)
        if r.status_code == 200:
            log.info(f'EDI search result: {r.text[:500]}')
    except Exception as e:
        log.warning(f'EDI search failed: {e}')

    # Try directly known GLEON packages
    # Format: https://pasta.lternet.edu/package/data/eml/{scope}/{identifier}/{revision}/{entity}
    known_packages = [
        # GLEON global lake datasets
        ('edi', '1029', '1'),  # Lake metabolism
        ('edi', '1027', '3'),  # GLEON water temperature
        ('knb-lter-ntl', '1', '59'),  # North Temperate Lakes LTER
    ]
    for scope, pkg_id, rev in known_packages[:2]:
        url = f'https://pasta.lternet.edu/package/data/eml/{scope}/{pkg_id}/{rev}'
        try:
            r = requests.get(url, timeout=30)
            log.info(f'GLEON {scope}.{pkg_id}.{rev}: {r.status_code} {r.text[:200]}')
        except Exception as e:
            log.warning(f'GLEON EDI {pkg_id} failed: {e}')

    # Search Zenodo for GLEON
    log.info('Searching Zenodo for GLEON data...')
    try:
        r = requests.get('https://zenodo.org/api/records', timeout=30, params={
            'q': 'GLEON global lake ecological observatory',
            'sort': 'mostrecent', 'size': 5, 'type': 'dataset', 'access_right': 'open'
        })
        if r.status_code == 200:
            for h in r.json().get('hits', {}).get('hits', []):
                title = h.get('metadata', {}).get('title', '')
                recid = h.get('id', '')
                sz = sum(f.get('size', 0) for f in h.get('files', [])) / 1e6
                log.info(f'  [{recid}] {title[:70]} ({sz:.0f} MB)')
    except Exception as e:
        log.warning(f'Zenodo search failed: {e}')

    status = {'dir': str(gleon_dir)}
    json.dump(status, open(gleon_dir / 'download_status.json', 'w'), indent=2)


if __name__ == '__main__':
    log.info('=== WHO/UNICEF JMP + FreshWater Watch + GLEON Download ===')

    jmp_records = download_jmp()
    download_freshwater_watch()
    download_gleon()

    # JMP status
    jmp_files = [f for f in OUT_DIR.iterdir() if f.is_file()]
    jmp_size = sum(f.stat().st_size for f in jmp_files)
    fww_files = [f for f in FWW_DIR.iterdir() if f.is_file()]
    fww_size = sum(f.stat().st_size for f in fww_files)

    status = {
        'jmp_files': [f.name for f in jmp_files],
        'jmp_mb': jmp_size / 1e6,
        'fww_files': [f.name for f in fww_files],
        'fww_mb': fww_size / 1e6,
    }
    json.dump(status, open(OUT_DIR / 'download_status.json', 'w'), indent=2)
    log.info(f'JMP: {jmp_size/1e6:.1f} MB, FWW: {fww_size/1e6:.1f} MB')
    log.info('Done.')
