"""
Download MODIS Ocean Color / inland water quality products.
Focus: L3 monthly composites for inland water bodies (global, 4km)
Products: MOD28 (SST), MOD09 (surface reflectance), MODOCGA
Also tries Copernicus Global Land Service water quality products.
"""
import os, sys, json, time, requests
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('modis_oc')

OUT_DIR = Path('data/raw/modis_oc')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NASA CMR API for dataset discovery
CMR_URL = 'https://cmr.earthdata.nasa.gov/search'

# Target: monthly L3 global 4km products (manageable size)
MODIS_PRODUCTS = {
    'MOD28M': 'MODIS/Terra Sea Surface Temperature (SST) monthly 4km',
    'MYD19A3': 'MODIS Aqua Aerosol & Water',
    'MODOCGA': 'MODIS Ocean Color daily 1km',
}


def search_nasa_cmr(short_name=None, keyword=None, limit=10):
    """Search NASA CMR for datasets"""
    params = {
        'page_size': limit,
        'sort_key': '-start_date',
    }
    if short_name:
        params['short_name'] = short_name
    if keyword:
        params['keyword'] = keyword

    try:
        r = requests.get(f'{CMR_URL}/collections.json', params=params, timeout=30)
        if r.status_code == 200:
            collections = r.json().get('feed', {}).get('entry', [])
            return collections
    except Exception as e:
        log.warning(f'CMR search failed: {e}')
    return []


def try_earthdata_download():
    """Try downloading MODIS data via NASA Earthdata (may need auth)"""
    log.info('Trying NASA Earthdata...')
    # Search for inland water body reflectance
    keywords = [
        'inland water quality MODIS',
        'lake water quality remote sensing',
        'MODIS chlorophyll inland',
    ]
    for kw in keywords:
        collections = search_nasa_cmr(keyword=kw, limit=5)
        for c in collections:
            log.info(f'  [{c.get("id", "")}] {c.get("title", "")[:80]}')


def try_copernicus_cwq():
    """Try Copernicus Global Land Service - Inland Water Quality (CWQ)"""
    log.info('Trying Copernicus Inland Water Quality products...')
    # Copernicus Land Monitoring Service
    base = 'https://land.copernicus.eu/en/products/water/inland-water-quality'

    # Try WEkEO Harmonized Data Access API (no auth for public datasets)
    wekeo_url = 'https://wekeo-broker.apps.mercator.dpi.wekeo.eu/databroker/queryids'
    datasets = [
        'EO:CLMS:DAT:GLOBAL_INLAND_WATER_QUALITY_NOBS',
        'EO:CLMS:DAT:GLOBAL_INLAND_WATER_QUALITY_TURBIDITY',
    ]
    for ds in datasets:
        try:
            r = requests.post(wekeo_url, json={
                'datasetId': ds,
                'boundingBoxValues': [{'name': 'bbox', 'bbox': [-180, -90, 180, 90]}],
                'dateRangeSelectValues': [{'name': 'position', 'start': '2020-01-01', 'end': '2020-12-31'}],
            }, timeout=30)
            log.info(f'  WEkEO {ds}: {r.status_code} {r.text[:200]}')
        except Exception as e:
            log.warning(f'  WEkEO failed: {e}')


def download_open_aquatic_remote_sensing():
    """Download openly available aquatic remote sensing datasets from Zenodo"""
    log.info('Searching Zenodo for aquatic remote sensing data...')
    queries = [
        'lake water quality remote sensing satellite',
        'inland water chlorophyll MODIS Landsat',
        'river turbidity satellite multitemporal',
        'MODIS inland water color',
    ]
    downloaded = []
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

                    # Download small datasets (<200 MB) that look relevant
                    if sz < 200 and sz > 0:
                        keywords = ['water quality', 'chlorophyll', 'turbidity', 'inland', 'lake', 'river']
                        if any(kw in title.lower() for kw in keywords):
                            for f in files[:2]:
                                fname = f['key']
                                fsize = f.get('size', 0)
                                furl = f['links']['self']
                                dest = OUT_DIR / fname
                                if not dest.exists() and fsize < 200e6:
                                    log.info(f'  Downloading: {fname} ({fsize/1e6:.1f} MB)')
                                    try:
                                        with requests.get(furl, stream=True, timeout=120) as resp:
                                            if resp.status_code == 200:
                                                with open(dest, 'wb') as fp:
                                                    for chunk in resp.iter_content(chunk_size=2*1024*1024):
                                                        fp.write(chunk)
                                                log.info(f'  Saved: {fname}')
                                                downloaded.append(fname)
                                    except Exception as e:
                                        log.warning(f'  Download failed: {e}')
        except Exception as e:
            log.warning(f'  Search failed: {e}')
    return downloaded


def download_glowabo():
    """Download GLObal WAter BOdies database"""
    log.info('Downloading GLOWABO (Global Water Bodies)...')
    zenodo_ids = ['6481897', '4536808']
    for rid in zenodo_ids:
        try:
            r = requests.get(f'https://zenodo.org/api/records/{rid}', timeout=30)
            if r.status_code == 200:
                meta = r.json()
                title = meta.get('metadata', {}).get('title', '')
                log.info(f'  [{rid}] {title}')
                files = meta.get('files', [])
                for f in files:
                    fname = f['key']
                    fsize = f.get('size', 0)
                    log.info(f'    {fname}: {fsize/1e6:.1f} MB')
                    if fsize < 1e9 and not (OUT_DIR / fname).exists():
                        furl = f['links']['self']
                        dest = OUT_DIR / fname
                        with requests.get(furl, stream=True, timeout=180) as resp:
                            if resp.status_code == 200:
                                with open(dest, 'wb') as fp:
                                    for chunk in resp.iter_content(chunk_size=2*1024*1024):
                                        fp.write(chunk)
                                log.info(f'    Saved: {fname}')
        except Exception as e:
            log.warning(f'  GLOWABO {rid} failed: {e}')


def download_aquawatch_insitu():
    """Download AquaWatch / global in-situ water clarity datasets"""
    log.info('Downloading AquaWatch/water clarity datasets...')
    # AquaSat - paired Landsat/in-situ WQ dataset
    aquasat_zenodo = '4139538'
    try:
        r = requests.get(f'https://zenodo.org/api/records/{aquasat_zenodo}', timeout=30)
        if r.status_code == 200:
            meta = r.json()
            title = meta.get('metadata', {}).get('title', '')
            log.info(f'  AquaSat: {title}')
            files = meta.get('files', [])
            for f in files[:3]:
                fname = f['key']
                fsize = f.get('size', 0)
                log.info(f'    {fname}: {fsize/1e6:.1f} MB')
                if fsize < 500e6:
                    dest = OUT_DIR / fname
                    if not dest.exists():
                        furl = f['links']['self']
                        with requests.get(furl, stream=True, timeout=180) as resp:
                            if resp.status_code == 200:
                                with open(dest, 'wb') as fp:
                                    for chunk in resp.iter_content(chunk_size=2*1024*1024):
                                        fp.write(chunk)
                                log.info(f'    Saved: {fname}')
    except Exception as e:
        log.warning(f'  AquaSat failed: {e}')


if __name__ == '__main__':
    log.info('=== MODIS / Satellite Water Quality Download ===')
    log.info(f'Output: {OUT_DIR.absolute()}')

    try_earthdata_download()
    try_copernicus_cwq()
    downloaded = download_open_aquatic_remote_sensing()
    download_glowabo()
    download_aquawatch_insitu()

    files = [f for f in OUT_DIR.iterdir() if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    status = {
        'files': [f.name for f in files],
        'total_mb': total_size / 1e6,
        'downloaded': downloaded,
    }
    json.dump(status, open(OUT_DIR / 'download_status.json', 'w'), indent=2)
    log.info(f'Total: {total_size/1e6:.1f} MB across {len(files)} files')
    log.info('Done.')
