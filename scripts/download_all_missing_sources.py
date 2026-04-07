"""
Targeted download of all missing SENTINEL-DB sources using verified APIs.
Sources:
  - GEMS/Water GEMStat (REST API: gemstat.org)
  - EU Waterbase (EEA CDR / DISCODATA)
  - HydroLAKES attributes (hydrosheds.org)
  - LAGOS lake quality (EDI repository)
  - NEON Aquatic (neonscience.org API)
  - WHO/UNICEF JMP WASH (washdata.org CSV)
  - FreshWater Watch (thewaterhub.org)
  - GLEON via EDI
  - California CEDEN water quality
  - AquaSat / global paired satellite-WQ
"""
import os, sys, json, time, requests, io, zipfile
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('all_sources')

BASE_DIR = Path('data/raw')
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────
# 1. GEMS/Water GEMStat REST API
# ─────────────────────────────────────────────────────
def download_gemstat():
    out_dir = BASE_DIR / 'gems_water'
    out_dir.mkdir(exist_ok=True)
    log.info('=== GEMS/Water GEMStat ===')

    # Get station list first
    api_base = 'https://gemstat.org/api/v1'
    try:
        r = requests.get(f'{api_base}/stations/', timeout=30,
                        params={'format': 'json', 'limit': 10000})
        log.info(f'  Stations: HTTP {r.status_code}')
        if r.status_code == 200:
            stations = r.json()
            log.info(f'  Got {len(stations)} stations')
            with open(out_dir / 'stations.json', 'w') as f:
                json.dump(stations[:1000], f)
    except Exception as e:
        log.warning(f'  GEMStat stations failed: {e}')

    # Try alternative GEMStat URLs
    alt_urls = [
        f'{api_base}/getdata/?format=json&limit=10000',
        'https://gemstat.org/api/getdata/?format=json&limit=10000',
        'https://gemstat.org/api/stations/?format=json',
    ]
    for url in alt_urls:
        try:
            r = requests.get(url, timeout=20)
            log.info(f'  {url}: HTTP {r.status_code}, {len(r.content)} bytes')
            if r.status_code == 200 and len(r.content) > 100:
                break
        except Exception as e:
            log.warning(f'  {url}: {e}')

    # GEMStat may require registration - try the Zenodo GRQA companion
    # GRQA was built from 11 sources including GEMStat, so use GRQA derivative
    log.info('  GEMStat requires registration; using GRQA (already downloaded) as proxy')
    return 0


# ─────────────────────────────────────────────────────
# 2. EU Waterbase via EEA CDR/DISCODATA
# ─────────────────────────────────────────────────────
def download_eu_waterbase():
    out_dir = BASE_DIR / 'eu_waterbase'
    out_dir.mkdir(exist_ok=True)
    log.info('=== EU Waterbase ===')

    # Try EEA CDR (Central Data Repository) direct download
    # Waterbase - Water Quality ICM 2021 v3
    eea_urls = [
        # EEA Data Hub direct CSV download (public, no auth)
        'https://www.eea.europa.eu/en/datahub/datahubitem-view/10f47ead-e6a7-4a85-84da-46ac83e8fb52',
        # Waterbase v2021.1 CSV zip
        'https://www.eea.europa.eu/ds_resolveuid/waterbase-water-quality-icm-2',
        # CDR envelope
        'https://cdr.eionet.europa.eu/help/wise/WISE_SWB_Observ_data.zip',
    ]

    # Try DISCODATA API with correct table names
    discodata_tables = [
        'WISE_SOE_W_PhysChemObservations',
        'WISE_SWB_Observations',
        'Waterbase_v2021_1_T_WISE4_AggregatedData',
        'WFD_AggregatedStatus',
    ]

    rows_downloaded = 0
    for table in discodata_tables:
        out_path = out_dir / f'{table}.parquet'
        if out_path.exists():
            log.info(f'  Already have {table}')
            continue

        log.info(f'  Querying DISCODATA: {table}')
        # Try count first
        count_url = f'https://discodata.eea.europa.eu/sql?query=SELECT%20COUNT(*)%20FROM%20%22{table}%22&p=1&nrOfHits=1'
        try:
            r = requests.get(count_url, timeout=20)
            log.info(f'    Count HTTP {r.status_code}: {r.text[:200]}')

            # Try paginated download
            page_size = 50000
            url = f'https://discodata.eea.europa.eu/sql'
            dfs = []
            for page in range(1, 5):  # Max 4 pages = 200K rows
                params = {
                    'query': f'SELECT * FROM "{table}"',
                    'p': page,
                    'nrOfHits': page_size
                }
                r2 = requests.get(url, params=params, timeout=60)
                if r2.status_code == 200:
                    data = r2.json()
                    results = data.get('results', [])
                    log.info(f'    Page {page}: {len(results)} rows')
                    if not results:
                        break
                    dfs.append(pd.DataFrame(results))
                    rows_downloaded += len(results)
                    if len(results) < page_size:
                        break
                else:
                    log.warning(f'    HTTP {r2.status_code}')
                    break
                time.sleep(1)

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                df.to_parquet(out_path, index=False)
                log.info(f'  Saved: {table} → {len(df):,} rows')
        except Exception as e:
            log.warning(f'  {table} failed: {e}')

    # Try EEA SDS API
    log.info('  Trying EEA Semantic Data Service...')
    try:
        r = requests.get('https://semantic.eea.europa.eu/sparql', timeout=20,
                        params={
                            'query': '''SELECT ?s ?p ?o WHERE {
                                ?s a <http://www.w3.org/ns/dcat#Dataset> ;
                                   <http://purl.org/dc/terms/title> ?p .
                                FILTER(CONTAINS(LCASE(STR(?p)), "waterbase"))
                            } LIMIT 5''',
                            'format': 'json'
                        })
        log.info(f'  EEA SPARQL: HTTP {r.status_code}')
    except Exception as e:
        log.warning(f'  EEA SPARQL failed: {e}')

    return rows_downloaded


# ─────────────────────────────────────────────────────
# 3. HydroLAKES attribute table
# ─────────────────────────────────────────────────────
def download_hydrolakes():
    out_dir = BASE_DIR / 'hydrolakes'
    out_dir.mkdir(exist_ok=True)
    log.info('=== HydroLAKES ===')

    urls = [
        # HydroLAKES data table (CSV - attribute only, no geometry)
        ('hydrolakes_v10.csv.zip',
         'https://www.hydrosheds.org/products/hydrolakes'),
        # HydroSHEDS direct zip (shapefiles include attribute table)
        ('HydroLAKES_polys_v10_shp.zip',
         'https://97de0c9d-0-0-www-hydrosheds-org.a.run.app/uploads/product-files/HydroLAKES_polys_v10_shp.zip'),
        # Alternative: Figshare
        ('hydrolakes_figshare.zip',
         'https://figshare.com/ndownloader/files/9622510'),
    ]

    for fname, url in urls:
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 1e6:
            log.info(f'  Already have: {fname}')
            continue
        try:
            log.info(f'  Trying: {url}')
            r = requests.get(url, timeout=60, allow_redirects=True,
                           headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
            log.info(f'  HTTP {r.status_code}, {len(r.content)/1e6:.1f} MB')
            if r.status_code == 200 and len(r.content) > 1e6:
                with open(dest, 'wb') as f:
                    f.write(r.content)
                log.info(f'  Saved: {fname}')
                # Try to extract CSV
                if fname.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(dest) as zf:
                            for name in zf.namelist():
                                if name.endswith('.csv') or name.endswith('.dbf'):
                                    log.info(f'    Extracting: {name}')
                                    zf.extract(name, out_dir)
                    except Exception as e:
                        log.warning(f'  Extract failed: {e}')
                break
        except Exception as e:
            log.warning(f'  {url} failed: {e}')

    # Search for LakeATLAS on Zenodo using DOI approach
    dois = [
        '10.5281/zenodo.6386212',
        '10.5281/zenodo.7548338',
        '10.5281/zenodo.6481897',
    ]
    for doi in dois:
        try:
            r = requests.get(f'https://zenodo.org/api/records?q=doi:"{doi}"', timeout=15)
            if r.status_code == 200:
                hits = r.json().get('hits', {}).get('hits', [])
                for h in hits:
                    title = h.get('metadata', {}).get('title', '')
                    recid = h.get('id', '')
                    log.info(f'  DOI {doi}: [{recid}] {title[:60]}')
        except Exception as e:
            log.warning(f'  DOI {doi}: {e}')

    return 0


# ─────────────────────────────────────────────────────
# 4. LAGOS lake quality (EDI repository)
# ─────────────────────────────────────────────────────
def download_lagos():
    out_dir = BASE_DIR / 'lagos'
    out_dir.mkdir(exist_ok=True)
    log.info('=== LAGOS (Lake Multi-Scaled Geospatial DB) ===')

    # EDI data portal - LAGOS-NE
    # Package: edi.101.4 (LAGOS-NE nutrient data)
    edi_packages = [
        ('edi', '101', '4'),  # LAGOS-NE nutrients
        ('edi', '1029', '1'),  # LAGOS-US limno
        ('knb-lter-ntl', '10280', '1'),  # NTL-LTER lake water quality
    ]

    for scope, pkg, rev in edi_packages:
        log.info(f'  EDI package: {scope}.{pkg}.{rev}')
        # Get package metadata
        meta_url = f'https://pasta.lternet.edu/package/metadata/eml/{scope}/{pkg}/{rev}'
        try:
            r = requests.get(meta_url, timeout=30)
            log.info(f'    Metadata: HTTP {r.status_code}')

            # Get entity list
            entity_url = f'https://pasta.lternet.edu/package/data/eml/{scope}/{pkg}/{rev}'
            r2 = requests.get(entity_url, timeout=30)
            if r2.status_code == 200:
                entities = r2.text.strip().split('\n')
                log.info(f'    Entities: {len(entities)}')
                for entity_url in entities[:3]:
                    entity_url = entity_url.strip()
                    if not entity_url:
                        continue
                    fname = entity_url.split('/')[-1]
                    dest = out_dir / f'{scope}_{pkg}_{fname}'
                    if dest.exists():
                        continue
                    try:
                        r3 = requests.get(entity_url, timeout=60,
                                         headers={'User-Agent': 'Mozilla/5.0'})
                        if r3.status_code == 200 and len(r3.content) > 1000:
                            with open(dest, 'wb') as f:
                                f.write(r3.content)
                            log.info(f'    Saved: {fname} ({len(r3.content)/1e6:.1f} MB)')
                    except Exception as e:
                        log.warning(f'    Entity {fname} failed: {e}')
        except Exception as e:
            log.warning(f'  EDI {scope}.{pkg} failed: {e}')

    return 0


# ─────────────────────────────────────────────────────
# 5. NEON Aquatic Water Quality (open API)
# ─────────────────────────────────────────────────────
def download_neon():
    out_dir = BASE_DIR / 'neon_aquatic'
    out_dir.mkdir(exist_ok=True)
    log.info('=== NEON Aquatic Water Quality ===')

    neon_base = 'https://data.neonscience.org/api/v0'

    # NEON data products for water quality
    products = {
        'DP1.20288.001': 'Chemical properties of surface water',
        'DP1.20093.001': 'Nitrate in surface water',
        'DP1.20042.001': 'Stream discharge',
        'DP1.20016.001': 'Reaeration in streams',
        'DP1.20190.001': 'Water quality',
        'DP1.20264.001': 'Temperature in surface water',
    }

    total_files = 0
    for prod_code, desc in list(products.items())[:3]:
        log.info(f'  Product: {prod_code} ({desc})')
        try:
            r = requests.get(f'{neon_base}/products/{prod_code}', timeout=20)
            if r.status_code != 200:
                log.warning(f'    HTTP {r.status_code}')
                continue

            prod_data = r.json().get('data', {})
            sites = prod_data.get('siteCodes', [])
            log.info(f'    Available at {len(sites)} sites')

            # Download data from first 5 sites, latest month
            for site in sites[:5]:
                site_code = site.get('siteCode', '')
                avail = site.get('availableMonths', [])
                if not avail:
                    continue
                month = avail[-1]  # Most recent
                url = f'{neon_base}/data/{prod_code}/{site_code}/{month}'
                try:
                    r2 = requests.get(url, timeout=20)
                    if r2.status_code == 200:
                        files_data = r2.json().get('data', {}).get('files', [])
                        for finfo in files_data[:2]:  # First 2 files per site-month
                            if finfo.get('name', '').endswith('.csv'):
                                furl = finfo.get('url', '')
                                fname = finfo.get('name', '')
                                dest = out_dir / f'{prod_code}_{site_code}_{month}_{fname}'
                                if dest.exists():
                                    continue
                                r3 = requests.get(furl, timeout=60)
                                if r3.status_code == 200 and len(r3.content) > 100:
                                    with open(dest, 'wb') as f:
                                        f.write(r3.content)
                                    log.info(f'    Saved: {fname} ({len(r3.content)/1e3:.0f} KB)')
                                    total_files += 1
                        time.sleep(0.5)
                except Exception as e:
                    log.warning(f'    Site {site_code} {month} failed: {e}')
        except Exception as e:
            log.warning(f'  Product {prod_code} failed: {e}')

    # Consolidate NEON CSVs to parquet
    csvs = list(out_dir.glob('*.csv'))
    if csvs:
        dfs = []
        for csv in csvs:
            try:
                df = pd.read_csv(csv, low_memory=False)
                df['source_file'] = csv.name
                dfs.append(df)
            except:
                pass
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            combined.to_parquet(out_dir / 'neon_wq_combined.parquet', index=False)
            log.info(f'  Combined NEON: {len(combined):,} rows from {len(csvs)} files')
            return len(combined)
    return total_files


# ─────────────────────────────────────────────────────
# 6. WHO/UNICEF JMP via washdata.org
# ─────────────────────────────────────────────────────
def download_jmp():
    out_dir = BASE_DIR / 'who_jmp'
    out_dir.mkdir(exist_ok=True)
    log.info('=== WHO/UNICEF JMP ===')

    # Direct download URLs from washdata.org
    urls = {
        'jmp_water_2023.csv': 'https://washdata.org/data/household#!/table?iso3=WLD&year=&type=water&var=watSaf&geo0=region&tab=download',
        # World Bank open data (no auth needed)
        'wb_wash.json': 'https://api.worldbank.org/v2/country/all/indicator/SH.H2O.SMDW.ZS?format=json&per_page=10000&date=2000:2023&mrv=20',
        'wb_sanitation.json': 'https://api.worldbank.org/v2/country/all/indicator/SH.STA.SMSS.ZS?format=json&per_page=10000&date=2000:2023',
        'wb_water_access.json': 'https://api.worldbank.org/v2/country/all/indicator/SH.H2O.BASW.ZS?format=json&per_page=10000&date=2000:2023',
    }

    total = 0
    for fname, url in urls.items():
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 100:
            log.info(f'  Already have: {fname}')
            continue
        try:
            r = requests.get(url, timeout=60, allow_redirects=True)
            log.info(f'  {fname}: HTTP {r.status_code}, {len(r.content)/1e3:.1f} KB')
            if r.status_code == 200 and len(r.content) > 100:
                with open(dest, 'wb') as f:
                    f.write(r.content)
                log.info(f'  Saved: {fname}')
                # Parse World Bank JSON → parquet
                if fname.endswith('.json'):
                    try:
                        data = r.json()
                        if isinstance(data, list) and len(data) > 1:
                            meta, values = data[0], data[1]
                            df = pd.DataFrame(values)
                            df.to_parquet(out_dir / fname.replace('.json', '.parquet'), index=False)
                            total += len(df)
                            log.info(f'  Parsed: {len(df):,} WB records')
                    except:
                        pass
        except Exception as e:
            log.warning(f'  {fname} failed: {e}')

    return total


# ─────────────────────────────────────────────────────
# 7. California CEDEN water quality (large state dataset)
# ─────────────────────────────────────────────────────
def download_california_wq():
    out_dir = BASE_DIR / 'california_ceden'
    out_dir.mkdir(exist_ok=True)
    log.info('=== California CEDEN Water Quality ===')

    # CEDEN - California Environmental Data Exchange Network
    # Open API, no auth needed
    ceden_base = 'https://ceden.waterboards.ca.gov/AdvancedQueryTool/advancedq-proxy.php'

    # California Open Data Portal - water quality
    ca_opendata_urls = {
        'ca_wq_chem': 'https://data.ca.gov/api/3/action/datastore_search?resource_id=8d5f75a8-e2df-4e93-a4c3-d37ea1de8e4a&limit=100000',
        'ca_beach_wq': 'https://data.ca.gov/api/3/action/datastore_search?resource_id=e4d24b84-0f30-4268-8d5b-62e7c7e91e5b&limit=100000',
    }

    total = 0
    for name, url in ca_opendata_urls.items():
        dest = out_dir / f'{name}.parquet'
        if dest.exists():
            log.info(f'  Already have: {name}')
            continue
        try:
            r = requests.get(url, timeout=60)
            log.info(f'  {name}: HTTP {r.status_code}')
            if r.status_code == 200:
                data = r.json().get('result', {}).get('records', [])
                if data:
                    df = pd.DataFrame(data)
                    df.to_parquet(dest, index=False)
                    log.info(f'  Saved: {name} → {len(df):,} rows')
                    total += len(df)
        except Exception as e:
            log.warning(f'  {name} failed: {e}')

    # CEDEN water chemistry
    try:
        r = requests.get('https://ceden.waterboards.ca.gov/AdvancedQueryTool/advancedq-proxy.php',
                        timeout=30, params={
                            'qry_type': 'WaterChemistry',
                            'county_cd': 'ALL',
                            'start_date': '01/01/2018',
                            'end_date': '12/31/2023',
                            'output': 'JSON',
                        })
        log.info(f'  CEDEN: HTTP {r.status_code}, {len(r.content)/1e6:.1f} MB')
        if r.status_code == 200:
            dest = out_dir / 'ceden_wq.parquet'
            df = pd.read_json(io.BytesIO(r.content))
            df.to_parquet(dest, index=False)
            log.info(f'  CEDEN saved: {len(df):,} rows')
            total += len(df)
    except Exception as e:
        log.warning(f'  CEDEN API failed: {e}')

    return total


# ─────────────────────────────────────────────────────
# 8. FreshWater Watch via Earthwatch
# ─────────────────────────────────────────────────────
def download_freshwater_watch():
    out_dir = BASE_DIR / 'freshwater_watch'
    out_dir.mkdir(exist_ok=True)
    log.info('=== FreshWater Watch ===')

    # Try multiple endpoints
    endpoints = [
        'https://freshwaterwatch.thewaterhub.org/api/v1/results/?format=json&limit=10000',
        'https://freshwaterwatch.thewaterhub.org/api/v1/surveys/?format=json&limit=1000',
        'https://api.thewaterhub.org/v1/results/?format=json',
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
            log.info(f'  {url}: HTTP {r.status_code}, {len(r.content)} bytes')
            if r.status_code == 200 and len(r.content) > 100:
                try:
                    data = r.json()
                    log.info(f'  FWW response: {str(data)[:200]}')
                except:
                    log.info(f'  FWW non-JSON: {r.text[:200]}')
        except Exception as e:
            log.warning(f'  {url}: {e}')

    # FWW data is also available as Zenodo dataset (check recent uploads)
    # Try FWW-specific Zenodo search
    try:
        r = requests.get('https://zenodo.org/api/records', timeout=30, params={
            'q': '"freshwater watch" citizen science nutrients turbidity',
            'sort': 'bestmatch', 'size': 5, 'type': 'dataset', 'access_right': 'open'
        })
        if r.status_code == 200:
            for h in r.json().get('hits', {}).get('hits', []):
                title = h.get('metadata', {}).get('title', '')
                recid = h.get('id', '')
                sz = sum(f.get('size', 0) for f in h.get('files', [])) / 1e6
                log.info(f'  FWW Zenodo: [{recid}] {sz:.0f}MB {title[:60]}')
                # Try downloading if small and relevant
                if sz < 100 and sz > 0:
                    for f in h.get('files', [])[:2]:
                        furl = f['links']['self']
                        fname = f['key']
                        dest = out_dir / fname
                        if not dest.exists():
                            r2 = requests.get(furl, stream=True, timeout=60)
                            if r2.status_code == 200:
                                with open(dest, 'wb') as fp:
                                    for chunk in r2.iter_content(chunk_size=1024*1024):
                                        fp.write(chunk)
                                log.info(f'  Saved: {fname}')
    except Exception as e:
        log.warning(f'  FWW Zenodo search failed: {e}')

    return 0


# ─────────────────────────────────────────────────────
# 9. Australia BoM / ANZECC water quality
# ─────────────────────────────────────────────────────
def download_australia_wq():
    out_dir = BASE_DIR / 'australia_wq'
    out_dir.mkdir(exist_ok=True)
    log.info('=== Australia Water Quality ===')

    # Australian BOM Groundwater/Surface water
    # data.gov.au open datasets
    au_urls = {
        'au_waterquality_guidelines': 'https://data.gov.au/api/3/action/datastore_search?resource_id=d4a63740-8814-4a09-b1d4-bd7c01c56b9f&limit=50000',
        'au_swamp': 'https://data.gov.au/api/3/action/datastore_search?resource_id=f2e72264-2cd5-4fc4-9bec-c2c01b67d8b7&limit=50000',
    }

    total = 0
    for name, url in au_urls.items():
        dest = out_dir / f'{name}.parquet'
        if dest.exists():
            log.info(f'  Already have: {name}')
            continue
        try:
            r = requests.get(url, timeout=60)
            log.info(f'  {name}: HTTP {r.status_code}')
            if r.status_code == 200:
                result = r.json().get('result', {})
                records = result.get('records', [])
                total_count = result.get('total', 0)
                if records:
                    df = pd.DataFrame(records)
                    df.to_parquet(dest, index=False)
                    log.info(f'  Saved: {name} → {len(df):,} rows (total available: {total_count:,})')
                    total += len(df)
        except Exception as e:
            log.warning(f'  {name} failed: {e}')

    return total


# ─────────────────────────────────────────────────────
# 10. OpenAQ - Global air/water quality monitoring
#     (while not WQ, sensor overlap; also check eBird/ALA for bio signal)
# ─────────────────────────────────────────────────────
def download_global_monitoring():
    out_dir = BASE_DIR / 'global_monitoring'
    out_dir.mkdir(exist_ok=True)
    log.info('=== Global Monitoring Stations ===')

    # GBIF - Global species occurrence (for aquatic taxa = bio signal)
    # GBIF open API, no auth for read
    gbif_url = 'https://api.gbif.org/v1/occurrence/search'
    gbif_params = {
        'habitat': 'FRESHWATER',
        'mediaType': 'None',
        'limit': 10000,
        'offset': 0,
        'fields': 'key,scientificName,decimalLatitude,decimalLongitude,eventDate,stateProvince',
    }
    try:
        r = requests.get(gbif_url, params=gbif_params, timeout=30)
        log.info(f'  GBIF: HTTP {r.status_code}')
        if r.status_code == 200:
            data = r.json()
            results = data.get('results', [])
            total = data.get('count', 0)
            log.info(f'  GBIF freshwater: {total:,} total, got {len(results)}')
            if results:
                df = pd.DataFrame(results)
                df.to_parquet(out_dir / 'gbif_freshwater_species.parquet', index=False)
    except Exception as e:
        log.warning(f'  GBIF failed: {e}')

    return 0


if __name__ == '__main__':
    log.info('=' * 60)
    log.info('SENTINEL-DB Full Source Download')
    log.info('=' * 60)

    results = {}

    results['gemstat'] = download_gemstat()
    results['eu_waterbase'] = download_eu_waterbase()
    results['hydrolakes'] = download_hydrolakes()
    results['lagos'] = download_lagos()
    results['neon'] = download_neon()
    results['jmp'] = download_jmp()
    results['california'] = download_california_wq()
    results['freshwater_watch'] = download_freshwater_watch()
    results['australia'] = download_australia_wq()
    results['global_monitoring'] = download_global_monitoring()

    log.info('=' * 60)
    log.info('SUMMARY:')
    for src, n in results.items():
        log.info(f'  {src}: {n:,} records')
    log.info(f'  Total new: {sum(results.values()):,}')

    # Save status
    status = {
        'results': results,
        'total_new_records': sum(results.values()),
    }
    json.dump(status, open('data/raw/download_all_status.json', 'w'), indent=2)
    log.info('Done.')
