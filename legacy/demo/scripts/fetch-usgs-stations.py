#!/usr/bin/env python
"""Fetch the active USGS NWIS iv (instantaneous-value) station catalog state-by-state
and bundle as src/data/usgsStations.json. Run once at build time; the demo never
hits the network for this list at runtime.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

# CONUS only (Alaska/Hawaii/PR omitted for the demo since the map is CONUS-focused).
STATES = [
    'al','az','ar','ca','co','ct','de','fl','ga','id','il','in','ia','ks','ky',
    'la','me','md','ma','mi','mn','ms','mo','mt','ne','nv','nh','nj','nm','ny',
    'nc','nd','oh','ok','or','pa','ri','sc','sd','tn','tx','ut','vt','va','wa',
    'wv','wi','wy',
]

OUT = Path(__file__).resolve().parents[1] / 'src' / 'data' / 'usgsStations.json'


def fetch_state(st: str) -> list[dict]:
    url = (
        'https://waterservices.usgs.gov/nwis/site/'
        f'?format=rdb&stateCd={st}&siteType=ST,LK&hasDataTypeCd=iv&siteStatus=active'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'sentinel-demo/1.0'})
    with urllib.request.urlopen(req, timeout=90) as r:
        text = r.read().decode('utf-8', errors='replace')

    out: list[dict] = []
    header_seen = False
    cols: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        if not header_seen:
            cols = line.split('\t')
            header_seen = True
            continue
        if line.startswith('5s'):  # the row of column widths
            continue
        parts = line.split('\t')
        if len(parts) < len(cols):
            continue
        row = dict(zip(cols, parts))
        try:
            lat = float(row['dec_lat_va'])
            lon = float(row['dec_long_va'])
        except (KeyError, ValueError):
            continue
        if not (-130 < lon < -60 and 22 < lat < 50):
            continue
        out.append({
            'id': row.get('site_no', '').strip(),
            'name': row.get('station_nm', '').strip(),
            'lat': lat,
            'lon': lon,
        })
    return out


def main() -> None:
    seen = set()
    all_rows: list[dict] = []
    for st in STATES:
        try:
            rows = fetch_state(st)
        except Exception as e:
            print(f'  {st}: {e}', file=sys.stderr)
            continue
        added = 0
        for r in rows:
            if r['id'] in seen:
                continue
            seen.add(r['id'])
            all_rows.append(r)
            added += 1
        print(f'  {st}: +{added} (running total {len(all_rows)})')
        time.sleep(0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_rows, separators=(',', ':')))
    print(f'wrote {len(all_rows)} stations → {OUT}')


if __name__ == '__main__':
    main()
