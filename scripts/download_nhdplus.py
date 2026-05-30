"""
Download NHDPlusV2 stream network data and build graph for the Stream GNN.

Uses two USGS APIs:
1. NLDI (Network Linked Data Index) -- maps USGS sites to NHDPlus COMIDs
   and discovers upstream/downstream connectivity.
   Base: https://api.water.usgs.gov/nldi/linked-data

2. Fabric pygeoapi -- fetches NHDPlusV2 flowline value-added attributes
   (stream order, drainage area, velocity, travel time, etc.)
   Base: https://api.water.usgs.gov/fabric/pygeoapi

The output graph is compatible with sentinel.models.graph.stream_gnn.build_stream_graph().

Usage:
    python scripts/download_nhdplus.py \\
        --site-catalog data/raw/sensor/full/station_catalog_smart.json

    python scripts/download_nhdplus.py \\
        --site-catalog data/raw/sensor/full/station_catalog_smart.json \\
        --max-sites 50
"""

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("download_nhdplus")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
FABRIC_BASE = "https://api.water.usgs.gov/fabric/pygeoapi/collections"
NWIS_SITE_SERVICE = "https://waterservices.usgs.gov/nwis/site/"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between API calls
_last_request_time = 0.0


def _rate_limited_get(
    url: str,
    params: Optional[dict] = None,
    timeout: int = 30,
    retries: int = 3,
) -> Optional[requests.Response]:
    """GET request with rate limiting, retries, and error handling."""
    global _last_request_time
    for attempt in range(retries):
        elapsed = time.time() - _last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        _last_request_time = time.time()
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "SENTINEL-WQ/1.0",
                },
            )
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = min(30, 2 ** (attempt + 1))
                log.warning(f"  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            elif resp.status_code == 404:
                log.debug(f"  Not found (404): {url}")
                return None
            else:
                log.warning(f"  HTTP {resp.status_code} for {url}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
        except requests.exceptions.Timeout:
            log.warning(f"  Timeout (attempt {attempt + 1}/{retries}): {url}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.exceptions.RequestException as e:
            log.warning(f"  Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------
class APICache:
    """Simple file-based JSON cache for API responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key_path(self, namespace: str, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "_").replace("?", "_")
        ns_dir = self.cache_dir / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{safe_key}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        path = self._key_path(namespace, key)
        if path.exists():
            self._hits += 1
            with open(path) as f:
                return json.load(f)
        self._misses += 1
        return None

    def put(self, namespace: str, key: str, data: Any):
        path = self._key_path(namespace, key)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def stats(self) -> str:
        total = self._hits + self._misses
        if total == 0:
            return "Cache: 0 requests"
        return f"Cache: {self._hits}/{total} hits ({100 * self._hits / total:.0f}%)"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + (
        math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Step 1: Get site info (lat/lon) from NWIS
# ---------------------------------------------------------------------------
def get_site_info_batch(site_nos: List[str], cache: APICache) -> Dict[str, dict]:
    """Fetch lat/lon and basic info for sites from NWIS site service.

    Uses batch queries (up to 100 sites at a time) to reduce API calls.
    """
    result = {}
    uncached = []

    for sno in site_nos:
        cached = cache.get("site_info", sno)
        if cached is not None:
            result[sno] = cached
        else:
            uncached.append(sno)

    if not uncached:
        log.info(f"  All {len(site_nos)} sites found in cache")
        return result

    log.info(f"  Fetching info for {len(uncached)} sites ({len(result)} cached)...")

    batch_size = 100
    for i in range(0, len(uncached), batch_size):
        batch = uncached[i : i + batch_size]
        sites_param = ",".join(batch)
        resp = _rate_limited_get(
            NWIS_SITE_SERVICE,
            params={
                "format": "rdb",
                "sites": sites_param,
                "siteOutput": "expanded",
                "siteStatus": "all",
            },
            timeout=60,
        )

        if resp is None:
            log.warning(f"  Failed to fetch batch {i // batch_size + 1}")
            continue

        # Parse RDB (tab-separated with comment headers)
        lines = resp.text.strip().split("\n")
        header_idx = None
        headers = []
        for line in lines:
            if line.startswith("#"):
                continue
            if header_idx is None:
                header_idx = True
                headers = line.split("\t")
                continue
            # Skip the format specification line (e.g. "5s 30s ...")
            if all(c in "0123456789sdnc \t" for c in line.strip()[:20]):
                continue
            fields = line.split("\t")
            if len(fields) < len(headers):
                continue

            row = dict(zip(headers, fields))
            sno = row.get("site_no", "").strip()
            if not sno:
                continue

            info = {
                "site_no": sno,
                "station_nm": row.get("station_nm", ""),
                "lat": _safe_float(row.get("dec_lat_va")),
                "lon": _safe_float(row.get("dec_long_va")),
                "drain_area_sq_mi": _safe_float(row.get("drain_area_va")),
                "huc_cd": row.get("huc_cd", ""),
                "state_cd": row.get("state_cd", ""),
                "site_type": row.get("site_tp_cd", ""),
            }
            result[sno] = info
            cache.put("site_info", sno, info)

        log.info(f"  Batch {i // batch_size + 1}: {len(result)} sites total so far")

    return result


# ---------------------------------------------------------------------------
# Step 2: Map sites to COMIDs via NLDI
# ---------------------------------------------------------------------------
def get_comid_for_site(site_no: str, cache: APICache) -> Optional[dict]:
    """Query NLDI to get the NHDPlus COMID for a USGS site.

    Endpoint: GET /linked-data/nwissite/USGS-{site_no}
    Returns dict with comid, lat, lon, name, or None if the site is not
    indexed in NLDI (e.g. tidal sites, springs, or very small streams).
    """
    cached = cache.get("nldi_comid", site_no)
    if cached is not None:
        return cached

    url = f"{NLDI_BASE}/nwissite/USGS-{site_no}"
    resp = _rate_limited_get(url)
    if resp is None:
        return None

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        log.warning(f"  Invalid JSON for site {site_no}")
        return None

    # NLDI returns a GeoJSON FeatureCollection
    features = data.get("features", [])
    if not features:
        log.debug(f"  No features in NLDI response for {site_no}")
        return None

    feat = features[0]
    props = feat.get("properties", {})
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [None, None])

    comid_val = props.get("comid")
    if comid_val is None:
        log.debug(f"  No comid in NLDI response for {site_no}")
        return None

    result = {
        "site_no": site_no,
        "comid": int(comid_val),
        "lon": coords[0] if coords else None,
        "lat": coords[1] if coords else None,
        "name": props.get("name", ""),
        "source": props.get("source", ""),
        "uri": props.get("uri", ""),
        "reachcode": props.get("reachcode", ""),
        "measure": _safe_float(props.get("measure")),
    }
    cache.put("nldi_comid", site_no, result)
    return result


def map_sites_to_comids(
    site_nos: List[str], cache: APICache
) -> Dict[str, dict]:
    """Map all sites to COMIDs. Returns {site_no: {comid, lat, lon, ...}}."""
    mapped = {}
    failed = []

    for i, sno in enumerate(site_nos):
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  NLDI lookup: {i + 1}/{len(site_nos)}...")

        result = get_comid_for_site(sno, cache)
        if result is not None and result.get("comid"):
            mapped[sno] = result
        else:
            failed.append(sno)

    log.info(f"  NLDI mapping complete: {len(mapped)} mapped, {len(failed)} failed")
    if failed:
        log.info(f"  Failed sites (first 20): {failed[:20]}")

    return mapped


# ---------------------------------------------------------------------------
# Step 3: Fetch NHDPlus value-added attributes from Fabric pygeoapi
# ---------------------------------------------------------------------------
def get_flowline_attributes(comid: int, cache: APICache) -> Optional[dict]:
    """Get NHDPlus value-added attributes for a COMID from the Fabric pygeoapi.

    Endpoint: GET /collections/nhdflowline_network/items?comid={comid}

    Returns rich attributes including:
        streamorde, totdasqkm, lengthkm, va_ma (mean annual velocity fps),
        qa_ma (mean annual flow cfs), slope, hydroseq, dnhydroseq, etc.
    """
    cached = cache.get("flowline_attrs", str(comid))
    if cached is not None:
        return cached

    url = f"{FABRIC_BASE}/nhdflowline_network/items"
    resp = _rate_limited_get(url, params={"comid": str(comid), "f": "json"})
    if resp is None:
        return None

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None

    features = data.get("features", [])
    if not features:
        return None

    props = features[0].get("properties", {})

    result = {
        "comid": comid,
        "gnis_name": props.get("gnis_name", ""),
        "lengthkm": _safe_float(props.get("lengthkm")),
        "streamorde": _safe_float(props.get("streamorde")),
        "streamleve": _safe_float(props.get("streamleve")),
        "totdasqkm": _safe_float(props.get("totdasqkm")),
        "areasqkm": _safe_float(props.get("areasqkm")),
        "divdasqkm": _safe_float(props.get("divdasqkm")),
        # EROM mean annual values
        "qa_ma": _safe_float(props.get("qa_ma")),  # flow, cfs
        "va_ma": _safe_float(props.get("va_ma")),  # velocity, ft/s
        # Hydrologic sequence (for topology)
        "hydroseq": _safe_float(props.get("hydroseq")),
        "dnhydroseq": _safe_float(props.get("dnhydroseq")),
        "uphydroseq": _safe_float(props.get("uphydroseq")),
        # Network position
        "levelpathi": _safe_float(props.get("levelpathi")),
        "pathlength": _safe_float(props.get("pathlength")),
        "pathtimema": _safe_float(props.get("pathtimema")),
        "arbolatesu": _safe_float(props.get("arbolatesu")),
        # Topology
        "fromnode": _safe_float(props.get("fromnode")),
        "tonode": _safe_float(props.get("tonode")),
        "divergence": _safe_float(props.get("divergence")),
        "startflag": _safe_float(props.get("startflag")),
        "terminalfl": _safe_float(props.get("terminalfl")),
        # Elevation and slope
        "slope": _safe_float(props.get("slope")),
        "maxelevsmo": _safe_float(props.get("maxelevsmo")),
        "minelevsmo": _safe_float(props.get("minelevsmo")),
        "tidal": _safe_float(props.get("tidal")),
        "ftype": props.get("ftype", ""),
        "reachcode": props.get("reachcode", ""),
    }
    cache.put("flowline_attrs", str(comid), result)
    return result


def fetch_all_flowline_attributes(
    comids: List[int], cache: APICache
) -> Dict[int, dict]:
    """Fetch NHDPlus attributes for all COMIDs, with progress logging."""
    attrs = {}
    for i, comid in enumerate(comids):
        if (i + 1) % 25 == 0 or i == 0:
            log.info(f"  Flowline attributes: {i + 1}/{len(comids)}...")
        result = get_flowline_attributes(comid, cache)
        if result:
            attrs[comid] = result
    log.info(f"  Got attributes for {len(attrs)}/{len(comids)} COMIDs")
    return attrs


# ---------------------------------------------------------------------------
# Step 4: Discover network connectivity via NLDI navigation
# ---------------------------------------------------------------------------
def get_downstream_sites(
    site_no: str,
    cache: APICache,
    distance_km: float = 200.0,
) -> List[dict]:
    """Get downstream NWIS sites from a given site using NLDI navigation.

    Endpoint: GET /linked-data/nwissite/USGS-{site_no}/navigation/DM/nwissite

    Args:
        site_no: USGS site number.
        distance_km: Max downstream navigation distance in km.
        cache: API cache.

    Returns:
        List of downstream site dicts with {site_no, comid, lat, lon}.
    """
    cache_key = f"{site_no}_DM_{int(distance_km)}"
    cached = cache.get("nldi_nav", cache_key)
    if cached is not None:
        return cached

    url = f"{NLDI_BASE}/nwissite/USGS-{site_no}/navigation/DM/nwissite"
    resp = _rate_limited_get(url, params={"distance": str(distance_km)})

    if resp is None:
        cache.put("nldi_nav", cache_key, [])
        return []

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        cache.put("nldi_nav", cache_key, [])
        return []

    features = data.get("features", [])
    downstream = []
    for feat in features:
        props = feat.get("properties", {})
        identifier = props.get("identifier", "")
        # identifier format: "USGS-01302020"
        if identifier.startswith("USGS-"):
            ds_site_no = identifier.replace("USGS-", "")
            if ds_site_no == site_no:
                continue  # Skip self
            ds_comid = props.get("comid")
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            downstream.append(
                {
                    "site_no": ds_site_no,
                    "comid": int(ds_comid) if ds_comid else None,
                    "lon": coords[0] if coords else None,
                    "lat": coords[1] if coords else None,
                    "name": props.get("name", ""),
                }
            )

    cache.put("nldi_nav", cache_key, downstream)
    return downstream


# ---------------------------------------------------------------------------
# Step 5: Build the stream network graph
# ---------------------------------------------------------------------------
def build_network_graph(
    site_comid_map: Dict[str, dict],
    site_info: Dict[str, dict],
    cache: APICache,
    max_downstream_km: float = 200.0,
) -> Tuple[List[dict], List[dict]]:
    """Build the stream network graph.

    For each mapped site, navigates downstream to find other mapped sites,
    then uses NHDPlus value-added attributes for edge properties.

    Args:
        site_comid_map: {site_no: {comid, lat, lon, ...}} from NLDI mapping.
        site_info: {site_no: {lat, lon, drain_area_sq_mi, ...}} from NWIS.
        cache: API cache.
        max_downstream_km: Maximum downstream navigation distance.

    Returns:
        (nodes, edges) where:
            nodes: [{site_id, comid, lat, lon, drainage_area_km2, stream_order}]
            edges: [{from_site, to_site, travel_time_hours, distance_km, stream_order}]
    """
    known_sites = set(site_comid_map.keys())

    # --- Fetch NHDPlus value-added attributes for all COMIDs ---
    log.info("Fetching NHDPlus flowline attributes from Fabric pygeoapi...")
    comids = list(
        set(v["comid"] for v in site_comid_map.values() if v.get("comid"))
    )
    flowline_attrs = fetch_all_flowline_attributes(comids, cache)

    # --- Discover downstream connections via NLDI navigation ---
    log.info("Discovering downstream connections via NLDI navigation...")
    raw_edges: List[Tuple[str, str]] = []
    site_list = sorted(site_comid_map.keys())

    for i, site_no in enumerate(site_list):
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"  Navigation: {i + 1}/{len(site_list)}...")

        comid = site_comid_map[site_no].get("comid")
        if not comid:
            continue

        downstream = get_downstream_sites(
            site_no, cache, distance_km=max_downstream_km
        )

        for ds in downstream:
            ds_site_no = ds.get("site_no")
            if ds_site_no and ds_site_no in known_sites and ds_site_no != site_no:
                raw_edges.append((site_no, ds_site_no))

    log.info(f"  Found {len(raw_edges)} raw downstream connections")

    # Deduplicate and remove transitive edges
    direct_edges = _find_direct_edges(raw_edges)
    log.info(
        f"  After pruning transitive edges: {len(direct_edges)} direct connections"
    )

    # --- Build nodes ---
    nodes = []
    for site_no, nldi_info in site_comid_map.items():
        comid = nldi_info.get("comid")
        fa = flowline_attrs.get(comid, {})
        nwis = site_info.get(site_no, {})

        # Coordinates: prefer NLDI (snapped to flowline), fall back to NWIS
        lat = nldi_info.get("lat") or nwis.get("lat")
        lon = nldi_info.get("lon") or nwis.get("lon")

        # Drainage area: prefer NHDPlus totdasqkm, fall back to NWIS (sq mi -> sq km)
        drainage_area_km2 = fa.get("totdasqkm")
        if drainage_area_km2 is None:
            drain_sq_mi = nwis.get("drain_area_sq_mi")
            if drain_sq_mi:
                drainage_area_km2 = drain_sq_mi * 2.58999

        stream_order = fa.get("streamorde")
        # NHDPlus uses negative values (e.g. -9) as sentinel for unknown
        if stream_order is not None and stream_order < 0:
            stream_order = None

        nodes.append(
            {
                "site_id": site_no,
                "comid": comid,
                "lat": lat,
                "lon": lon,
                "drainage_area_km2": (
                    round(drainage_area_km2, 2) if drainage_area_km2 else None
                ),
                "stream_order": (
                    int(stream_order) if stream_order else None
                ),
                "station_name": nldi_info.get("name", "")
                or nwis.get("station_nm", ""),
            }
        )

    # --- Build edges with travel time estimates ---
    edges = []
    for from_site, to_site in direct_edges:
        from_comid = site_comid_map[from_site].get("comid")
        to_comid = site_comid_map[to_site].get("comid")
        from_fa = flowline_attrs.get(from_comid, {})
        to_fa = flowline_attrs.get(to_comid, {})

        # --- Distance ---
        # Method 1: Use NHDPlus pathlength difference (network distance to outlet)
        distance_km = None
        from_pathlength = from_fa.get("pathlength")
        to_pathlength = to_fa.get("pathlength")
        if from_pathlength is not None and to_pathlength is not None:
            # pathlength = distance from reach to network terminal (km)
            # Upstream site has larger pathlength than downstream site
            dist = from_pathlength - to_pathlength
            if dist > 0:
                distance_km = round(dist, 3)

        # Method 2: Fall back to haversine with sinuosity factor
        if not distance_km or distance_km <= 0:
            from_info = site_comid_map[from_site]
            to_info = site_comid_map[to_site]
            if from_info.get("lat") and to_info.get("lat"):
                straight_line = _haversine(
                    from_info["lat"],
                    from_info["lon"],
                    to_info["lat"],
                    to_info["lon"],
                )
                distance_km = round(straight_line * 1.3, 3)  # sinuosity

        # --- Velocity ---
        # Use EROM mean annual velocity (ft/s -> km/h)
        velocity_fps = from_fa.get("va_ma")
        if velocity_fps and velocity_fps > 0:
            velocity_kmh = velocity_fps * 0.3048 * 3.6  # ft/s -> km/h
        else:
            velocity_kmh = 1.8  # Default: ~0.5 m/s

        # --- Travel time ---
        travel_time_hours = None
        if distance_km and distance_km > 0:
            travel_time_hours = round(distance_km / velocity_kmh, 2)

        # --- Stream order ---
        stream_order = from_fa.get("streamorde") or to_fa.get("streamorde")
        if stream_order:
            stream_order = int(stream_order)
        else:
            stream_order = 0

        edges.append(
            {
                "from_site": from_site,
                "to_site": to_site,
                "from_comid": from_comid,
                "to_comid": to_comid,
                "travel_time_hours": travel_time_hours,
                "distance_km": round(distance_km, 2) if distance_km else None,
                "stream_order": stream_order,
            }
        )

    return nodes, edges


def _find_direct_edges(
    raw_edges: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    """Remove transitive edges, keeping only direct upstream-downstream connections.

    If A->B and A->C and B->C all exist as raw edges, remove A->C since
    the connection goes through B.
    """
    downstream_of: Dict[str, set] = defaultdict(set)
    for from_s, to_s in raw_edges:
        downstream_of[from_s].add(to_s)

    direct = set()
    for from_s, to_s in raw_edges:
        # Check if any intermediate site exists: from_s -> X -> to_s
        is_transitive = False
        for intermediate in downstream_of[from_s]:
            if intermediate != to_s and to_s in downstream_of.get(
                intermediate, set()
            ):
                is_transitive = True
                break
        if not is_transitive:
            direct.add((from_s, to_s))

    return sorted(direct)


# ---------------------------------------------------------------------------
# Step 6: Save outputs
# ---------------------------------------------------------------------------
def save_outputs(
    nodes: List[dict],
    edges: List[dict],
    site_comid_map: Dict[str, dict],
    flowline_attrs: Dict[int, dict],
    output_dir: Path,
) -> dict:
    """Save the graph and mapping files."""
    processed_dir = output_dir / "processed" / "hydrology"
    raw_dir = output_dir / "raw" / "hydrology" / "nhdplus"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main graph file (compatible with build_stream_graph)
    graph_data = {
        "description": "NHDPlusV2 stream network graph for SENTINEL Stream GNN",
        "sources": {
            "nldi": f"{NLDI_BASE} (site-COMID mapping, navigation)",
            "fabric": f"{FABRIC_BASE}/nhdflowline_network (flowline attributes)",
        },
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
    graph_path = processed_dir / "nhdplus_graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    log.info(f"Saved graph: {graph_path} ({len(nodes)} nodes, {len(edges)} edges)")

    # 2. COMID mapping file
    comid_mapping = {}
    for site_no, info in site_comid_map.items():
        comid_mapping[site_no] = {
            "comid": info.get("comid"),
            "lat": info.get("lat"),
            "lon": info.get("lon"),
            "reachcode": info.get("reachcode"),
            "measure": info.get("measure"),
        }
    mapping_path = processed_dir / "comid_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(comid_mapping, f, indent=2)
    log.info(f"Saved COMID mapping: {mapping_path} ({len(comid_mapping)} sites)")

    # 3. Raw NLDI + Fabric data
    raw_nldi_path = raw_dir / "nldi_site_comid_map.json"
    with open(raw_nldi_path, "w") as f:
        json.dump(site_comid_map, f, indent=2)
    log.info(f"Saved raw NLDI data: {raw_nldi_path}")

    # Save serializable flowline attrs (convert int keys to str)
    raw_fabric_path = raw_dir / "fabric_flowline_attrs.json"
    with open(raw_fabric_path, "w") as f:
        json.dump({str(k): v for k, v in flowline_attrs.items()}, f, indent=2)
    log.info(f"Saved raw Fabric data: {raw_fabric_path}")

    # 4. Summary statistics
    connected_sites = set()
    for e in edges:
        connected_sites.add(e["from_site"])
        connected_sites.add(e["to_site"])

    travel_times = [
        e["travel_time_hours"] for e in edges if e.get("travel_time_hours")
    ]
    distances = [e["distance_km"] for e in edges if e.get("distance_km")]
    stream_orders = [
        n["stream_order"] for n in nodes if n.get("stream_order")
    ]

    stats = {
        "total_sites_in_catalog": len(site_comid_map),
        "sites_mapped_to_comid": sum(
            1 for v in site_comid_map.values() if v.get("comid")
        ),
        "sites_in_graph": len(nodes),
        "sites_connected": len(connected_sites),
        "sites_isolated": len(nodes) - len(connected_sites),
        "edges": len(edges),
        "flowline_attrs_fetched": len(flowline_attrs),
        "travel_time_hours": {
            "min": round(min(travel_times), 2) if travel_times else None,
            "max": round(max(travel_times), 2) if travel_times else None,
            "mean": (
                round(sum(travel_times) / len(travel_times), 2)
                if travel_times
                else None
            ),
        },
        "distance_km": {
            "min": round(min(distances), 2) if distances else None,
            "max": round(max(distances), 2) if distances else None,
            "mean": (
                round(sum(distances) / len(distances), 2) if distances else None
            ),
        },
        "stream_order": {
            "min": min(stream_orders) if stream_orders else None,
            "max": max(stream_orders) if stream_orders else None,
            "distribution": (
                {
                    str(o): stream_orders.count(o)
                    for o in sorted(set(stream_orders))
                }
                if stream_orders
                else {}
            ),
        },
    }
    stats_path = processed_dir / "nhdplus_graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Saved stats: {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download NHDPlusV2 stream network data and build graph for Stream GNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_nhdplus.py \\
      --site-catalog data/raw/sensor/full/station_catalog_smart.json

  python scripts/download_nhdplus.py \\
      --site-catalog data/raw/sensor/full/station_catalog_smart.json \\
      --max-sites 50 --output-dir data/
        """,
    )
    parser.add_argument(
        "--site-catalog",
        required=True,
        help="Path to station_catalog_smart.json from sensor download",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Limit number of sites to process (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory (default: data/)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="API response cache directory (default: <output-dir>/raw/hydrology/nhdplus/cache)",
    )
    parser.add_argument(
        "--max-downstream-km",
        type=float,
        default=200.0,
        help="Maximum downstream navigation distance in km (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else output_dir / "raw" / "hydrology" / "nhdplus" / "cache"
    )

    # Load site catalog
    log.info(f"Loading site catalog: {args.site_catalog}")
    with open(args.site_catalog) as f:
        catalog = json.load(f)
    log.info(f"  Loaded {len(catalog)} sites")

    site_nos = [entry["site_no"] for entry in catalog]
    if args.max_sites:
        site_nos = site_nos[: args.max_sites]
        log.info(f"  Limited to {len(site_nos)} sites (--max-sites {args.max_sites})")

    # Initialize cache
    cache = APICache(cache_dir)

    # ---- Step 1: Get site info (lat/lon) from NWIS ----
    log.info("=" * 60)
    log.info("Step 1: Fetching site info from NWIS...")
    site_info = get_site_info_batch(site_nos, cache)
    log.info(f"  Got info for {len(site_info)}/{len(site_nos)} sites")

    # ---- Step 2: Map sites to COMIDs via NLDI ----
    log.info("=" * 60)
    log.info("Step 2: Mapping sites to NHDPlus COMIDs via NLDI...")
    site_comid_map = map_sites_to_comids(site_nos, cache)

    # Merge NWIS site info into COMID map (fill missing lat/lon)
    for sno, info in site_comid_map.items():
        nwis = site_info.get(sno, {})
        if info.get("lat") is None and nwis.get("lat"):
            info["lat"] = nwis["lat"]
        if info.get("lon") is None and nwis.get("lon"):
            info["lon"] = nwis["lon"]

    mapped_count = sum(
        1 for v in site_comid_map.values() if v.get("comid")
    )
    log.info(f"  Successfully mapped {mapped_count}/{len(site_nos)} sites to COMIDs")

    if mapped_count == 0:
        log.error(
            "No sites could be mapped to COMIDs. "
            "Check network connectivity and NLDI availability."
        )
        sys.exit(1)

    # ---- Step 3: Build stream network graph ----
    log.info("=" * 60)
    log.info("Step 3: Building stream network graph...")
    nodes, edges = build_network_graph(
        site_comid_map,
        site_info,
        cache,
        max_downstream_km=args.max_downstream_km,
    )

    # Collect flowline attrs for saving (re-fetch from cache, no extra API calls)
    comids = list(
        set(v["comid"] for v in site_comid_map.values() if v.get("comid"))
    )
    flowline_attrs = {}
    for comid in comids:
        fa = cache.get("flowline_attrs", str(comid))
        if fa:
            flowline_attrs[comid] = fa

    # ---- Step 4: Save outputs ----
    log.info("=" * 60)
    log.info("Step 4: Saving outputs...")
    stats = save_outputs(
        nodes, edges, site_comid_map, flowline_attrs, output_dir
    )

    # ---- Summary ----
    log.info("=" * 60)
    log.info("NHDPlus download complete!")
    log.info(f"  Sites in catalog:     {len(site_nos)}")
    log.info(f"  Sites mapped (COMID): {stats['sites_mapped_to_comid']}")
    log.info(f"  Sites in graph:       {stats['sites_in_graph']}")
    log.info(f"  Sites connected:      {stats['sites_connected']}")
    log.info(f"  Sites isolated:       {stats['sites_isolated']}")
    log.info(f"  Network edges:        {stats['edges']}")
    log.info(f"  Flowline attrs:       {stats['flowline_attrs_fetched']}")
    tt = stats["travel_time_hours"]
    if tt.get("mean"):
        log.info(f"  Travel time (hours):  {tt['min']:.1f} / {tt['mean']:.1f} / {tt['max']:.1f}  (min/mean/max)")
    dk = stats["distance_km"]
    if dk.get("mean"):
        log.info(f"  Distance (km):        {dk['min']:.1f} / {dk['mean']:.1f} / {dk['max']:.1f}  (min/mean/max)")
    so = stats["stream_order"]
    if so.get("distribution"):
        log.info(f"  Stream orders:        {so['distribution']}")
    log.info(f"  {cache.stats()}")
    log.info(f"  Output: {output_dir / 'processed' / 'hydrology'}/")


if __name__ == "__main__":
    main()
