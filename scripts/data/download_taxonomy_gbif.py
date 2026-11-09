#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Look up every source taxon name (NARS + EA) in the GBIF backbone (species/match API).

Inputs: processed('nars_streams'), processed('nars_lakes'), processed('uk_ea_ecology').
Output: raw('taxonomy_gbif')/gbif_matches.jsonl (one JSON object per (group, name) query; the
        cache is append-only and resume-safe), manifest ``taxonomy_gbif_raw``.
Requests: GET https://api.gbif.org/v1/species/match?name=..&kingdom=..&strict=false (no key,
          no personal data), <= 4 concurrent, retries with backoff.

    python -B scripts/data/download_taxonomy_gbif.py [--workers 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import sys
import threading
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import bio_common as BC  # noqa: E402
from sentinel.data import bio_taxonomy as T  # noqa: E402

log = logging.getLogger("download_taxonomy_gbif")
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"


def collect_sources() -> pd.DataFrame:
    return T.collect_sources({r: paths.processed(r) for r in ("nars_streams", "nars_lakes")},
                             paths.processed("uk_ea_ecology"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = paths.raw("taxonomy_gbif")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "download.log"), logging.StreamHandler()])
    src = collect_sources()
    q = T.query_names(src)
    cache = out / "gbif_matches.jsonl"
    done = set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["group"], r["name_clean"]))
    todo = [(g, n) for g, n in q.itertuples(index=False) if (g, n) not in done]
    log.info("%d (group, name) pairs; %d cached; %d to query", len(q), len(done), len(todo))
    lock = threading.Lock()
    local = threading.local()

    def one(pair):
        g, n = pair
        if not hasattr(local, "s"):
            local.s = BC.make_session()
        params = {"name": T.gbif_query_name(n), "strict": "false", "verbose": "false"}
        if T.GBIF_KINGDOM_HINT.get(g):
            params["kingdom"] = T.GBIF_KINGDOM_HINT[g]
        try:
            r = BC.http_request("GET", GBIF_MATCH_URL, params=params, sess=local.s, timeout=60, retries=5)
            res = r.json()
        except RuntimeError as e:
            res = {"error": str(e)[:300]}
        rec = {"group": g, "name_clean": n, "query": params, "response": res,
               "retrieved": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        with lock, open(cache, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    with cf.ThreadPoolExecutor(max_workers=max(1, min(a.workers, 4))) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if (i + 1) % 1000 == 0:
                log.info("%d / %d", i + 1, len(todo))
    # pass 2: verify source higher-rank / family values that no name match uses (spelling check)
    bb = T.load_backbone(cache)
    rv_cache = out / "gbif_rank_values.jsonl"
    rv_done = set(T.load_rank_values(rv_cache))
    rv_todo = [(r, v) for r, v in T.rank_values_to_verify(src, bb) if f"{r}|{v}" not in rv_done]
    log.info("rank values to verify: %d", len(rv_todo))

    def one_rank(pair):
        r, v = pair
        if not hasattr(local, "s"):
            local.s = BC.make_session()
        params = {"name": v.capitalize(), "rank": r.upper(), "strict": "false", "verbose": "false"}
        try:
            res = BC.http_request("GET", GBIF_MATCH_URL, params=params, sess=local.s, timeout=60, retries=5).json()
        except RuntimeError as e:
            res = {"error": str(e)[:300]}
        rec = {"rank": r, "value": v, "query": params, "response": res,
               "retrieved": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        with lock, open(rv_cache, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    with cf.ThreadPoolExecutor(max_workers=max(1, min(a.workers, 4))) as ex:
        list(ex.map(one_rank, rv_todo))
    recs = [json.loads(x) for x in cache.read_text().splitlines() if x.strip()]
    mt = pd.Series([r["response"].get("matchType", "ERROR") for r in recs]).value_counts().to_dict()
    write_manifest("taxonomy_gbif_raw", files=[cache] + ([rv_cache] if rv_cache.exists() else []),
                   sources=[GBIF_MATCH_URL],
                   description="GBIF backbone species/match responses for every cleaned source taxon name "
                               "(NARS streams/lakes, EA invertebrates/diatoms).",
                   license="GBIF Backbone Taxonomy: CC BY 4.0 (GBIF Secretariat)",
                   stats={"retrieved": dt.date.today().isoformat(), "n_queries": len(recs), "match_types": mt})
    log.info("done: %s", mt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
