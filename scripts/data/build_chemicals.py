#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build processed('chemicals'): CAS -> DTXSID -> canonical SMILES / InChIKey / MW for ChemTox.

CAS universe = ADORE (all 2,408 chemicals) + ECOTOX aquatic tests (organism_habitat == 'Water')
+ NRC materials with a valid CAS. Identifiers ECOTOX and ADORE already carry are seeds; each CAS
is then looked up in the EPA CompTox dashboard search (keyless) and, only when CompTox has no
record at all, in PubChem PUG-REST. Lookups are cached in $SENTINEL_DATA/interim/chemicals/
(JSONL, resumable), <= 4 requests/s per host. RDKit recomputes canonical SMILES, InChIKey and
MolWt from the source SMILES and assigns a structure class (organic / organic_multicomponent /
organometallic / inorganic / no_structure / unresolved).

Run after download_adore, download_ecotox and build_nrc; afterwards re-run build_ecotox and
build_nrc so their tables pick up identities and molar conversions.

    python -B scripts/data/build_chemicals.py [--workers 3] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.data import chemtox_http as H  # noqa: E402
from sentinel.data.sources import adore as A  # noqa: E402
from sentinel.data.sources import chem_ids as C  # noqa: E402
from sentinel.data.sources import ecotox as E  # noqa: E402

log = logging.getLogger("build_chemicals")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def adore_seeds() -> pd.DataFrame:
    cols = [
        "test_cas",
        "chem_dtxsid",
        "chem_name",
        "chem_inchikey",
        "chem_rdkit_can_smiles",
        "chem_pcp_can_smiles",
        "chem_mw",
    ]
    df = A.load_challenge("a-FCA2FCA", columns=cols)[cols].drop_duplicates("test_cas")
    df["cas"] = df["test_cas"].map(C.normalize_cas)
    return df.rename(
        columns={
            "chem_dtxsid": "adore_dtxsid",
            "chem_name": "adore_name",
            "chem_inchikey": "adore_inchikey",
            "chem_rdkit_can_smiles": "adore_smiles",
            "chem_pcp_can_smiles": "adore_pcp_smiles",
            "chem_mw": "adore_mw",
        }
    ).drop(columns="test_cas")


def ecotox_seeds() -> tuple[pd.DataFrame, set[str]]:
    chem = E.read_table("chemicals")
    chem["cas"] = chem["cas_number"].map(C.normalize_cas)
    chem = chem.rename(
        columns={
            "chemical_name": "ecotox_name",
            "dtxsid": "ecotox_dtxsid",
            "ecotox_group": "ecotox_chem_group",
        }
    )[["cas", "cas_number", "ecotox_name", "ecotox_dtxsid", "ecotox_chem_group"]]
    tests = E.read_table("tests", columns=["test_cas", "organism_habitat"])
    aq = (
        tests.loc[tests["organism_habitat"].eq("Water").fillna(False).to_numpy(dtype=bool), "test_cas"]
        .dropna()
        .unique()
    )
    aq_cas = {c for c in (C.normalize_cas(x) for x in aq) if c}
    return chem.dropna(subset=["cas"]).drop_duplicates("cas"), aq_cas


def nrc_cas() -> set[str]:
    p = paths.PROCESSED / "nrc_incidents" / "materials.parquet"
    if not p.exists():
        log.warning("NRC materials not built yet; NRC CAS numbers not included")
        return set()
    m = pd.read_parquet(p, columns=["cas"])
    return set(m["cas"].dropna().unique())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=0.25, help="seconds between requests per host")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N CAS numbers")
    a = ap.parse_args()
    out = paths.processed("chemicals")
    cache_dir = paths.interim("chemicals")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "build.log"), logging.StreamHandler()],
    )
    ad = adore_seeds()
    ec, ec_aq = ecotox_seeds()
    nr = nrc_cas()
    ad_set = set(ad["cas"].dropna())
    universe = sorted(ad_set | ec_aq | nr)
    log.info(
        "CAS universe: %d (ADORE %d, ECOTOX aquatic %d, NRC %d)",
        len(universe),
        len(ad_set),
        len(ec_aq),
        len(nr),
    )
    if a.limit:
        universe = universe[: a.limit]
    seeds = pd.DataFrame({"cas": universe}).merge(ec, on="cas", how="left").merge(ad, on="cas", how="left")
    seeds["in_adore"] = seeds["cas"].isin(ad_set)
    seeds["in_ecotox_aquatic"] = seeds["cas"].isin(ec_aq)
    seeds["in_nrc"] = seeds["cas"].isin(nr)

    ct_cache = C.JsonlCache(cache_dir / "comptox_cache.jsonl")
    pc_cache = C.JsonlCache(cache_dir / "pubchem_cache.jsonl")
    s = H.make_session()
    ct_rl, pc_rl = H.RateLimiter(a.min_interval), H.RateLimiter(a.min_interval)
    lock = threading.Lock()
    done = {"n": 0}

    failures = {"comptox": [], "pubchem": []}

    def comptox(cas: str) -> None:
        if cas in ct_cache:
            return
        try:
            r = H.get(C.comptox_url(cas), session=s, rate_limiter=ct_rl, pass_status=(400, 404), timeout=60)
        except RuntimeError as e:  # not cached -> retried on the next run
            log.warning("CompTox %s failed: %s", cas, e)
            failures["comptox"].append(cas)
            return
        payload = r.json() if r.status_code == 200 and r.text.strip() else None
        with lock:
            ct_cache.put(cas, r.status_code, payload, now())
            done["n"] += 1
            if done["n"] % 500 == 0:
                log.info("CompTox lookups: %d", done["n"])

    todo = [c for c in universe if c not in ct_cache]
    log.info("CompTox: %d cached, %d to fetch", len(universe) - len(todo), len(todo))
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(comptox, todo))
    ct = {cas: C.parse_comptox_hits((ct_cache.get(cas) or {}).get("payload"), cas) for cas in universe}
    need_pc = [cas for cas in universe if not ct[cas]["dtxsid"]]
    log.info("PubChem fallback for %d CAS without a CompTox record", len(need_pc))

    def pubchem(cas: str) -> None:
        if cas in pc_cache:
            return
        try:
            r = H.get(C.pubchem_url(cas), session=s, rate_limiter=pc_rl, pass_status=(400, 404), timeout=60)
        except RuntimeError as e:
            log.warning("PubChem %s failed: %s", cas, e)
            failures["pubchem"].append(cas)
            return
        payload = r.json() if r.status_code == 200 else None
        with lock:
            pc_cache.put(cas, r.status_code, payload, now())

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(pubchem, [c for c in need_pc if c not in pc_cache]))
    rows = []
    seed_cols = [c for c in seeds.columns if c != "cas"]
    for rec in seeds.to_dict("records"):
        cas = rec["cas"]
        pc = C.parse_pubchem((pc_cache.get(cas) or {}).get("payload")) if cas in need_pc else None
        sd = {
            k: (None if (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in rec.items()
            if k in seed_cols
        }
        rows.append(C.resolve_identity(cas, ct[cas], pc, sd))
    df = C.identity_frame(rows)
    df["has_smiles"] = df["smiles_canonical"].notna()
    df["is_multicomponent"] = df["n_components"].fillna(0).astype(int) > 1
    df["is_inorganic"] = df["structure_class"].eq("inorganic")
    df["no_structure"] = df["structure_class"].isin(["no_structure", "unresolved"])
    df["comptox_lookup_status"] = [(ct_cache.get(c) or {}).get("status") for c in df["cas"]]
    df["pubchem_lookup_status"] = [
        (pc_cache.get(c) or {}).get("status") if c in need_pc else None for c in df["cas"]
    ]
    # QC: agreement with identifiers ADORE already carries
    both = df["adore_inchikey"].notna() & df["inchikey"].notna()
    ik_agree = df.loc[both, "adore_inchikey"] == df.loc[both, "inchikey"]
    ik_block = df.loc[both, "adore_inchikey"].str[:14] == df.loc[both, "inchikey"].str[:14]
    dt_both = df["adore_dtxsid"].notna() & df["dtxsid_final"].notna()
    dt_agree = df.loc[dt_both, "adore_dtxsid"] == df.loc[dt_both, "dtxsid_final"]
    for c in df.columns:
        if df[c].dtype == object:
            nonnull = df[c].dropna()
            if len(nonnull) and nonnull.map(lambda v: isinstance(v, bool)).all():
                df[c] = df[c].astype("boolean")
            else:
                df[c] = df[c].astype("string")
    df.to_parquet(out / "chem_ids.parquet", index=False, compression="zstd")
    summary = {
        "n_cas": int(len(df)),
        "n_in_adore": int(df["in_adore"].sum()),
        "n_in_ecotox_aquatic": int(df["in_ecotox_aquatic"].sum()),
        "n_in_nrc": int(df["in_nrc"].sum()),
        "with_dtxsid": int(df["dtxsid_final"].notna().sum()),
        "with_smiles": int(df["has_smiles"].sum()),
        "structure_class": {str(k): int(v) for k, v in df["structure_class"].value_counts().items()},
        "structure_source": {
            str(k): int(v) for k, v in df["structure_source"].value_counts(dropna=False).items()
        },
        "structure_class_by_source_set": {
            name: {str(k): int(v) for k, v in df.loc[df[col], "structure_class"].value_counts().items()}
            for name, col in (
                ("adore", "in_adore"),
                ("ecotox_aquatic", "in_ecotox_aquatic"),
                ("nrc", "in_nrc"),
            )
        },
        "comptox_match": {str(k): int(v) for k, v in df["comptox_match"].value_counts(dropna=False).items()},
        "dtxsid_conflict_vs_ecotox": int(df["dtxsid_conflict"].sum()),
        "adore_inchikey_full_agreement": float(ik_agree.mean()) if len(ik_agree) else None,
        "adore_inchikey_connectivity_agreement": float(ik_block.mean()) if len(ik_block) else None,
        "adore_inchikey_compared": int(both.sum()),
        "adore_dtxsid_agreement": float(dt_agree.mean()) if len(dt_agree) else None,
        "adore_dtxsid_compared": int(dt_both.sum()),
        "rdkit_parse_failures": int(
            (df["smiles_source"].notna() & ~df["rdkit_parse_ok"].fillna(False).astype(bool)).sum()
        ),
        "retrieved_utc": now(),
        "lookup_failures": {k: len(v) for k, v in failures.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary))
    files = [out / "chem_ids.parquet", out / "summary.json", cache_dir / "comptox_cache.jsonl"]
    if (cache_dir / "pubchem_cache.jsonl").exists():
        files.append(cache_dir / "pubchem_cache.jsonl")
    write_manifest(
        "chemicals",
        files=files,
        sources=[
            C.COMPTOX_SEARCH_URL + "<CAS>",
            C.PUBCHEM_NAME_URL.replace("{q}", "<CAS>"),
            "manifest:adore_raw",
            "manifest:ecotox_raw",
            "manifest:nrc_incidents",
        ],
        description="CAS -> DTXSID -> RDKit canonical SMILES / InChIKey / MW with structure-class "
        "flags for every chemical in ADORE, ECOTOX aquatic tests and NRC materials.",
        license="CompTox/DSSTox: US EPA public data; PubChem: public domain (NCBI policy); "
        "ADORE seeds CC BY 4.0",
        stats=summary | {"request_rate_s_per_host": a.min_interval, "workers": a.workers},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
