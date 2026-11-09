# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build tidy NEON aquatic datasets from ``raw('neon')`` (see scripts/data/download_neon.py).

Outputs (``processed('neon_<name>')``), all zstd Parquet, every row tagged with ``release``
(``RELEASE-2026`` or ``PROVISIONAL``) and ``provisional`` (bool):

* ``neon_sites``            site table (domain, aquatic type, lat/lon, tz, HUC2-12, COMID, coverage)
* ``neon_microbe_16s``      16S samples (+ primer set), counts (long), OTU dictionary (sequence +
                            taxonomy) for DP1.20141.002 (RELEASE-2026 OTUs) and the superseded
                            DP1.20141.001 ASV tables (2014-2018; separate ``*_v1`` files)
* ``neon_microbe_seqmeta``  DP1.20282.001 field / extraction / PCR / sequencing / raw-file tables
* ``neon_metagenome_meta``  DP1.20281.001 metadata + raw-file identifiers (sequence data via NMDC)
* ``neon_chem``             DP1.20093.001 grab chemistry: long (censoring + detection limits),
                            event-wide, field (in-situ DO/SC/T at sampling), domain lab (ANC, pH)
* ``neon_sonde_hourly`` / ``neon_sonde_daily``  DP1.20288.001 (QF-masked; per site & HOR)
* ``neon_nitrate``          DP1.20033.001 15-min (masked) + hourly + daily, µmol/L and mg N/L
* ``neon_tsw``, ``neon_tsd``, ``neon_eos``, ``neon_discharge``  30-min / 1-min -> hourly + daily
* ``neon_macroinv``, ``neon_periphyton``, ``neon_fish``, ``neon_zooplankton``, ``neon_microbe_qpcr``
                            stacked observational tables (+ macroinvertebrate densities)

Columns that identify NEON staff (``*By``, e-mail addresses) are dropped from processed tables.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.sources import neon

RAW = paths.RAW / "neon"
PII_RE = re.compile(r"(By|ByEmail|Email)$")


# --------------------------------------------------------------------------- generic helpers
def ledger_releases(code: str, root: Path = RAW) -> dict[tuple[str, str], str]:
    """(site, month) -> release label actually returned by the API."""
    p = root / "_ledger" / f"{code}.jsonl"
    out: dict[tuple[str, str], str] = {}
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "sitemonth" and e.get("status") == "done":
            out[(e["site"], e["month"])] = e.get("release_returned") or e["release"]
    return out


def latest_variables(code: str, root: Path = RAW) -> pd.DataFrame:
    files = sorted((root / code / "_shared").glob(f"*{code}.variables.*.csv"))
    if not files:
        return pd.DataFrame(columns=["table", "fieldName", "dataType", "units"])
    v = pd.concat([neon.read_variables(f) for f in files], ignore_index=True)
    return v.drop_duplicates(["table", "fieldName"], keep="last")


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if PII_RE.search(c)]
    return df.drop(columns=cols)


def table_files(code: str, table: str, root: Path = RAW) -> list[Path]:
    out = []
    for f in sorted((root / code).glob("*/*/*.csv")):
        if f.parent.name.startswith("_"):
            continue
        info = neon.parse_file_name(f.name)
        if info.get("role") == "data" and info.get("table") == table:
            out.append(f)
    return out


def tables_present(code: str, root: Path = RAW) -> list[str]:
    seen = set()
    for f in (root / code).glob("*/*/*.csv"):
        if f.parent.name.startswith("_"):
            continue
        info = neon.parse_file_name(f.name)
        if info.get("role") == "data":
            seen.add(info["table"])
    return sorted(seen)


def stack_table(code: str, table: str, root: Path = RAW,
                variables: pd.DataFrame | None = None) -> pd.DataFrame:
    """All site-month CSVs of one table -> one typed frame with release/provisional."""
    files = table_files(code, table, root)
    if not files:
        return pd.DataFrame()
    variables = latest_variables(code, root) if variables is None else variables
    rel = ledger_releases(code, root)
    parts = []
    for f in files:
        d = neon.read_table(f, variables, table)
        site, month = f.parent.parent.name, f.parent.name
        d["release"] = rel.get((site, month), "unknown")
        d["source_month"] = month
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["release"] = df["release"].astype("string")
    df["provisional"] = df["release"].eq(neon.PROVISIONAL).astype(bool)
    if "uid" in df.columns:
        # a record re-published in several months / both labels: keep the pinned release
        df["_p"] = df["provisional"].astype(int)
        df = df.sort_values(["_p"]).drop_duplicates("uid", keep="first").drop(columns="_p")
    return drop_pii(df).reset_index(drop=True)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    paths.ensure(path.parent)
    tmp = path.with_suffix(".parquet.part")
    df.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(path)
    return path


def stack_product(short: str, out_name: str, root: Path = RAW) -> dict[str, Path]:
    """Every data table of a product -> ``processed(out_name)/<table>.parquet``."""
    spec = neon.PRODUCTS[short]
    out = paths.processed(out_name)
    variables = latest_variables(spec.code, root)
    written = {}
    for t in tables_present(spec.code, root):
        df = stack_table(spec.code, t, root, variables)
        if len(df):
            written[t] = write_parquet(df, out / f"{t}.parquet")
    return written


def date_col(t: pd.Series) -> pd.Series:
    return pd.to_datetime(t, utc=True).dt.tz_convert(None).dt.normalize()


# --------------------------------------------------------------------------- sites
def build_sites(root: Path = RAW) -> pd.DataFrame:
    """Site table for leave-one-domain-out splits and joins."""
    rows = []
    meta = root / "_meta" / "sites"
    reg = None
    reg_p = paths.PROCESSED / "sites" / "sites.parquet"
    if reg_p.exists():
        reg = pd.read_parquet(reg_p)
        reg = reg[reg["network"] == "NEON"].set_index("site_id")
    for s in neon.AQUATIC_SITES:
        site = json.loads((meta / f"{s}_site.json").read_text())["data"]
        loc = json.loads((meta / f"{s}_location.json").read_text())["data"]
        props = {p["locationPropertyName"].replace("Value for ", ""): p["locationPropertyValue"]
                 for p in loc.get("locationProperties") or []}
        habitat = props.get("HABITAT")
        r = {
            "site": s, "site_name": site.get("siteName"), "domain": site.get("domainCode"),
            "domain_name": site.get("domainName"), "state": site.get("stateCode"),
            "neon_site_type": site.get("siteType"), "aquatic_type": habitat,
            "aquatic_class": ("lake" if habitat and "Lake" in habitat else
                              "river" if habitat and "Non-wadeable" in habitat else
                              "stream" if habitat and "Wadeable" in habitat else None),
            "site_lat": site.get("siteLatitude"), "site_lon": site.get("siteLongitude"),
            "elev_m": loc.get("locationElevation"), "timezone": props.get("Site Timezone"),
        }  # NEON's "ZoneTime" property reads "MST" for every site, so it is not used
        if reg is not None and s in reg.index:
            g = reg.loc[s]
            for c in ("lat", "lon", "coord_source", "huc2", "huc4", "huc8", "huc12", "huc_source",
                      "utc_offset_h", "comid", "comid_method", "in_conus"):
                if c in g.index:
                    r[{"lat": "station_lat", "lon": "station_lon"}.get(c, c)] = g[c]
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("huc2", "huc4", "huc8", "huc12", "comid"):
        if c in df.columns:
            df[c] = df[c].astype("string")
    # coverage (site-months per product and release) from cached /products JSON
    for short in neon.PRIORITY_ORDER:
        code = neon.PRODUCTS[short].code
        p = root / "_meta" / "products" / f"{code}.json"
        if not p.exists():
            continue
        sm = neon.site_months(json.loads(p.read_text()), neon.AQUATIC_SITES)
        c = sm.groupby("site").agg(n=("month", "size"), first=("month", "min"),
                                   last=("month", "max"))
        df[f"n_months_{short}"] = df["site"].map(c["n"]).fillna(0).astype(int)
        df[f"first_{short}"] = df["site"].map(c["first"]).astype("string")
        df[f"last_{short}"] = df["site"].map(c["last"]).astype("string")
    aop = root / "_meta" / "aop_availability.parquet"
    if aop.exists():
        a = pd.read_parquet(aop)
        df["n_months_aop_any"] = df["site"].map(a.groupby("site")["month"].nunique()).fillna(0
                                                                                    ).astype(int)
    return df


def site_offsets(root: Path = RAW) -> dict[str, float]:
    p = paths.PROCESSED / "neon_sites" / "sites.parquet"
    df = pd.read_parquet(p) if p.exists() else build_sites(root)
    off = {}
    for r in df.itertuples(index=False):
        v = getattr(r, "utc_offset_h", None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            # standard-time offset of the IANA zone (January = no DST in the US)
            v = pd.Timestamp("2021-01-15", tz=str(r.timezone)).utcoffset().total_seconds() / 3600
        off[r.site] = float(v)
    return off


# --------------------------------------------------------------------------- microbes
def _first(df: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if df.empty or key not in df.columns:
        return pd.DataFrame(columns=[key] + cols)
    d = df.sort_values("provisional").drop_duplicates(key)
    return d[[key] + [c for c in cols if c != key]]


def build_microbe_seqmeta(root: Path = RAW) -> dict[str, pd.DataFrame]:
    """Stack DP1.20282.001 tables (all of them) and return them by table name."""
    code = neon.PRODUCTS["microbe_seq"].code
    v = latest_variables(code, root)
    return {t: stack_table(code, t, root, v) for t in tables_present(code, root)}


def primer_table(seqmeta: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pcr = seqmeta.get("mmg_swPcrAmplification_16S", pd.DataFrame())
    if pcr.empty:
        return pd.DataFrame()
    p = pcr.copy()
    p["primer_set"] = [neon.primer_set(f, r, s) for f, r, s in
                       zip(p["forwardPrimer"].fillna(""), p["reversePrimer"].fillna(""),
                           p.get("targetSubfragment", pd.Series("", index=p.index)).fillna(""))]
    p["primer_record_issue"] = [neon.primer_record_issue(f, r, s) for f, r, s in
                                zip(p["forwardPrimer"].fillna(""), p["reversePrimer"].fillna(""),
                                    p.get("targetSubfragment", pd.Series("", index=p.index)
                                          ).fillna(""))]
    p["primer_record_issue"] = p["primer_record_issue"].astype("string")
    keep = ["dnaSampleID", "siteID", "namedLocation", "collectDate", "forwardPrimer",
            "reversePrimer", "targetSubfragment", "testProtocolVersion", "primer_set",
            "primer_record_issue",
            "laboratoryName", "processedDate", "qaqcStatus", "release", "provisional"]
    return p[[c for c in keep if c in p.columns]].reset_index(drop=True)


def _taxon_files(code: str, root: Path) -> list[Path]:
    return sorted((root / code / "_external").glob("*.parquet"))


def build_microbe_16s(root: Path = RAW) -> dict[str, pd.DataFrame]:
    """16S samples / counts / OTUs for DP1.20141.002 (+ v1 DP1.20141.001)."""
    seqmeta = build_microbe_seqmeta(root)
    primers = primer_table(seqmeta)
    seq16 = seqmeta.get("mmg_swMarkerGeneSequencing_16S", pd.DataFrame())
    gen = seqmeta.get("amc_fieldGenetic", pd.DataFrame())
    sup = seqmeta.get("amc_fieldSuperParent", pd.DataFrame())
    out: dict[str, pd.DataFrame] = {"primers": primers}
    for short, meta_table, suffix in (("microbe_tax", "mct_surfaceWaterSampleMetadata_16S", ""),
                                      ("microbe_comp_v1", "mcc_swSeqVariantMetadata_16S", "_v1")):
        code = neon.PRODUCTS[short].code
        meta = stack_table(code, meta_table, root)
        if meta.empty:
            continue
        s = meta.rename(columns={"siteID": "site", "domainID": "domain",
                                 "namedLocation": "named_location", "collectDate": "time_utc"})
        s["date"] = date_col(s["time_utc"])
        s["product"] = code
        s = s.merge(_first(primers, "dnaSampleID", ["forwardPrimer", "reversePrimer",
                                                    "targetSubfragment", "primer_set",
                                                    "primer_record_issue",
                                                    "testProtocolVersion"]),
                    on="dnaSampleID", how="left")
        s["primer_set"] = s["primer_set"].fillna("unknown")
        s = s.merge(_first(seq16, "dnaSampleID", ["instrument_model", "sequencingMethod",
                                                  "sampleTotalReadNumber", "ncbiProjectID",
                                                  "sequencingFacilityID"]),
                    on="dnaSampleID", how="left", suffixes=("", "_seq"))
        # genetic sample -> field parent (parentSampleID, eventID links to chemistry)
        if not gen.empty:
            g = _first(gen, "geneticSampleID", ["parentSampleID"])
            s = s.merge(g, on="geneticSampleID", how="left")
        if "parentSampleID" not in s.columns:
            s["parentSampleID"] = pd.NA
        # v1 samples carry no geneticSampleID link sometimes: derive from dnaSampleID
        miss = s["parentSampleID"].isna()
        if miss.any():
            s.loc[miss, "parentSampleID"] = s.loc[miss, "dnaSampleID"].str.replace(
                r"\.DNA-DNA\d+$", "", regex=True)
        if not sup.empty:
            sp = _first(sup, "parentSampleID", ["eventID", "aquaticSiteType", "samplerType",
                                               "dissolvedOxygen", "specificConductance",
                                               "waterTemp", "decimalLatitude",
                                               "decimalLongitude"])
            sp = sp.rename(columns={"dissolvedOxygen": "field_do_mgl",
                                    "specificConductance": "field_spcond_uscm",
                                    "waterTemp": "field_temp_c"})
            s = s.merge(sp, on="parentSampleID", how="left")
        # taxon tables
        counts, otus, seen = [], [], set()
        for f in _taxon_files(code, root):
            if "16S" not in f.name:
                continue
            d = pd.read_parquet(f)
            if "sequenceName" not in d.columns or "individualCount" not in d.columns:
                continue  # e.g. the reference sequence table
            c, o = neon.tidy_taxon_table(d, source_file=f.stem)
            counts.append(c)
            o = o[~o["sequenceName"].isin(seen)]
            seen.update(o["sequenceName"].dropna().tolist())
            otus.append(o)
        if counts:
            cnt = pd.concat(counts, ignore_index=True)
            cnt = cnt.drop_duplicates(["dnaSampleID", "sequenceName", "sequencerRunID"])
            otu = pd.concat(otus, ignore_index=True).drop_duplicates("sequenceName")
            # taxon tables sometimes spell IDs in lower case ('WLOU.ss.2022...' vs metadata
            # 'WLOU.SS.2022...'): canonicalise case-insensitively to the metadata spelling
            canon = {str(k).upper(): k for k in s["dnaSampleID"].dropna()}
            cnt["dnaSampleID_raw"] = cnt["dnaSampleID"]
            up = cnt["dnaSampleID"].str.upper()
            cnt["dnaSampleID"] = up.map(canon).fillna(cnt["dnaSampleID"]).astype("string")
            cnt["in_metadata"] = up.isin(set(canon))
            tot = cnt.groupby("dnaSampleID")["count"].agg(["sum", "size"])
            s["reads_in_table"] = s["dnaSampleID"].map(tot["sum"]).astype("Int64")
            s["n_taxa"] = s["dnaSampleID"].map(tot["size"]).astype("Int64")
            out[f"counts{suffix}"] = cnt
            out[f"otus{suffix}"] = otu
        s["otu_scheme"] = ("RELEASE-2026 OTUs (VSEARCH 97%, RDP 2.14 trainset 19); IDs valid only "
                           "within this release" if suffix == "" else
                           "DP1.20141.001 sequence variants (md5 names), 2014-2018")
        out[f"samples{suffix}"] = s.reset_index(drop=True)
    return out


def primer_release_summary(samples: pd.DataFrame, primers: pd.DataFrame) -> pd.DataFrame:
    """Primer set x product: samples, sites, date range — for cross-primer test design."""
    rows = []
    if not primers.empty:
        p = primers.copy()
        p["date"] = date_col(p["collectDate"])
        for ps, g in p.groupby("primer_set"):
            rows.append({"scope": "DP1.20282.001 PCR (all sequenced)", "primer_set": ps,
                         "n": len(g), "sites": g["siteID"].nunique(),
                         "first": g["date"].min(), "last": g["date"].max(),
                         "protocols": "; ".join(sorted(g["testProtocolVersion"].dropna().unique()))})
    for key, s in samples.items():
        for (ps, prod), g in s.groupby(["primer_set", "product"]):
            rows.append({"scope": f"{prod} {key}", "primer_set": ps, "n": len(g),
                         "sites": g["site"].nunique(), "first": g["date"].min(),
                         "last": g["date"].max(),
                         "protocols": "; ".join(sorted(g["testProtocolVersion"].dropna().unique()))
                         if "testProtocolVersion" in g else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- chemistry
def slug(s: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(s).lower())).strip("_")


UNIT_SHORT = {"milligramsPerLiter": "mgl", "microgramsPerLiter": "ugl",
              "milliequivalentsPerLiter": "meql", "microequivalentsPerLiter": "ueql",
              "absorbance units": "au",
              "microsiemensPerCentimeter": "uscm", "microgram": "ug", "milligram": "mg",
              "nephelometricTurbidityUnit": "ntu", "pH": "", "percent": "pct",
              "absorbanceUnitsPerCentimeter": "aucm", "permil": "permil"}


def chem_var(analyte: str, unit) -> str:
    u = UNIT_SHORT.get(str(unit), slug(unit)) if unit is not None and not pd.isna(unit) else ""
    return slug(analyte) + (f"_{u}" if u else "")


def build_chem(root: Path = RAW) -> dict[str, pd.DataFrame]:
    code = neon.PRODUCTS["chem"].code
    v = latest_variables(code, root)
    ext = stack_table(code, "swc_externalLabDataByAnalyte", root, v)
    labs = [neon.read_table(f, v, "swc_externalLabSummaryData")
            for f in sorted((root / code / "_shared").glob("*swc_externalLabSummaryData*.csv"))]
    lab = pd.concat(labs, ignore_index=True) if labs else pd.DataFrame()
    long = neon.tidy_chem(ext, lab)
    long["release"] = ext["release"].to_numpy()
    long["provisional"] = ext["provisional"].to_numpy()
    long = neon.fill_missing_units(long)
    long["var"] = [chem_var(a, u) for a, u in zip(long["analyte"], long["unit"])]
    long["parentSampleID"] = long["sample_id"].str.replace(
        r"\.(FIL|RAW|PCN|POM(?:\.\d+)?|CN|ANC)$", "", regex=True)
    field = stack_table(code, "swc_fieldSuperParent", root, v)
    fd = stack_table(code, "swc_fieldData", root, v)
    dom = stack_table(code, "swc_domainLabData", root, v)
    # event-wide (one row per parentSampleID): value + censor flag per var
    w = long.dropna(subset=["parentSampleID"])
    w = w[~w["sample_id"].str.contains(r"\.POM", na=False)]  # POM replicates kept in long only
    agg = w.groupby(["parentSampleID", "var"]).agg(value=("value", "mean"),
                                                   censored=("censored", "max"))
    wide = agg["value"].unstack("var")
    cen = agg["censored"].unstack("var").astype("boolean")
    cen.columns = [f"{c}__censored" for c in cen.columns]
    wide = wide.join(cen).reset_index()
    ev = long.groupby("parentSampleID").agg(site=("site", "first"), domain=("domain", "first"),
                                            named_location=("named_location", "first"),
                                            time_utc=("time_utc", "min"),
                                            release=("release", "first"),
                                            provisional=("provisional", "max"))
    wide = ev.reset_index().merge(wide, on="parentSampleID", how="right")
    wide["date"] = date_col(wide["time_utc"])
    if not field.empty:
        f = field.rename(columns={"dissolvedOxygen": "field_do_mgl",
                                  "dissolvedOxygenSaturation": "field_do_sat_pct",
                                  "specificConductance": "field_spcond_uscm",
                                  "waterTemp": "field_temp_c"})
        wide = wide.merge(_first(f, "parentSampleID",
                                 ["eventID", "aquaticSiteType", "samplerType", "field_do_mgl",
                                  "field_do_sat_pct", "field_spcond_uscm", "field_temp_c"]),
                          on="parentSampleID", how="left")
    if not dom.empty:
        d = dom.rename(columns={"initialSamplepH": "lab_ph", "ancMeqPerL": "lab_anc_meql",
                                "alkMeqPerL": "lab_alk_meql"})
        d = d.sort_values("provisional").drop_duplicates("parentSampleID")
        wide = wide.merge(d[[c for c in ["parentSampleID", "lab_ph", "lab_anc_meql",
                                         "lab_alk_meql"] if c in d.columns]],
                          on="parentSampleID", how="left")
    units = (long.groupby("var")["unit"].first().rename("unit").reset_index()
             .merge(long.groupby("var")["analyte"].first().reset_index(), on="var"))
    return {"chem_long": long, "chem_event_wide": wide, "field": field, "field_samples": fd,
            "domain_lab": dom, "lab_summary": drop_pii(lab), "units": units}


# --------------------------------------------------------------------------- sensors
SENSOR_PRODUCTS = {
    # short: (vars, table regex, needs ver column)
    "sonde": (neon.SONDE_VARS, r"waq_instantaneous"),
    "nitrate": (neon.NITRATE_VARS, r"NSW_15_minute"),
    "tsw": (neon.TSW_VARS, r"TSW_30min"),
    "tsd": (neon.TSD_VARS, r"TSD_30_min"),
    "eos": (neon.EOS_VARS, r"EOS_30_min"),
    "discharge": (neon.DISCHARGE_VARS, r"csd_continuousDischarge"),
}
UMOL_NO3_TO_MG_N = 14.0067 / 1000.0  # µmol/L NO3 -> mg N/L (molar mass of N)

# Same plausible ranges / zero-clipping as the USGS daily tier (sentinel/data/sonde_daily.py),
# keyed by the equivalent USGS parameter code. Nitrate (µmol/L), depth, stage, WSE, discharge:
# no range screen (units differ from the USGS codes or values are site-specific).
USGS_EQUIV = {"spcond_uscm": "00095", "do_mgl": "00300", "do_sat_sealevel_pct": "00301",
              "do_sat_local_pct": "00301", "ph": "00400", "turb_fnu": "63680",
              "chl_ugl": "32316", "chl_rfu": "32315", "fdom_qsu": "32295", "temp_c": "00010"}


def plausibility_screen(df: pd.DataFrame, found: Iterable[neon.SensorVar]
                        ) -> tuple[pd.DataFrame, dict]:
    """Mask values outside the shared plausible range; clip small negatives to 0 where the
    daily tier does (turbidity, chlorophyll). Returns (frame, {var: {implausible, clipped}})."""
    from .sonde_daily import CLIP_ZERO, PLAUSIBLE
    df = df.copy()
    counts = {}
    for v in found:
        code = USGS_EQUIV.get(v.out)
        if code is None or v.value not in df.columns:
            continue
        lo, hi = PLAUSIBLE[code]
        x = pd.to_numeric(df[v.value], errors="coerce").astype("float64")
        bad = (x < lo) | (x > hi)
        clip = (code in CLIP_ZERO) & (x < 0) & ~bad
        x[bad] = np.nan
        x[clip] = 0.0
        df[v.value] = x
        counts[v.out] = {"implausible": int(bad.sum()), "clipped_to_zero": int(clip.sum())}
    return df, counts


def sensor_files(code: str, table_re: str, site: str, root: Path = RAW) -> list[tuple[Path, dict]]:
    out = []
    for f in sorted((root / code / site).glob("*/*.parquet")):
        info = neon.parse_file_name(f.name[:-8] + ".csv")
        if info.get("role") == "data" and re.fullmatch(table_re, info.get("table") or ""):
            out.append((f, info))
    return out


def build_sensor_site(short: str, site: str, utc_offset_h: float, root: Path = RAW,
                      keep_native: bool = False) -> dict[str, pd.DataFrame | dict]:
    """One site of one sensor product -> hourly + daily frames (columns hor, ver)."""
    spec = neon.PRODUCTS[short]
    variables, table_re = SENSOR_PRODUCTS[short]
    rel = ledger_releases(spec.code, root)
    hourly_parts, native_parts = [], []
    qc: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "masked_qf": 0})
    for f, info in sensor_files(spec.code, table_re, site, root):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        month = f.parent.name
        tcol = "startDateTime" if "startDateTime" in df.columns else "endDate"
        found = neon.resolve_vars(df.columns, variables)
        if not found:
            print(f"WARNING {short} {site} {f.name}: none of {[v.value for v in variables]} "
                  f"in columns {list(df.columns)[:12]}", flush=True)
            continue
        df, counts = neon.mask_qf(df, found)
        for k, c in counts.items():
            qc[k]["n"] += c["n"]
            qc[k]["masked_qf"] += c["masked_qf"]
        df, pcounts = plausibility_screen(df, found)
        for k, c in pcounts.items():
            qc[k]["masked_implausible"] = qc[k].get("masked_implausible", 0) + c["implausible"]
            qc[k]["clipped_to_zero"] = qc[k].get("clipped_to_zero", 0) + c["clipped_to_zero"]
        hor = info.get("hor") or (str(df["stationHorizontalID"].dropna().iloc[0])
                                  if "stationHorizontalID" in df.columns and
                                  df["stationHorizontalID"].notna().any() else "000")
        ver = info.get("ver") or "100"
        h = neon.aggregate_hourly(df, found, time_col=tcol)
        if h.empty:
            continue
        h = h.reset_index()
        h.insert(1, "hor", hor)
        h.insert(2, "ver", ver)
        h["release"] = rel.get((site, month), "unknown")
        hourly_parts.append(h)
        if keep_native:
            nat = pd.DataFrame({"time_utc": pd.to_datetime(df[tcol], utc=True), "hor": hor,
                                "ver": ver})
            for v in found:
                nat[v.out] = df[v.value].astype("float32").to_numpy()
            nat["release"] = rel.get((site, month), "unknown")
            native_parts.append(nat.dropna(subset=[v.out for v in found], how="all"))
    if not hourly_parts:
        return {"hourly": pd.DataFrame(), "daily": pd.DataFrame(), "qc": dict(qc)}
    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = hourly.sort_values(["hor", "ver", "time_utc"]).drop_duplicates(
        ["hor", "ver", "time_utc"], keep="first")
    if short == "nitrate":
        for s in ("mean", "min", "max"):
            hourly[f"no3n_mgl_{s}"] = (hourly[f"no3_umoll_{s}"].astype("float64")
                                       * UMOL_NO3_TO_MG_N).astype("float32")
        hourly["no3n_mgl_n"] = hourly["no3_umoll_n"]
    hourly["release"] = hourly["release"].astype("string")
    hourly["provisional"] = hourly["release"].eq(neon.PROVISIONAL)
    hourly.insert(0, "site", site)
    daily_parts = []
    for (hor, ver), g in hourly.groupby(["hor", "ver"]):
        stat_cols = [c for c in g.columns if re.search(r"_(mean|min|max|n)$", c)]
        d = neon.hourly_to_daily(g.set_index("time_utc")[stat_cols], utc_offset_h)
        if d.empty:
            continue
        prov = g.assign(date=(g["time_utc"].dt.tz_convert(None)
                              + pd.Timedelta(hours=utc_offset_h)).dt.normalize()
                        ).groupby("date")["provisional"].max()
        d = d.reset_index()
        d.insert(0, "site", site)
        d.insert(2, "hor", hor)
        d.insert(3, "ver", ver)
        d["provisional"] = d["date"].map(prov).fillna(False).astype(bool)
        daily_parts.append(d)
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    out = {"hourly": hourly.reset_index(drop=True), "daily": daily, "qc": dict(qc)}
    if keep_native and native_parts:
        nat = pd.concat(native_parts, ignore_index=True)
        nat["release"] = nat["release"].astype("string")
        nat["provisional"] = nat["release"].eq(neon.PROVISIONAL)
        nat.insert(0, "site", site)
        out["native"] = nat
    return out


def sensor_positions(code: str, root: Path = RAW) -> pd.DataFrame:
    parts = []
    for site_dir in sorted((root / code).glob("*/_site")):
        files = sorted(site_dir.glob("*sensor_positions*.csv"))
        if not files:
            continue
        d = pd.read_csv(files[-1], dtype=str, keep_default_na=False, na_values=[""])
        d.insert(0, "site", site_dir.parent.name)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sensor_units(short: str) -> dict[str, str]:
    variables, _ = SENSOR_PRODUCTS[short]
    u = {v.out: v.unit for v in variables}
    if short == "nitrate":
        u["no3n_mgl"] = "milligramsPerLiter as N (derived: µmol/L x 0.0140067)"
    return u


# --------------------------------------------------------------------------- macroinvertebrates
def macroinv_density(tax: pd.DataFrame, persample: pd.DataFrame, field: pd.DataFrame
                     ) -> pd.DataFrame:
    """inv_taxonomyProcessed -> per-sample taxon density (individuals / m2)."""
    if tax.empty:
        return pd.DataFrame()
    t = tax.copy()
    keys = [c for c in ("sampleID",) if c in t.columns]
    if field is not None and not field.empty and "benthicArea" in field.columns:
        fa = _first(field, "sampleID", ["benthicArea", "habitatType", "samplerType",
                                        "substratumSizeClass", "eventID"])
        t = t.merge(fa, on="sampleID", how="left", suffixes=("", "_field"))
    if "estimatedTotalCount" in t.columns and "benthicArea" in t.columns:
        t["density_per_m2"] = (pd.to_numeric(t["estimatedTotalCount"], errors="coerce")
                               / pd.to_numeric(t["benthicArea"], errors="coerce"))
    return t


# --------------------------------------------------------------------------- links
def link_microbe_chem(samples: pd.DataFrame, chem_wide: pd.DataFrame, tol_days: int = 1
                      ) -> pd.DataFrame:
    """16S sample -> chemistry event (docs/research/03 d.1 join rule).

    1. ``parentSampleID`` identical (same field event) -> ``link='parentSampleID'``.
    2. else same ``named_location`` with |date difference| <= ``tol_days`` (nearest) ->
       ``link='location_date'``. Samples from a different named location are never linked.
    """
    s = samples[["dnaSampleID", "product", "site", "named_location", "date",
                 "parentSampleID"]].copy()
    c = chem_wide[["parentSampleID", "named_location", "date"]].dropna(subset=["parentSampleID"])
    c = c.rename(columns={"parentSampleID": "chem_parentSampleID", "date": "chem_date"})
    exact = s.merge(c[["chem_parentSampleID", "chem_date"]], left_on="parentSampleID",
                    right_on="chem_parentSampleID", how="inner")
    exact["link"] = "parentSampleID"
    rest = s[~s["dnaSampleID"].isin(exact["dnaSampleID"])]
    near = rest.merge(c, on="named_location", how="inner")
    near["dt"] = (near["chem_date"] - near["date"]).dt.days
    near = near[near["dt"].abs() <= tol_days]
    near = near.assign(adt=near["dt"].abs()).sort_values(["dnaSampleID", "adt"]
                                                         ).drop_duplicates("dnaSampleID")
    near["link"] = "location_date"
    out = pd.concat([exact, near], ignore_index=True)
    # .001 samples sequenced in two runs appear twice in the sample table
    out = out.drop_duplicates(["dnaSampleID", "product", "chem_parentSampleID"])
    out["dt_days"] = (out["chem_date"] - out["date"]).dt.days.astype("Int64")
    return out[["dnaSampleID", "product", "site", "named_location", "date", "parentSampleID",
                "chem_parentSampleID", "chem_date", "dt_days", "link"]].reset_index(drop=True)
