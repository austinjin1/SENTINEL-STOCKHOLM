# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Builder for processed('nars_streams') and processed('nars_lakes').

Tables (Parquet, zstd), all keyed by ``visit_id`` (survey-scoped site-visit key):

    visits               one row per site visit (location, ecoregions, HUC8 string, COMID, flags)
    benthic_counts       long: visit x taxon (count, 300-count, distinctness, source taxonomy)
    fish_counts          long (streams)
    zooplankton_counts   long (lakes)
    phytoplankton_counts long (lakes; NLA 2007 soft-algae and diatom analyses kept separate)
    taxa_source          every (survey, assemblage, taxa_id, name) with published hierarchy/traits
    chemistry_long       one row per (visit, analyte) with source + canonical unit, censoring, flags
    chemistry_wide       one row per visit, canonical units in the column names
    periphyton           long periphyton biomass / chlorophyll (streams; no taxonomic counts exist)
    condition_long       condition classes and index scores per (visit, indicator, source)
    condition_wide       one row per visit, preferred source per indicator
    phab_long            physical-habitat metrics (all numeric metrics of the cycle's metric file)
    profiles             lake depth profiles (temperature, DO, pH, conductivity)

Nothing is imputed. Non-detects keep value NaN with ``censored`` and ``detection_limit``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .bio_common import write_parquet
from .sources import nars as N

log = logging.getLogger(__name__)

# Which file feeds which parser, per survey (see N.FILE_ROLES for the file names).
CHEM_PLAN = {
    "nrsa0809": [("plain", "chem", {"columns": N.NRSA0809_CHEM_COLUMNS, "flag_suffix": "_ALERT"}),
                 ("field", "field_chem", {})],
    "nrsa1314": [("suffix", "chem", {}), ("suffix", "algal_toxin", {"analytes": ["MICX"]}),
                 ("suffix", "water_chla", {"analytes": ["CHLA"]}), ("field", "field_chem", {})],
    "nrsa1819": [("suffix", "chem", {}), ("long", "algal_toxin", {}), ("field", "field_chem", {})],
    "nrsa2324": [("suffix", "chem", {}), ("long", "algal_toxin", {})],
    "nla07": [("plain", "chem", {"columns": N.NLA07_WQ_COLUMNS}), ("nla07_micx", "algal_toxin", {}),
              ("secchi", "secchi", {})],
    "nla12": [("suffix", "chem", {}), ("suffix", "chla", {}), ("suffix", "algal_toxin", {}),
              ("suffix", "atrazine", {}), ("secchi", "secchi", {})],
    "nla17": [("long", "chem", {}), ("long", "algal_toxin", {}), ("long", "atrazine", {}), ("secchi", "secchi", {})],
    "nla22": [("suffix", "chem", {}), ("long", "algal_toxin", {}), ("long", "atrazine", {}), ("secchi", "secchi", {})],
}
PERIPHYTON_PLAN = {
    "nrsa1314": [("suffix", "nrsa1314_widepbio_04232019.csv", {}), ("suffix", "nrsa1314_widepchl_04232019.csv", {})],
    "nrsa1819": [("long", "NRSA_1819_pbio_0.csv", {}), ("long", "NRSA_1819_PeriChla.csv", {"fixed_analyte": "PERI_CHLA"})],
    "nrsa2324": [("long", "nrsa2324_periphytonbiomass.csv", {}),
                 ("long", "nrsa2324_periphytonchlorophyll-a.csv", {"fixed_analyte": "PERI_CHLA"})],
}
CONDITION_FILES = {
    "nrsa0809": ["bentcond.csv", "fishcond.csv", "chemcond.csv", "phabcond_0.csv", "enterocond.csv"],
    "nrsa1314": ["nrsa1314_allcond_05312019_0.csv", "nrsa1314_bentmmi_04232019.csv", "nrsa1314_fishmmi_04232019.csv"],
    "nrsa1819": ["nrsa1819_data_for_populationestimates.csv", "nrsa-1819-benthic-macroinvertebrate-mmi-data.csv",
                 "nrsa-1819-fish-mmi-data.csv"],
    "nrsa2324": ["nrsa2324_benthicmmi.csv", "nrsa2324_fishmmi.csv"],
    "nla07": ["nla2007_bentcond_08232016.csv", "nla2007_chemical_conditionestimates_20091123.csv",
              "nla2007_trophic_conditionestimate_20091123.csv", "nla2007_plankton_oemodel_conditionestimates_20091125.csv",
              "nla2007_meando_conditionestimate_20091123.csv", "nla2007_recreational_conditionestimates_20091123.csv",
              "nla2007_phab_condtionestimates_20091130.csv"],
    "nla12": ["nla_2012_condition_categories.csv", "nla2012_bentcond_08232016.csv", "nla2012_zoopcond_08192016.csv"],
    "nla17": ["nla2017_BentMMI.csv", "nla2017_zoopMMI.csv"],
    "nla22": ["nla22_benthic_mmi_0.csv", "nla22_zooplankton_mmi_1.csv"],
}
NLA_POPEST_FILE = ("nla22", "nla2007-2022_data_forpopestimates_indexvisits_probsites_0.csv")
# Preferred source per indicator in condition_wide (first match wins); anything else -> first seen.
PREFERRED_CONDITION_SOURCES = {
    "bent_mmi_cond": ["nrsa-1819-benthic-macroinvertebrate-mmi-data.csv", "nrsa2324_benthicmmi.csv",
                      "nrsa1314_allcond_05312019_0.csv", "bentcond.csv",
                      "nla2007-2022_data_forpopestimates_indexvisits_probsites_0.csv"],
}
PHAB_KEY_METRICS = {
    "streams": ["XFC_NAT", "L_XFC_NAT", "PCT_FN", "PCT_SAFN", "LRBS_USE", "W1_HALL", "XCMGW", "L_XCMGW",
                "XCDENBK", "XCDENMID", "XWIDTH", "XDEPTH_CM", "XSLOPE", "RP100", "PCT_FAST", "PCT_SLOW",
                "PCT_POOL", "V1W_MSQ", "XBKF_W", "XBKF_H", "SINU"],
    "lakes": ["RDIS_IX", "RVEGQC3OE", "LITCVRQC3OE", "LITRIPCVRQC3OE", "AMITOTAL", "BFXHORIZDIST_DD",
              "BFXVERTHEIGHT_DD", "FCFAQUATIC", "FCIALL", "HIIALL", "SSIBEDBLD", "RDIS_INEX"],
}


def _read(survey_dir: Path, name: str | None) -> pd.DataFrame | None:
    if not name:
        return None
    p = survey_dir / name
    if not p.exists():
        log.warning("missing raw file %s", p)
        return None
    return N.read_raw(p)


class UidAligner:
    """Some files of a cycle use a superseded UID numbering (e.g. NRSA 2018-19 fish counts and
    periphyton chlorophyll). Rows whose UID is not in the site-information file are re-keyed to
    the site-information UID through an unambiguous (SITE_ID, collection date) match, falling
    back to an unambiguous (SITE_ID, VISIT_NO) match. Every remap is counted (build stats)."""

    def __init__(self, siteinfo: pd.DataFrame, survey: str):
        self.survey = survey
        self.enabled = survey != "nla07" and "UID" in siteinfo.columns
        self.counts: dict[str, dict[str, int]] = {}
        if not self.enabled:
            return
        si = siteinfo[siteinfo["UID"].astype(str).str.strip() != ""]
        self.uids = set(si["UID"].astype(str).str.strip())
        d = N.parse_date(si["DATE_COL"]).dt.strftime("%Y-%m-%d")
        self.by_date = self._unique_map(si["SITE_ID"].str.strip() + "|" + d.fillna(""), si["UID"])
        self.by_visit = self._unique_map(si["SITE_ID"].str.strip() + "|" + si["VISIT_NO"].str.strip(), si["UID"])

    @staticmethod
    def _unique_map(keys: pd.Series, values: pd.Series) -> dict:
        df = pd.DataFrame({"k": keys.to_numpy(), "v": values.astype(str).str.strip().to_numpy()})
        n = df.groupby("k")["v"].nunique()
        ok = set(n[n == 1].index)
        return dict(df[df["k"].isin(ok)].drop_duplicates("k").to_numpy())

    def __call__(self, raw: pd.DataFrame | None, name: str) -> pd.DataFrame | None:
        if raw is None or not self.enabled or "SITE_ID" not in raw.columns:
            return raw
        uc = N._uid_column(raw)
        if uc is None:
            return raw
        uid = raw[uc].astype(str).str.strip()
        bad = ~uid.isin(self.uids)
        if not bad.any():
            return raw
        raw = raw.copy()
        sid = raw["SITE_ID"].astype(str).str.strip()
        new = pd.Series(None, index=raw.index, dtype=object)
        if "DATE_COL" in raw.columns:
            d = N.parse_date(raw["DATE_COL"]).dt.strftime("%Y-%m-%d").fillna("")
            new = (sid + "|" + d).map(self.by_date)
        if "VISIT_NO" in raw.columns:
            fb = (sid + "|" + raw["VISIT_NO"].astype(str).str.strip()).map(self.by_visit)
            new = new.where(new.notna(), fb)
        fix = bad & new.notna()
        raw.loc[fix, uc] = new[fix]
        self.counts[name] = {"rows_with_unknown_uid": int(bad.sum()), "remapped": int(fix.sum()),
                             "unresolved": int((bad & new.isna()).sum())}
        log.info("%s %s: %s", self.survey, name, self.counts[name])
        return raw


def _names(v) -> list[str]:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _parse_chem(kind, raw, survey, fname, kw):
    if kind == "suffix":
        return N.parse_chem_suffix_wide(raw, survey, source_file=fname, **kw)
    if kind == "plain":
        return N.parse_chem_plain_wide(raw, survey, source_file=fname, **kw)
    if kind == "long":
        return N.parse_chem_long(raw, survey, source_file=fname, **kw)
    if kind == "field":
        return N.parse_field_measures(raw, survey, source_file=fname)
    if kind == "nla07_micx":
        return N.parse_nla07_microcystin(raw, survey, source_file=fname)
    if kind == "secchi":
        return N.parse_secchi(raw, survey, source_file=fname)
    raise ValueError(kind)


def build_survey(raw_root: Path, survey: str) -> dict[str, pd.DataFrame]:
    roles = N.FILE_ROLES[survey]
    sdir = raw_root / survey
    resource = "streams" if survey.startswith("nrsa") else "lakes"
    out: dict[str, pd.DataFrame] = {}

    si_raw = _read(sdir, roles["siteinfo"])
    if "SITESAMP" in si_raw.columns or "SAMPLED" in si_raw.columns:
        sc = "SITESAMP" if "SITESAMP" in si_raw.columns else "SAMPLED"
        si_raw = si_raw[si_raw[sc].astype(str).str.strip().str.upper().isin(["Y", "YES"])]
    align = UidAligner(si_raw, survey)
    visits = N.parse_visits(si_raw, survey)
    dup = visits["visit_id"].duplicated(keep="first")
    if dup.any():
        log.warning("%s: %d duplicated visit rows in site info dropped", survey, int(dup.sum()))
    out["visits"] = visits[~dup].reset_index(drop=True)

    def rd(d, name):  # every data table goes through the UID aligner
        return align(_read(d, name), name) if name else None

    comm = {"benthic": "benthic_count"}
    if resource == "streams":
        comm["fish"] = "fish_count"
    else:
        comm["zooplankton"] = "zoop_count"
        comm["phytoplankton"] = "phyto_count"
    taxa_parts = []
    for assemblage, role in comm.items():
        frames = []
        for fname in _names(roles.get(role)):
            raw = rd(sdir, fname)
            if raw is not None:
                frames.append(N.parse_counts(raw, survey, assemblage, source_file=fname))
        if frames:
            allc = pd.concat(frames, ignore_index=True)
            prim = allc["sample_category"].eq("P")
            out[f"{assemblage}_counts"] = allc[prim].reset_index(drop=True)
            if (~prim).any():
                out[f"{assemblage}_counts_qc_duplicates"] = allc[~prim].reset_index(drop=True)
        trole = {"benthic": "benthic_taxa", "fish": "fish_taxa", "zooplankton": "zoop_taxa",
                 "phytoplankton": "phyto_taxa"}[assemblage]
        for fname in _names(roles.get(trole)):
            raw = rd(sdir, fname)
            if raw is not None:
                t = N.parse_taxa_table(raw, survey, assemblage)
                t["taxa_source"] = fname
                taxa_parts.append(t)
    if taxa_parts:
        out["taxa_tables"] = pd.concat(taxa_parts, ignore_index=True)

    chem = []
    for kind, role, kw in CHEM_PLAN[survey]:
        for fname in _names(roles.get(role)):
            raw = rd(sdir, fname)
            if raw is not None:
                c = _parse_chem(kind, raw, survey, fname, kw)
                if not c.empty:
                    chem.append(c)
    out["chemistry_long"] = pd.concat(chem, ignore_index=True) if chem else pd.DataFrame()

    peri = []
    for kind, fname, kw in PERIPHYTON_PLAN.get(survey, []):
        raw = rd(sdir, fname)
        if raw is not None:
            # periphyton files use MATRIX = PERIPHYTON; bypass the water-only filter.
            raw = raw.assign(MATRIX="")
            c = _parse_chem(kind, raw, survey, fname, kw)
            if not c.empty:
                peri.append(c)
    if peri:
        out["periphyton"] = pd.concat(peri, ignore_index=True)

    cond = []
    for fname in CONDITION_FILES.get(survey, []):
        raw = rd(sdir, fname)
        if raw is not None:
            c = N.parse_condition(raw, survey, source_file=fname)
            if not c.empty:
                cond.append(c)
    out["condition_long"] = pd.concat(cond, ignore_index=True) if cond else pd.DataFrame()

    out["uid_alignment"] = pd.DataFrame([{"survey": survey, "file": k, **v} for k, v in align.counts.items()])
    ph = rd(sdir, roles.get("phab"))
    if ph is not None:
        out["phab_long"] = N.parse_phab_metrics(ph, survey, source_file=roles.get("phab"))
    if resource == "lakes":
        pr = rd(sdir, roles.get("profile"))
        if pr is not None:
            out["profiles"] = N.parse_profile(pr, survey, source_file=roles.get("profile"))
    return out


def _slug(unit: str) -> str:
    return (unit.replace("/", "_").replace(" ", "").replace("uS", "us").replace("ueq", "ueq")
            .lower().replace("degc", "degc"))


def chemistry_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per visit; columns ``<analyte>_<unit>`` hold canonical-unit values (NaN when
    missing, non-detect or unknown unit). ``cond_unit_source`` keeps the conductivity label."""
    d = long[long["unit_status"].isin(["ok", "converted", "assumed_unit"])].copy()
    d["col"] = d["analyte"] + "_" + d["unit"].astype(str).map(_slug)
    d = d.sort_values(["visit_id", "col", "value_canonical"], na_position="last")
    wide = d.pivot_table(index="visit_id", columns="col", values="value_canonical", aggfunc="first")
    n_cens = d[d["censored"]].groupby("visit_id").size().rename("n_censored_analytes")
    cond_lab = d[d["analyte"] == "cond"].groupby("visit_id")["unit_source"].first().rename("cond_unit_source")
    wide = wide.join(n_cens).join(cond_lab)
    wide["n_censored_analytes"] = wide["n_censored_analytes"].fillna(0).astype(int)
    return wide.reset_index()


def condition_wide(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=["visit_id"])
    d = long.copy()
    d["pref"] = 99
    for ind, files in PREFERRED_CONDITION_SOURCES.items():
        for rank, f in enumerate(files):
            d.loc[(d["indicator"] == ind) & (d["source_file"] == f), "pref"] = rank
    d = d.sort_values(["visit_id", "indicator", "pref"])
    cls = d[d["class"].notna()].drop_duplicates(["visit_id", "indicator"])
    sco = d[d["score"].notna()].drop_duplicates(["visit_id", "indicator"])
    w1 = cls.pivot(index="visit_id", columns="indicator", values="class")
    w2 = sco.pivot(index="visit_id", columns="indicator", values="score")
    w2.columns = [c + "_score" if not c.endswith(("mmi", "oe")) else c for c in w2.columns]
    return w1.join(w2, how="outer").reset_index()


def taxa_source_table(parts: dict[str, pd.DataFrame], taxa_tables: pd.DataFrame | None) -> pd.DataFrame:
    """Union of published taxa lists and the taxonomy embedded in count files."""
    frames = []
    for df in parts.values():
        cols = ["survey", "assemblage", "taxa_id", "taxon_name"] + [c for c in df.columns if c.startswith("src_")]
        t = df[cols].drop_duplicates(["survey", "assemblage", "taxa_id", "taxon_name"]).copy()
        t["taxa_source"] = "count_file"
        frames.append(t)
    if taxa_tables is not None and not taxa_tables.empty:
        frames.append(taxa_tables)
    t = pd.concat(frames, ignore_index=True)
    # prefer the published taxa list over count-file columns for the same (survey, taxa_id, name)
    t["is_count_file"] = (t["taxa_source"] == "count_file").astype(int)
    t = t.sort_values("is_count_file").drop_duplicates(["survey", "assemblage", "taxa_id", "taxon_name"])
    return t.drop(columns="is_count_file").reset_index(drop=True)


def build_resource(raw_root: Path, out_dir: Path, resource: str, surveys=None) -> dict:
    surveys = surveys or (N.STREAM_SURVEYS if resource == "streams" else N.LAKE_SURVEYS)
    acc: dict[str, list[pd.DataFrame]] = {}
    for s in surveys:
        log.info("building %s", s)
        for k, v in build_survey(raw_root, s).items():
            if v is not None and not v.empty:
                acc.setdefault(k, []).append(v)
    if resource == "lakes":
        pe_survey, pe_file = NLA_POPEST_FILE
        raw = _read(raw_root / pe_survey, pe_file)
        if raw is not None:
            c = N.parse_condition(raw, None, source_file=pe_file, survey_from_design_cycle=True)
            acc.setdefault("condition_long", []).append(c)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    visits = tables["visits"]
    # Rows of data tables whose visit is not a sampled visit of the survey design (special-study
    # sites, blank SITE_ID rows) are dropped and counted.
    dropped_orphans = {}
    vset = set(visits["visit_id"])
    for k, df in list(tables.items()):
        if k in ("visits", "taxa_tables", "uid_alignment") or "visit_id" not in df.columns:
            continue
        o = ~df["visit_id"].isin(vset)
        if o.any():
            dropped_orphans[k] = {"rows": int(o.sum()), "visits": int(df.loc[o, "visit_id"].nunique())}
            tables[k] = df[~o].reset_index(drop=True)

    # cross-cycle identity: number of cycles each UNIQUE_ID appears in
    uq = visits.dropna(subset=["unique_id"]).groupby("unique_id")["survey"].nunique()
    visits["n_cycles_unique_id"] = visits["unique_id"].map(uq).astype("Int64")
    # COMID missing in a cycle's site info -> take it from another cycle of the same UNIQUE_ID
    # (same probability site; only when that UNIQUE_ID maps to exactly one COMID).
    known = visits.dropna(subset=["unique_id", "comid"]).groupby("unique_id")["comid"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else None).dropna()
    fill = visits["comid"].isna() & visits["unique_id"].isin(known.index)
    visits.loc[fill, "comid"] = visits.loc[fill, "unique_id"].map(known)
    visits.loc[fill, "comid_source"] = "other_cycle_same_unique_id"

    count_tables = {a: tables[f"{a}_counts"] for a in ("benthic", "fish", "zooplankton", "phytoplankton")
                    if f"{a}_counts" in tables}
    taxa = taxa_source_table(count_tables, tables.get("taxa_tables"))
    for a, df in count_tables.items():
        drop = [c for c in df.columns if c.startswith("src_") and c not in ("src_ptv", "src_ffg", "src_habit")]
        count_tables[a] = df.drop(columns=drop)
        n = df.groupby("visit_id").agg(n_taxa=("taxon_name", "nunique"), total_count=("count", "sum"))
        visits[f"has_{a}"] = visits["visit_id"].isin(n.index)
        visits[f"{a}_n_taxa"] = visits["visit_id"].map(n["n_taxa"]).astype("Int64")
        orphan = ~df["visit_id"].isin(visits["visit_id"])
        if orphan.any():
            log.warning("%s: %d %s count rows whose visit is absent from site info (kept)", resource,
                        int(orphan.sum()), a)

    chem = tables.get("chemistry_long", pd.DataFrame())
    wide = chemistry_wide(chem) if not chem.empty else pd.DataFrame(columns=["visit_id"])
    visits["has_chemistry"] = visits["visit_id"].isin(wide["visit_id"])
    cond = tables.get("condition_long", pd.DataFrame())
    cw = condition_wide(cond)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    write_parquet(visits, out_dir / "visits.parquet")
    written["visits"] = len(visits)
    for a, df in count_tables.items():
        write_parquet(df, out_dir / f"{a}_counts.parquet")
        written[f"{a}_counts"] = len(df)
    write_parquet(taxa, out_dir / "taxa_source.parquet")
    written["taxa_source"] = len(taxa)
    for name, df in (("chemistry_long", chem), ("chemistry_wide", wide), ("condition_long", cond),
                     ("condition_wide", cw)):
        write_parquet(df, out_dir / f"{name}.parquet")
        written[name] = len(df)
    for a in ("benthic", "fish", "zooplankton", "phytoplankton"):
        k = f"{a}_counts_qc_duplicates"
        if k in tables:
            write_parquet(tables[k].drop(columns=[c for c in tables[k].columns if c.startswith("src_")
                                                   and c not in ("src_ptv", "src_ffg", "src_habit")]),
                          out_dir / f"{k}.parquet")
            written[k] = len(tables[k])
    for name in ("periphyton", "phab_long", "profiles"):
        if name in tables:
            write_parquet(tables[name], out_dir / f"{name}.parquet")
            written[name] = len(tables[name])
    if "phab_long" in tables:
        key = PHAB_KEY_METRICS[resource]
        ph = tables["phab_long"]
        phw = ph[ph["metric"].isin(key)].pivot_table(index="visit_id", columns="metric", values="value",
                                                     aggfunc="first").reset_index()
        write_parquet(phw, out_dir / "phab_key_wide.parquet")
        written["phab_key_wide"] = len(phw)

    analytes = (chem.groupby(["analyte", "unit"], dropna=False)
                .agg(n=("value", "size"), n_censored=("censored", "sum"),
                     source_codes=("source_code", lambda s: sorted(set(s))),
                     unit_source_labels=("unit_source", lambda s: sorted(set(map(str, s)))),
                     unit_status=("unit_status", lambda s: dict(pd.Series(s).value_counts())))
                .reset_index()) if not chem.empty else pd.DataFrame()
    if not analytes.empty:
        (out_dir / "analyte_dictionary.json").write_text(
            json.dumps(analytes.to_dict(orient="records"), indent=1, default=str))

    if "uid_alignment" in tables:
        tables["uid_alignment"].to_csv(out_dir / "uid_alignment.csv", index=False)
    stats = {
        "uid_alignment": tables["uid_alignment"].to_dict(orient="records") if "uid_alignment" in tables else [],
        "comid_source": visits["comid_source"].value_counts(dropna=False).to_dict(),
        "dropped_orphan_rows": dropped_orphans,
        "rows": written,
        "visits_by_survey": visits.groupby("survey").size().to_dict(),
        "sites_by_survey": visits.groupby("survey")["site_id"].nunique().to_dict(),
        "within_cycle_revisits": visits.groupby("survey")["within_cycle_revisit"].sum().astype(int).to_dict(),
        "visits_with_benthic": visits.groupby("survey")["has_benthic"].sum().astype(int).to_dict(),
        "benthic_taxa_by_survey": count_tables["benthic"].groupby("survey")["taxon_name"].nunique().to_dict(),
        "ag_eco9_values": sorted(visits["ag_eco9"].dropna().unique().tolist()),
        "huc8_non_null": int(visits["huc8"].notna().sum()),
        "comid_non_null": int(visits["comid"].notna().sum()),
        "reachcode_mangled": int(visits["reachcode_mangled"].sum()),
        "chem_unit_status": chem["unit_status"].value_counts().to_dict() if not chem.empty else {},
        "chem_censored_rows": int(chem["censored"].sum()) if not chem.empty else 0,
    }
    for a in ("fish", "zooplankton", "phytoplankton"):
        if a in count_tables:
            stats[f"{a}_taxa_by_survey"] = count_tables[a].groupby("survey")["taxon_name"].nunique().to_dict()
            stats[f"visits_with_{a}"] = visits.groupby("survey")[f"has_{a}"].sum().astype(int).to_dict()
    (out_dir / "build_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    return stats
