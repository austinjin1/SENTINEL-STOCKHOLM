# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Taxonomy harmonisation across NARS (US) and Environment Agency (England) community data.

Output: processed('taxonomy_harmonised')

    taxon_map.parquet        one row per source taxon (dataset x source id x name): cleaned name,
                             identification rank, harmonised lineage (phylum..species), lineage
                             source, analysis rank/taxon, mixed-resolution OTU, conflict flags
    name_backbone.parquet    cleaned name -> GBIF backbone match (rank, status, lineage)
    conflicts.parquet        (name, rank) pairs where sources disagree with each other or with GBIF
    otu_rules.parquet        macroinvertebrate families and whether genus-level ID is consistent
    summary.json             counts, conflict rates, chosen analysis resolutions

Rules (fixed, documented in docs/datasheets/taxonomy_harmonised.md):

1. **Name cleaning** — upper-case, collapse spaces, drop vernacular parentheses ("Sphaeriidae (Pea
   mussels)"), trailing "SP."/"SPP."/"GR."/"AGG."/"S.L.", "UNDETERMINED"/"UNIDENTIFIED" words.
   Slash complexes ("CRICOTOPUS/ORTHOCLADIUS") are kept as one taxon of rank ``complex`` whose
   lineage is the deepest rank shared by all parts.
2. **Lineage** — family = vote over distinct source datasets (+1 vote for the GBIF backbone match,
   which also breaks ties; then the most recent NARS list). Ranks above family come from a
   family -> (kingdom, phylum, class, order) lookup built only from names whose GBIF family equals
   the voted family, so every member of a family gets the same higher lineage and GBIF homonym
   matches (e.g. the beetle genus Sphaeridium matched to a bivalve family) are rejected. Without a
   lookup entry: GBIF (if its family agrees), else the source consensus. Names identified above
   family use GBIF when the rank matches, else the source consensus. EA parent chains contribute
   only family and below (EA labels some orders as 'Class'). Source misspellings are fixed only
   from the reviewed list ``REVIEWED_SPELLING_FIXES``; automatic fuzzy correction was tested and
   rejected (it merged distinct taxa). ``lineage_source`` says which rule applied.
3. **Conflicts** — for every name and rank (phylum, class, order, family, genus) all distinct
   non-null values across the GBIF match and every source lineage are compared; disagreements are
   listed with ``conflict_type`` (between_sources / source_vs_gbif / spelling_corrected) and
   ``systematic`` = the same value pattern occurs for >= 5 names (classification-system
   differences such as Bacillariophyta vs Ochrophyta) as opposed to isolated disagreements.
4. **Analysis resolution** — macroinvertebrates: family (EA records are 50% family-level);
   fish: species; zooplankton: genus; algae/diatoms: genus. Identifications coarser than the
   analysis rank keep their own name (``analysis_coarser`` = True) so models can use mixed
   resolution. ``mixed_otu`` (macroinvertebrates): genus where the family is consistently
   identified to genus (>= 90% of records at genus or finer in that data source, >= 20 records),
   else family.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .bio_common import write_parquet

log = logging.getLogger(__name__)

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
RANK_ORDER = {"kingdom": 0, "phylum": 1, "subphylum": 1.5, "class": 2, "subclass": 2.5, "superorder": 2.7,
              "order": 3, "suborder": 3.5, "infraorder": 3.6, "superfamily": 3.8, "family": 4,
              "subfamily": 4.5, "tribe": 4.7, "complex": 4.9, "genus": 5, "subgenus": 5.5, "species": 6,
              "subspecies": 6.5, "variety": 6.6, "form": 6.7}
ASSEMBLAGE_GROUP = {"benthic": "macroinvertebrate", "invertebrate": "macroinvertebrate", "fish": "fish",
                    "zooplankton": "zooplankton", "phytoplankton": "algae", "diatom": "algae"}
ANALYSIS_RANK = {"macroinvertebrate": "family", "fish": "species", "zooplankton": "genus", "algae": "genus",
                 "macrophyte": "genus", "other_vertebrate": "species", "protozoa": "genus", "bacteria": "genus"}
GBIF_KINGDOM_HINT = {"macroinvertebrate": "Animalia", "fish": "Animalia", "zooplankton": "Animalia",
                     "other_vertebrate": "Animalia", "macrophyte": "Plantae"}
NARS_RECENCY = {"nrsa2324": 8, "nla22": 7, "nrsa1819": 6, "nla17": 5, "nrsa1314": 4, "nla12": 3,
                "nrsa0809": 2, "nla07": 1}
GENUS_CONSISTENCY = 0.90
GENUS_CONSISTENCY_MIN_RECORDS = 20

_TRAIL = re.compile(r"(\s+(SP|SPP|SP\.|SPP\.|GR|GR\.|AGG|AGG\.|S\.L\.|S\.STR\.|CF\.?|NR\.?|"
                    r"UNDETERMINED|UNIDENTIFIED|UNKNOWN|UNID\.?|IMMATURE|LARVAE|PUPAE|ADULT)\b\.?)+$")


def clean_name(name) -> str | None:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip().upper()
    s = re.sub(r"\([^)]*\)", " ", s)          # vernacular / authority parentheses
    s = s.replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    prev = None
    while prev != s:
        prev = s
        s = _TRAIL.sub("", s).strip()
    s = re.sub(r"\s*/\s*", "/", s)
    return s or None


def gbif_query_name(clean: str) -> str:
    """'BAETIS RHODANI' -> 'Baetis rhodani' (GBIF matching expects scientific capitalisation)."""
    parts = clean.split(" ")
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])


def identification_rank(name: str, lineage: dict) -> str:
    """Rank of the identified taxon from its name and the published lineage (finest match)."""
    if name is None:
        return "unknown"
    if "/" in name:
        return "complex"
    for r in ("species", "genus", "tribe", "subfamily", "family", "superfamily", "suborder", "order",
              "subclass", "class", "subphylum", "phylum"):
        v = lineage.get(r)
        if v and str(v).upper() == name:
            return r
    words = name.split(" ")
    if len(words) >= 2 and lineage.get("genus") and words[0] == str(lineage["genus"]).upper():
        return "species"
    if len(words) >= 2:
        return "species"
    return "unknown"


# --------------------------------------------------------------- source lineages

def nars_source_taxa(taxa_source: pd.DataFrame, counts: dict[str, pd.DataFrame], dataset_prefix: str) -> pd.DataFrame:
    """One row per (survey, assemblage, taxa_id, name) with lineage and record counts."""
    t = taxa_source.copy()
    rows = []
    rec = {}
    for a, c in counts.items():
        g = c.groupby(["survey", "taxa_id", "taxon_name"], dropna=False).agg(
            n_records=("visit_id", "size"), total_count=("count", "sum")).reset_index()
        g["assemblage"] = a
        rec[a] = g
    recs = pd.concat(rec.values(), ignore_index=True) if rec else pd.DataFrame()
    for _, r in t.iterrows():
        lin = {k: r.get(f"src_{k}") for k in ("phylum", "subphylum", "class", "subclass", "order", "suborder",
                                                 "family", "subfamily", "tribe", "genus")}
        sp = r.get("src_species")
        if isinstance(sp, str) and sp:
            lin["species"] = sp if " " in sp else (f"{lin['genus']} {sp}" if lin.get("genus") else None)
        if r["assemblage"] == "phytoplankton" and not lin.get("phylum") and r.get("src_division"):
            lin["phylum"] = r.get("src_division")
        rows.append({"source": dataset_prefix, "dataset": r["survey"], "assemblage": r["assemblage"],
                     "source_taxon_id": r["taxa_id"], "source_taxon_name": r["taxon_name"],
                     **{f"src_{k}": (str(v).upper() if isinstance(v, str) and v else None) for k, v in lin.items()}})
    out = pd.DataFrame(rows)
    if not recs.empty:
        recs = recs.rename(columns={"survey": "dataset", "taxa_id": "source_taxon_id",
                                    "taxon_name": "source_taxon_name"})
        out = out.merge(recs, on=["dataset", "assemblage", "source_taxon_id", "source_taxon_name"], how="left")
    return out


def fish_scientific_names(src: pd.DataFrame) -> pd.DataFrame:
    """NARS fish are named by common name (FINAL_NAME). Lists that carry GENUS/SPECIES (2013-14,
    2023-24) map common name -> scientific name; other cycles inherit it by common name."""
    f = src[src["assemblage"] == "fish"].copy()
    known = f.dropna(subset=["src_genus"]).copy()
    known["sci"] = np.where(known["src_species"].notna(), known["src_species"], known["src_genus"])
    known["rec"] = known["dataset"].map(NARS_RECENCY).fillna(0)
    best = known.sort_values("rec", ascending=False).drop_duplicates("source_taxon_name")
    m = best.set_index("source_taxon_name")[["sci", "src_family", "src_genus", "src_species"]]
    for c in ("src_family", "src_genus", "src_species"):
        f[c] = f[c].where(f[c].notna(), f["source_taxon_name"].map(m[c]))
    f["scientific_name"] = f["source_taxon_name"].map(m["sci"])
    return f


def ea_source_taxa(dictionary: pd.DataFrame, counts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """EA keys used in counts with the parent chain flattened to standard ranks.

    Only standard rank labels are used (Phylum, Class, Order, Family, Subfamily, Tribe, Genus,
    Species); the EA dictionary mislabels some intermediate nodes (e.g. Diptera as 'Class'), so
    EA lineages are a last-resort lineage source (GBIF and NARS lists come first)."""
    d = dictionary.set_index("taxon_list_item_key")
    par = d["parent_tlik"].to_dict()
    rank = d["taxon_rank"].str.lower().to_dict()
    pname = d["preferred_taxon_name"].to_dict()
    rows = []
    for a, c in counts.items():
        g = c.groupby("taxon_list_item_key").agg(n_records=("analysis_id", "size"),
                                                 total_count=("abundance", "sum")).reset_index()
        for k, n, tot in g.itertuples(index=False):
            lin, seen, cur, chain = {}, set(), k, []
            while cur and cur in rank and cur not in seen:
                seen.add(cur)
                r = rank[cur]
                chain.append(f"{r}:{pname[cur]}")
                # EA labels some orders as 'Class' (e.g. Diptera), so only family and below are used
                if r in ("family", "subfamily", "tribe", "genus", "species") and r not in lin:
                    lin[r] = clean_name(pname[cur])
                cur = par.get(cur)
            info = d.loc[k] if k in d.index else None
            rows.append({"source": "uk_ea", "dataset": f"ea_{a}", "assemblage": a, "source_taxon_id": k,
                         "source_taxon_name": None if info is None else str(info["taxon_name"]).upper(),
                         "preferred_name": None if info is None else str(info["preferred_taxon_name"]).upper(),
                         "ea_rank": None if info is None else str(info["taxon_rank"]).lower(),
                         "ea_taxon_type": None if info is None else info["taxon_type"],
                         "n_records": n, "total_count": tot, "ea_chain": " > ".join(chain),
                         **{f"src_{r}": v for r, v in lin.items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ harmonising

def _gbif_lineage(m: dict | None) -> dict:
    if not m:
        return {}
    return {r: (str(m.get(r)).upper() if m.get(r) else None) for r in RANKS}


def _gbif_ok(m: dict | None) -> bool:
    if not m:
        return False
    mt = m.get("matchType")
    return mt == "EXACT" or (mt == "FUZZY" and (m.get("confidence") or 0) >= 90)


def _compatible(id_rank: str, gbif_rank: str | None) -> bool:
    if not gbif_rank:
        return False
    g = gbif_rank.lower()
    if id_rank in ("unknown", "complex"):
        return False
    if id_rank in ("variety", "form", "subspecies"):
        return g in ("species", "subspecies", "variety", "form")
    return g == id_rank


EA_TYPE_GROUP = {"Other Macroinvertebrates": "macroinvertebrate", "Macrophytes": "macrophyte", "Algae": "algae",
                 "Fish": "fish", "Birds": "other_vertebrate", "Mammals": "other_vertebrate",
                 "Amphibians": "other_vertebrate", "Reptiles": "other_vertebrate", "Protozoa": "protozoa",
                 "Bacteria": "bacteria"}


def assign_groups(src: pd.DataFrame) -> pd.DataFrame:
    """Taxon group per row: by assemblage, except EA records, which follow the EA taxon type
    (EA invertebrate samples also list macrophytes, algae and vertebrates seen in the net)."""
    df = src.copy()
    df["group"] = df["assemblage"].map(ASSEMBLAGE_GROUP)
    if "ea_taxon_type" in df:
        ea = df["source"] == "uk_ea"
        tg = df.loc[ea, "ea_taxon_type"].map(EA_TYPE_GROUP)
        df.loc[ea, "group"] = tg.where(tg.notna(), df.loc[ea, "group"])
    return df


def query_names(src: pd.DataFrame) -> pd.DataFrame:
    """Distinct (group, cleaned name) pairs to look up in the GBIF backbone (complex parts too)."""
    df = assign_groups(src)
    base = df["source_taxon_name"]
    if "preferred_name" in df:
        base = df["preferred_name"].where(df["preferred_name"].notna(), base)
    if "scientific_name" in df:
        base = df["scientific_name"].where(df["scientific_name"].notna(), base)
    df["name_clean"] = base.map(clean_name)
    pairs = set()
    for grp, nm in df[["group", "name_clean"]].dropna().itertuples(index=False):
        pairs.add((grp, nm))
        if "/" in nm:
            for part in nm.split("/"):
                if part:
                    pairs.add((grp, part))
    return pd.DataFrame(sorted(pairs), columns=["group", "name_clean"])


def _vote(g: pd.DataFrame, col: str, gbif_value: str | None) -> str | None:
    """Most supported value of a source lineage column: distinct datasets + one vote for GBIF;
    ties -> the GBIF value if tied, else the most recent NARS dataset."""
    v = g[[col, "dataset", "recency"]].dropna(subset=[col])
    if v.empty:
        return gbif_value
    agg = v.groupby(col).agg(n=("dataset", "nunique"), rec=("recency", "max"))
    if gbif_value is not None:
        agg["n"] = agg["n"] + (agg.index == gbif_value).astype(int)
    best = agg["n"].max()
    tied = agg[agg["n"] == best]
    if gbif_value is not None and gbif_value in tied.index:
        return gbif_value
    return tied.sort_values("rec", ascending=False).index[0]


HIGHER = ["kingdom", "phylum", "class", "order"]


SPELLING_RANKS = ("phylum", "class", "order", "family")
# Reviewed corrections of misspellings / orthographic variants found in the source taxa lists
# (2026-10-22). Each pair was checked by hand: same taxon, different spelling. Candidates that are
# distinct real taxa (e.g. Phoridae vs Phoronidae, Mesogastropoda vs Neogastropoda, superfamilies
# in the family column) are deliberately NOT corrected and appear in conflicts.parquet instead.
REVIEWED_SPELLING_FIXES = {
    ("phylum", "ANNILIDA"): "ANNELIDA", ("phylum", "NEMATA"): "NEMATODA",
    ("class", "MAXILLIPODA"): "MAXILLOPODA",
    ("order", "DECOPODA"): "DECAPODA", ("order", "CHATTONETALES"): "CHATTONELLALES",
    ("order", "LITTORIDINOMORPHA"): "LITTORINIMORPHA", ("order", "TUBIDICIDA"): "TUBIFICIDA",
    ("order", "CHROMALINALES"): "CHROMULINALES", ("order", "DINOPHYSIALIS"): "DINOPHYSIALES",
    ("order", "PEDINELLAES"): "PEDINELLALES", ("order", "ANTHOATHECATAE"): "ANTHOATHECATA",
    ("order", "COLLOTHECACEAE"): "COLLOTHECACEA", ("order", "RHYNCHOBDELLAE"): "RHYNCHOBDELLIDA",
    ("order", "VENEROIDA"): "VENERIDA", ("order", "UNIONOIDA"): "UNIONIDA", ("order", "MYTILOIDA"): "MYTILIDA",
    ("family", "CHAETOCEROTACAEA"): "CHAETOCEROTACEAE", ("family", "CHLORELLACAE"): "CHLORELLACEAE",
    ("family", "GOMPHONENATACEAE"): "GOMPHONEMATACEAE", ("family", "GYMONDINIACEAE"): "GYMNODINIACEAE",
    ("family", "OSYRRHINACEAE"): "OXYRRHINACEAE", ("family", "PRYENOMONADACEAE"): "PYRENOMONADACEAE",
    ("family", "THALASSIOSIARCEAE"): "THALASSIOSIRACEAE", ("family", "TRIBONEMATACAE"): "TRIBONEMATACEAE",
    ("family", "TRIBONEMACEAE"): "TRIBONEMATACEAE", ("family", "GERRIEDAE"): "GERRIDAE",
    ("family", "PALAMONIDAE"): "PALAEMONIDAE", ("family", "PARANELLIDAE"): "PARONELLIDAE",
    ("family", "D'ARCYTHOMPSONIIDAE"): "DARCYTHOMPSONIIDAE", ("family", "SKELETONEMATACEAE"): "SKELETONEMACEAE",
    ("family", "DINOPHYSIACEAE"): "DINOPHYSACEAE", ("family", "GEITLERINEMATACEAE"): "GEITLERINEMACEAE",
}


def _known_rank_values(backbone: dict[str, dict]) -> dict[str, set]:
    known = {r: set() for r in SPELLING_RANKS}
    for v in backbone.values():
        if _gbif_ok(v):
            for r, vals in known.items():
                if v.get(r):
                    vals.add(str(v[r]).upper())
    return known


def rank_values_to_verify(src: pd.DataFrame, backbone: dict[str, dict]) -> list[tuple[str, str]]:
    """(rank, value) pairs from source lineages that no GBIF name match uses — to be checked
    against GBIF at that rank before any spelling correction is considered."""
    df = assign_groups(src)
    known = _known_rank_values(backbone)
    out = []
    for r in SPELLING_RANKS:
        col = f"src_{r}"
        if col in df:
            out += [(r, v) for v in sorted(set(df[col].dropna().unique()) - known[r])]
    return out


def harmonise(src: pd.DataFrame, backbone: dict[str, dict],
              rank_values: dict[str, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (taxon_map, conflicts). See the module docstring for the rules.

    ``rank_values`` (GBIF matches of source higher-rank values, see ``rank_values_to_verify``)
    is only reported (``rank_value_checks.parquet``); corrections come from the reviewed list."""
    df = assign_groups(src)
    base = df["source_taxon_name"]
    if "preferred_name" in df:
        base = df["preferred_name"].where(df["preferred_name"].notna(), base)
    if "scientific_name" in df:
        base = df["scientific_name"].where(df["scientific_name"].notna(), base)
    df["name_clean"] = base.map(clean_name)
    df["recency"] = df["dataset"].map(NARS_RECENCY).fillna(0)
    src_cols = [c for c in df.columns if c.startswith("src_")]

    # pass 0: reviewed spelling fixes only (REVIEWED_SPELLING_FIXES); automatic fuzzy correction
    # was tried and rejected because it merged distinct real taxa (e.g. Phoridae -> Phoronidae).
    corrections = []
    for (r, bad), good in REVIEWED_SPELLING_FIXES.items():
        col = f"src_{r}"
        if col not in df:
            continue
        hit = df[col] == bad
        if not hit.any():
            continue
        for (grp, nm), gg in df[hit].groupby(["group", "name_clean"]):
            corrections.append({"group": grp, "name_clean": nm, "rank": r,
                                "values": {bad: sorted(set(gg["dataset"])), good: ["reviewed_fix"]},
                                "conflict_type": "spelling_corrected", "chosen": good,
                                "lineage_source": "reviewed_spelling_fix"})
        df.loc[hit, col] = good

    # pass 1: per (group, name) identification rank, GBIF lineage, voted family
    keys = {}
    for (grp, nm), g in df.groupby(["group", "name_clean"], dropna=False):
        m = backbone.get(f"{grp}|{nm}") if isinstance(nm, str) else None
        gl = _gbif_lineage(m) if _gbif_ok(m) else {}
        lin_src = {}
        for c in src_cols:
            r = c[4:]
            val = _vote(g, c, gl.get(r) if r in ("family", "genus") else None)
            if val is not None:
                lin_src[r] = val
        id_rank = identification_rank(nm, lin_src) if isinstance(nm, str) else "unknown"
        if id_rank == "unknown" and "ea_rank" in g and g["ea_rank"].notna().any():
            er = g["ea_rank"].dropna().iloc[0]
            id_rank = {"species aggregate": "species", "species group": "species", "species sensu lato": "species",
                       "subspecies aggregate": "subspecies"}.get(er, er if er in RANK_ORDER else "unknown")
        fam = lin_src.get("family")
        if fam is None and RANK_ORDER.get(id_rank, -1) >= RANK_ORDER["family"]:
            fam = gl.get("family")
        if id_rank == "family" and isinstance(nm, str):
            fam = nm
        keys[(grp, nm)] = {"g": g, "m": m, "gl": gl, "lin_src": lin_src, "id_rank": id_rank, "family": fam}

    # pass 2: family -> higher ranks, only from names where GBIF and the voted family agree
    votes: dict[tuple, dict] = {}
    for (grp, nm), k in keys.items():
        f, gl = k["family"], k["gl"]
        if f and gl.get("family") == f:
            d = votes.setdefault((grp, f), {})
            hk = tuple(gl.get(r) for r in HIGHER)
            d[hk] = d.get(hk, 0) + 1
    higher_by_family = {key: dict(zip(HIGHER, max(d.items(), key=lambda kv: kv[1])[0]))
                        for key, d in votes.items()}

    out_rows, conflicts = [], []
    for (grp, nm), k in keys.items():
        g, m, gl, lin_src, id_rank, fam = k["g"], k["m"], k["gl"], k["lin_src"], k["id_rank"], k["family"]
        rank_val = RANK_ORDER.get(id_rank, -1)
        harmon: dict = {}
        gbif_rejected = False
        if id_rank == "complex" and isinstance(nm, str):
            parts = [backbone.get(f"{grp}|{p}") for p in nm.split("/")]
            pl = [_gbif_lineage(p) for p in parts if _gbif_ok(p)]
            if pl and len(pl) == len(parts):
                for r in RANKS[:-2]:
                    vals = {p.get(r) for p in pl}
                    if len(vals) == 1 and None not in vals:
                        harmon[r] = vals.pop()
                    else:
                        break
            lsrc = "gbif_complex_shared" if harmon else "source_consensus"
            if not harmon:
                harmon = {r: lin_src.get(r) for r in RANKS[:-2]}
            if lin_src.get("family") and harmon.get("family") in (None, lin_src.get("family")):
                harmon["family"] = lin_src.get("family")
        elif rank_val >= RANK_ORDER["family"] and fam:
            harmon["family"] = fam
            hb = higher_by_family.get((grp, fam))
            if hb:
                harmon.update({r: v for r, v in hb.items() if v})
                lsrc = "family_lookup"
            elif gl and gl.get("family") in (None, fam) and _compatible(id_rank, (m or {}).get("rank")):
                harmon.update({r: gl.get(r) for r in HIGHER if gl.get(r)})
                lsrc = "gbif"
            else:
                harmon.update({r: lin_src.get(r) for r in HIGHER if lin_src.get(r)})
                lsrc = "source_consensus"
                gbif_rejected = bool(gl) and gl.get("family") not in (None, fam)
            if rank_val >= RANK_ORDER["genus"]:
                gen = lin_src.get("genus") or (nm.split(" ")[0] if isinstance(nm, str) and " " in nm else None)
                if id_rank == "genus" and isinstance(nm, str):
                    gen = nm
                harmon["genus"] = gen or gl.get("genus")
            if rank_val >= RANK_ORDER["species"] and isinstance(nm, str) and len(nm.split(" ")) >= 2:
                harmon["species"] = " ".join(nm.split(" ")[:2])
            for r in ("subfamily", "tribe"):
                if lin_src.get(r):
                    harmon[r] = lin_src[r]
        else:
            # identified above family (order, class, phylum ...) or unknown rank
            if gl and _compatible(id_rank, (m or {}).get("rank")):
                harmon = {r: gl.get(r) for r in HIGHER if gl.get(r)}
                lsrc = "gbif"
            elif lin_src:
                harmon = {r: lin_src.get(r) for r in HIGHER if lin_src.get(r)}
                lsrc = "source_consensus"
            elif gl:
                harmon = {r: gl.get(r) for r in HIGHER if gl.get(r)}
                lsrc = "gbif_rank_mismatch"
            else:
                lsrc = "none"
        # conflicts (all distinct values per rank across datasets and GBIF)
        for r in ("phylum", "class", "order", "family", "genus"):
            vals: dict[str, set] = {}
            col = f"src_{r}"
            if col in g:
                for v, ds in g[[col, "dataset"]].dropna().itertuples(index=False):
                    vals.setdefault(v, set()).add(ds)
            if gl.get(r):
                vals.setdefault(gl[r], set()).add("gbif")
            if len(vals) > 1:
                src_vals = {v for v, d in vals.items() if d - {"gbif"}}
                conflicts.append({"group": grp, "name_clean": nm, "rank": r,
                                  "values": {kk: sorted(vv) for kk, vv in vals.items()},
                                  "conflict_type": "between_sources" if len(src_vals) > 1 else "source_vs_gbif",
                                  "chosen": harmon.get(r), "lineage_source": lsrc})
        for row in g.itertuples(index=False):
            rd = row._asdict()
            out_rows.append({
                **{c: rd.get(c) for c in ("source", "dataset", "assemblage", "group", "source_taxon_id",
                                          "source_taxon_name", "n_records", "total_count", "ea_taxon_type")},
                "name_clean": nm, "identification_rank": id_rank,
                **{f"h_{r}": harmon.get(r) for r in RANKS + ["subfamily", "tribe"]},
                "lineage_source": lsrc, "gbif_rejected_homonym_or_synonym": gbif_rejected,
                "gbif_usage_key": (m or {}).get("usageKey"), "gbif_match_type": (m or {}).get("matchType"),
                "gbif_rank": (m or {}).get("rank"), "gbif_status": (m or {}).get("status"),
                "gbif_confidence": (m or {}).get("confidence"),
                **{f"gbif_{r}": gl.get(r) for r in RANKS},
            })
    tm = pd.DataFrame(out_rows)
    cf = pd.DataFrame(conflicts + corrections)
    if not cf.empty:
        cf["pattern"] = cf["rank"] + ":" + cf["values"].map(lambda d: "|".join(sorted(d)))
        cf["pattern_n_names"] = cf.groupby("pattern")["name_clean"].transform("size")
        cf["systematic"] = cf["pattern_n_names"] >= 5
        cf["values"] = cf["values"].map(json.dumps)
        conflicted = set(zip(cf["group"], cf["name_clean"]))
        isolated = set(zip(cf.loc[~cf["systematic"], "group"], cf.loc[~cf["systematic"], "name_clean"]))
        between = set(zip(cf.loc[cf["conflict_type"] == "between_sources", "group"],
                          cf.loc[cf["conflict_type"] == "between_sources", "name_clean"]))
        pairs = list(zip(tm["group"], tm["name_clean"]))
        tm["has_conflict"] = [p in conflicted for p in pairs]
        tm["has_isolated_conflict"] = [p in isolated for p in pairs]
        tm["has_between_source_conflict"] = [p in between for p in pairs]
    else:
        tm["has_conflict"] = tm["has_isolated_conflict"] = tm["has_between_source_conflict"] = False
    return tm, cf


def add_analysis_resolution(tm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tm = tm.copy()
    ar = tm["group"].map(ANALYSIS_RANK)
    tm["analysis_rank"] = ar
    idr = tm["identification_rank"].map(RANK_ORDER)
    target = ar.map(RANK_ORDER)
    at, coarse, at_rank = [], [], []
    for i in range(len(tm)):
        r = ar.iat[i]
        val = tm[f"h_{r}"].iat[i] if r in RANKS else None
        finer_or_equal = pd.notna(idr.iat[i]) and idr.iat[i] >= target.iat[i]
        if finer_or_equal and isinstance(val, str) and val:
            at.append(val)
            coarse.append(False)
            at_rank.append(r)
        else:
            at.append(tm["name_clean"].iat[i])
            coarse.append(True)
            at_rank.append(tm["identification_rank"].iat[i])
    tm["analysis_taxon"] = at
    tm["analysis_taxon_rank"] = at_rank
    tm["analysis_coarser"] = coarse

    # mixed-resolution OTU for macroinvertebrates: genus where consistently identified
    mi = tm[tm["group"] == "macroinvertebrate"].copy()
    mi["source_group"] = mi["source"]
    mi["genus_or_finer"] = mi["identification_rank"].map(RANK_ORDER).fillna(-1) >= RANK_ORDER["genus"]
    fam = mi.dropna(subset=["h_family"]).groupby(["source_group", "h_family"]).apply(
        lambda g: pd.Series({"records": g["n_records"].fillna(0).sum(),
                             "records_genus_or_finer": g.loc[g["genus_or_finer"], "n_records"].fillna(0).sum()}),
        include_groups=False).reset_index()
    fam["frac_genus_or_finer"] = fam["records_genus_or_finer"] / fam["records"].replace(0, np.nan)
    fam["genus_consistent"] = (fam["frac_genus_or_finer"] >= GENUS_CONSISTENCY) & \
                              (fam["records"] >= GENUS_CONSISTENCY_MIN_RECORDS)
    rule = dict(zip(zip(fam["source_group"], fam["h_family"]), fam["genus_consistent"]))
    otu = []
    for i in range(len(tm)):
        if tm["group"].iat[i] != "macroinvertebrate":
            otu.append(tm["analysis_taxon"].iat[i])
            continue
        f, gname = tm["h_family"].iat[i], tm["h_genus"].iat[i]
        if isinstance(f, str) and rule.get((tm["source"].iat[i], f), False) and isinstance(gname, str) \
                and RANK_ORDER.get(tm["identification_rank"].iat[i], -1) >= RANK_ORDER["genus"]:
            otu.append(gname)
        else:
            otu.append(tm["analysis_taxon"].iat[i])
    tm["mixed_otu"] = otu
    return tm, fam


# ------------------------------------------------------------------- orchestration

def collect_sources(nars_dirs: dict[str, Path], ea_dir: Path) -> pd.DataFrame:
    """Source taxa (with record counts) from processed NARS and EA tables."""
    parts = []
    for res, d in nars_dirs.items():
        ts = pd.read_parquet(d / "taxa_source.parquet")
        counts = {a: pd.read_parquet(d / f"{a}_counts.parquet",
                                     columns=["visit_id", "survey", "taxa_id", "taxon_name", "count"])
                  for a in ("benthic", "fish", "zooplankton", "phytoplankton") if (d / f"{a}_counts.parquet").exists()}
        src = nars_source_taxa(ts, counts, res)
        if (src["assemblage"] == "fish").any():
            src = pd.concat([src[src["assemblage"] != "fish"], fish_scientific_names(src)], ignore_index=True)
        parts.append(src)
    dic = pd.read_parquet(ea_dir / "taxon_dictionary.parquet")
    counts = {a: pd.read_parquet(ea_dir / f"counts_{a}.parquet", columns=["analysis_id", "taxon_list_item_key", "abundance"])
              for a in ("invertebrate", "diatom")}
    parts.append(ea_source_taxa(dic, counts))
    return pd.concat(parts, ignore_index=True)


def load_backbone(jsonl: Path) -> dict[str, dict]:
    bb = {}
    if not jsonl.exists():
        return bb
    for line in jsonl.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            bb[f"{r['group']}|{r['name_clean']}"] = r.get("response") or {}
    return bb


def load_rank_values(jsonl: Path) -> dict[str, dict]:
    out = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[f"{r['rank']}|{r['value']}"] = r.get("response") or {}
    return out


def build(nars_dirs: dict[str, Path], ea_dir: Path, gbif_jsonl: Path, out_dir: Path) -> dict:
    src = collect_sources(nars_dirs, ea_dir)
    bb = load_backbone(gbif_jsonl)
    rv = load_rank_values(gbif_jsonl.parent / "gbif_rank_values.jsonl")
    tm, cf = harmonise(src, bb, rv)
    tm, fam = add_analysis_resolution(tm)
    nb = pd.DataFrame([{"key": k, "group": k.split("|", 1)[0], "name_clean": k.split("|", 1)[1],
                        **{f: v.get(f) for f in ("usageKey", "scientificName", "canonicalName", "rank", "status",
                                                 "matchType", "confidence", "kingdom", "phylum", "class", "order",
                                                 "family", "genus", "species", "acceptedUsageKey", "note")}}
                       for k, v in bb.items()])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(tm, out_dir / "taxon_map.parquet")
    write_parquet(cf, out_dir / "conflicts.parquet")
    write_parquet(fam, out_dir / "otu_rules.parquet")
    write_parquet(nb, out_dir / "name_backbone.parquet")
    rvc = pd.DataFrame([{"rank": k.split("|", 1)[0], "value": k.split("|", 1)[1],
                         "gbif_match_type": v.get("matchType"), "gbif_rank": v.get("rank"),
                         "gbif_canonical_name": v.get("canonicalName"), "gbif_status": v.get("status"),
                         "gbif_confidence": v.get("confidence")}
                        for k, v in rv.items()]) if rv else pd.DataFrame()
    if not rvc.empty:
        rvc["reviewed_fix"] = [REVIEWED_SPELLING_FIXES.get((r, v)) for r, v in zip(rvc["rank"], rvc["value"])]
        write_parquet(rvc, out_dir / "rank_value_checks.parquet")
    by = tm.groupby(["source", "group"])
    summary = {
        "source_taxa": len(tm),
        "by_source_group": by.size().rename("n").reset_index().to_dict(orient="records"),
        "distinct_names_by_group": tm.groupby("group")["name_clean"].nunique().to_dict(),
        "lineage_source_counts": tm["lineage_source"].value_counts().to_dict(),
        "gbif_match_types": nb["matchType"].value_counts(dropna=False).to_dict() if not nb.empty else {},
        "names_with_conflict": int(tm.drop_duplicates(["group", "name_clean"])["has_conflict"].sum()),
        "names_with_isolated_conflict": int(tm.drop_duplicates(["group", "name_clean"])["has_isolated_conflict"].sum()),
        "names_with_between_source_conflict": int(
            tm.drop_duplicates(["group", "name_clean"])["has_between_source_conflict"].sum()),
        "conflict_rows_by_rank": cf["rank"].value_counts().to_dict() if not cf.empty else {},
        "conflict_rows_by_type": cf.groupby(["conflict_type", "systematic"]).size().rename("n").reset_index()
        .to_dict(orient="records") if not cf.empty else [],
        "gbif_rejected_names": int(tm.drop_duplicates(["group", "name_clean"])["gbif_rejected_homonym_or_synonym"].sum()),
        "top_systematic_patterns": cf[cf["systematic"]].drop_duplicates("pattern").nlargest(15, "pattern_n_names")[
            ["pattern", "pattern_n_names"]].to_dict(orient="records") if not cf.empty else [],
        "analysis_rank": ANALYSIS_RANK,
        "records_at_or_finer_than_analysis_rank": {
            f"{s}|{g}": float(1 - (d["analysis_coarser"] * d["n_records"].fillna(0)).sum()
                              / max(d["n_records"].fillna(0).sum(), 1)) for (s, g), d in by},
        "genus_consistent_families": {s: int(fam[(fam["source_group"] == s) & fam["genus_consistent"]].shape[0])
                                      for s in fam["source_group"].unique()},
        "families_evaluated": {s: int((fam["source_group"] == s).sum()) for s in fam["source_group"].unique()},
        "genus_consistency_rule": {"threshold": GENUS_CONSISTENCY, "min_records": GENUS_CONSISTENCY_MIN_RECORDS},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
