"""M4 architecture-round-v2 tracks T1 (unseen chemical clusters) and T6 (duration extrapolation).

New module. It edits nothing: ``unseen_taxa.py``, ``unseen_taxa_guard.py`` and the M5-frozen
``{__init__,evaluate,features,tabpfn_model}.py`` / ``gpboost_model.py`` are read-only here, and the
T2 pins are untouched. ``prospective.parent_key`` is *called*, never modified, so the parent-key
definition is the project's own.

It implements the lead's rulings of 2026-11-10 on the v1 amendment review:

L1  every ADORE test partition carrying a scored registry row is derived BY CODE and its
    ``result_id``s leave the universe entirely (no test block, no pool, no auxiliary row).
L2  the hold-out unit is the connected component over (``t1_cluster``, ``parent_key`` family), so
    neither a cluster nor a parent-key family can span a held-out and a retained unit.
L4  the hold-out is drawn as a share of UNITS, since unit count and not row count governs power.
L6  no label-derived statistic of a sealed block is computed here. ``build_split`` never reads the
    target column, and the reports it writes carry counts and keys only.
L7  ``auxiliary_rows`` enforces the AUX-LEAK exclusions as counted assertions.
L8  the builder deterministically emits the sealed artefacts at their pinned paths, doing the
    column trim itself, so the pinned builder is the thing that produced the pinned files.

Every count-bearing step raises :class:`ArchV2Error` rather than warning, and
:func:`build_split` returns a report whose numbers the amendment quotes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.chemtox import prospective as PRO

__all__ = [
    "ArchV2Error", "EXPERIMENT", "ACUTE_ENDPOINTS", "CHRONIC_ENDPOINTS", "T2_RESERVED_CLASSES",
    "DURATION_WINDOW", "MIN_DURATION_H", "BAND_SPEC", "BAND_NAMES", "WINDOW_BAND",
    "MIN_UNITS_FOR_CLAIM", "BUTINA_CUTOFF",
    "SPLIT_SEED", "HOLDOUT_UNIT_SHARE", "ADORE_STRUCTURE_CLASSES",
    "SPENT_IDS_PATH", "CLUSTER_PATH", "SPLIT_PATH", "BLOCKS",
    "scored_adore_partitions", "spent_adore_result_ids", "adore_partition_inventory",
    "load_usable_acute", "build_universe", "cluster_chemicals", "holdout_units",
    "assign_bands", "build_split", "auxiliary_rows", "sha256_file",
]


class ArchV2Error(RuntimeError):
    """A pre-registered invariant of the T1/T6 tracks was violated."""


# --------------------------------------------------------------------------- pinned identifiers
EXPERIMENT = "chemtox_arch_v2_select"          #: validation/selection rows
EXPERIMENT_TEST = "chemtox_arch_v2_test"       #: the sealed looks (guard-owned)

ACUTE_ENDPOINTS = ("LC50", "EC50")
CHRONIC_ENDPOINTS = ("NOEC", "LOEC", "EC10", "MATC", "NOEL", "LOEL")

#: T2's three frozen unseen-taxa blocks, by ECOTOX ``class``. A strict superset of T2's reserved
#: ``result_id``s, so those blocks stay out without ever reading ``test_blocks.parquet``.
T2_RESERVED_CLASSES = ("Insecta", "Gastropoda", "Bivalvia", "Polyplacophora", "Cephalopoda",
                       "Amphibia")

#: ADORE's applicability domain (``unseen_taxa.ADORE_STRUCTURE_CLASSES``, quoted not imported so
#: this module does not depend on frozen code at import time).
ADORE_STRUCTURE_CLASSES = ("organic", "organic_multicomponent")

DURATION_WINDOW = (24.0, 96.0)

#: Minimum admissible exposure duration (DURATION-FLOOR-AND-BANDS). Below 15 minutes an aquatic
#: LC50/EC50 record is not distinguishable from an instantaneous dose, and both the crude baseline
#: and the candidate's time law diverge as t -> 0: Haber's rule gives C = k/t -> inf, and the
#: composition layer's -log10(-expm1(-k t)) -> -log10(k t) -> inf. Rows at or below it are dropped
#: from every block with the count recorded.
MIN_DURATION_H = 0.25

#: Duration bands. Every band is half-open **except** the training-window band, which is the
#: closed interval [24, 96] because that is what ``DURATION_WINDOW`` means and 24 h is the single
#: most common acute duration in the frame. Writing the bands literally as
#: ``(0,24], (24,96], ...`` (the review's wording) would put every exactly-24 h row -- 9,759 of
#: them, the largest single duration -- into the *short tail* while the training window kept them,
#: so "band" and "window" would disagree on the same row; or, if the window were narrowed to
#: (24,96] to match, a standard 24 h acute test would be scored as duration extrapolation, which
#: it is not. Aligning the one shared edge removes the ambiguity the half-open wording was for.
#: ``in_window`` is therefore not a separate predicate: it is ``band == "w24_96"``, asserted in
#: :func:`build_split`.
BAND_SPEC = (("lt24", "MIN < d < 24"), ("w24_96", "24 <= d <= 96"),
             ("gt96_168", "96 < d <= 168"), ("gt168_720", "168 < d <= 720"),
             ("gt720_2160", "720 < d <= 2160"), ("gt2160", "d > 2160"))
BAND_NAMES = tuple(n for n, _ in BAND_SPEC)
WINDOW_BAND = "w24_96"

#: No claim may rest on a band holding fewer than this many resampling units.
MIN_UNITS_FOR_CLAIM = 50

BUTINA_CUTOFF = 0.4           #: Tanimoto DISTANCE cutoff; justification in the amendment §2
MORGAN_RADIUS, MORGAN_BITS = 2, 2048
SPLIT_SEED = 20261110         #: unchanged from v1
HOLDOUT_UNIT_SHARE = 0.30     #: proposed share of UNITS (lead rules at the stamp, L4)

BLOCKS = ("train_pool", "T1_test", "T6a_test", "T6b_test")

_PREREG = Path(__file__).resolve().parents[3] / "docs" / "preregistration"
_PROC = paths.DATA_ROOT / "processed" / "chemtox_arch_v2"

SPENT_IDS_PATH = _PREREG / "chemtox_arch_v2_spent_adore_ids_v2.json"
CLUSTER_PATH = _PREREG / "chemtox_t1_clusters_v2.csv"
SPLIT_PATH = _PROC / "chemtox_t1t6_split_v2.parquet"

ECOTOX = paths.DATA_ROOT / "processed" / "ecotox" / "ecotox_aquatic.parquet"
PROSPECTIVE = paths.DATA_ROOT / "processed" / "chemtox_prospective" / "prospective.parquet"
ADORE_SPLITS = paths.DATA_ROOT / "processed" / "adore" / "split_assignments.parquet"
REGISTRY = Path(__file__).resolve().parents[3] / "results" / "registry.jsonl"

#: columns the sealed split carries. NO target column: the split is structure only (L6).
SPLIT_COLUMNS = ("result_id", "test_cas", "t1_cluster", "holdout_unit", "duration_h",
                 "duration_band", "in_adore_domain", "shared_reference", "block")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== L1  spent ADORE partitions

#: ``adore_trivial_baselines`` writes the split without the ``split_`` prefix, so partition names
#: are normalised before the union is taken. Nothing is typed by hand: the challenge x split pairs
#: come out of the registry.
_SPLIT_ALIASES = {"occurrence": "split_occurrence", "scaffold-murcko": "split_scaffold-murcko",
                  "scaffold-generic": "split_scaffold-generic", "random": "split_random",
                  "totallyrandom": "split_totallyrandom"}
_ADORE_CHALLENGE = re.compile(r"^(?:t|s|a)-[A-Za-z0-9-]+$")


def scored_adore_partitions(registry: Path | str = REGISTRY) -> dict[tuple[str, str], list[str]]:
    """Every ADORE ``challenge x split`` partition with at least one SCORED test row, by code.

    A registry row counts as having spent the partition when its ``split`` field names that
    partition's ``test`` role, whatever the experiment and whatever the model — a trivial baseline
    reads the same held-out labels as a candidate does, so it spends the block just the same
    (verifier finding SPENT-ADORE, lead ruling L1).
    """
    found: dict[tuple[str, str], list[str]] = {}
    with open(registry, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            split = row.get("split")
            if not isinstance(split, str):
                continue
            parts = split.split("/")
            if len(parts) != 3 or parts[2] != "test":
                continue
            challenge, name = parts[0], _SPLIT_ALIASES.get(parts[1], parts[1])
            if not _ADORE_CHALLENGE.match(challenge):
                continue
            found.setdefault((challenge, name), []).append(str(row.get("experiment")))
    return {k: sorted(set(v)) for k, v in sorted(found.items())}


def spent_adore_result_ids(registry: Path | str = REGISTRY,
                           splits: Path | str = ADORE_SPLITS) -> tuple[set[str], dict]:
    """The union of the ``result_id``s of every scored ADORE test partition, plus a report."""
    partitions = scored_adore_partitions(registry)
    if not partitions:
        raise ArchV2Error("no scored ADORE test partition found in the registry; refusing to "
                          "build a universe that silently keeps spent rows")
    sa = pd.read_parquet(splits, columns=["challenge", "result_id", "split", "role"])
    test = sa[sa["role"].astype("string") == "test"]
    ids: set[str] = set()
    per = {}
    for (challenge, name), experiments in partitions.items():
        g = test[(test["challenge"].astype("string") == challenge)
                 & (test["split"].astype("string") == name)]
        if g.empty:
            raise ArchV2Error(
                f"partition {challenge}/{name} carries a scored registry row but no test rows in "
                f"{splits}; the registry and the split table disagree")
        part_ids = set(g["result_id"].astype(str))
        per[f"{challenge}/{name}"] = {"test_result_ids": len(part_ids),
                                      "scored_in_experiments": experiments}
        ids |= part_ids
    report = {"n_partitions": len(partitions), "n_result_ids": len(ids), "per_partition": per,
              "rule": "every ADORE challenge x split partition with >= 1 scored test-role registry "
                      "row, in any experiment, including trivial baselines"}
    return ids, report


def adore_partition_inventory(registry: Path | str = REGISTRY,
                              splits: Path | str = ADORE_SPLITS) -> dict:
    """Scored vs unscored ADORE test partitions, counts only (the L1 report to the lead).

    Unscored partitions are **not** subtracted: that needs a lead ruling.
    """
    scored = set(scored_adore_partitions(registry))
    sa = pd.read_parquet(splits, columns=["challenge", "result_id", "split", "role"])
    test = sa[sa["role"].astype("string") == "test"]
    rows = []
    for (ch, sp), g in test.groupby([test["challenge"].astype(str), test["split"].astype(str)]):
        rows.append({"partition": f"{ch}/{sp}", "test_result_ids": int(g["result_id"].nunique()),
                     "scored": (ch, sp) in scored})
    rows.sort(key=lambda r: (-r["test_result_ids"], r["partition"]))
    return {"n_partitions_total": len(rows),
            "n_scored": sum(1 for r in rows if r["scored"]),
            "n_unscored": sum(1 for r in rows if not r["scored"]),
            "unscored_result_ids_total": sum(r["test_result_ids"] for r in rows
                                             if not r["scored"]),
            "partitions": rows}


# ================================================================================ the universe

@dataclass
class UniverseReport:
    counts: dict = field(default_factory=dict)

    def step(self, name: str, frame: pd.DataFrame, **extra) -> None:
        self.counts[name] = {"rows": int(len(frame)),
                             "cas": int(frame["test_cas"].nunique()) if len(frame) else 0,
                             "species": int(frame["latin_name"].nunique()) if len(frame) else 0,
                             **extra}


_USABLE_COLUMNS = ["result_id", "test_cas", "endpoint_norm", "censored", "log10_conc_mol_L",
                   "chem_smiles", "duration_h", "latin_name", "class", "reference_number",
                   "chem_structure_class", "chem_inchikey"]


def load_usable_acute(ecotox: Path | str = ECOTOX) -> pd.DataFrame:
    """The usable acute endpoint: LC50/EC50, uncensored, molar value present, SMILES present.

    ``log10_conc_mol_L`` is required to be PRESENT but its value is never read here (L6): the
    column is loaded for the presence test and dropped before anything is written.
    """
    e = pd.read_parquet(ecotox, columns=_USABLE_COLUMNS)
    keep = (e["endpoint_norm"].isin(ACUTE_ENDPOINTS)
            & (~e["censored"].astype(bool))
            & e["log10_conc_mol_L"].notna()
            & e["chem_smiles"].notna())
    return e[keep].drop(columns=["log10_conc_mol_L"]).copy()


def build_universe(ecotox: Path | str = ECOTOX, prospective: Path | str = PROSPECTIVE,
                   registry: Path | str = REGISTRY, splits: Path | str = ADORE_SPLITS,
                   *, expect: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply the four subtractions, each as a counted assertion. Returns (universe, report)."""
    rep = UniverseReport()
    usable = load_usable_acute(ecotox)
    rep.step("usable_acute", usable)

    # (1) T2's reserved classes, outright
    uni = usable[~usable["class"].astype("string").isin(T2_RESERVED_CLASSES)].copy()
    rep.step("after_t2_class_exclusion", uni, rows_removed=int(len(usable) - len(uni)))
    if (uni["class"].astype("string").isin(T2_RESERVED_CLASSES)).any():
        raise ArchV2Error("a T2-reserved class survived the exclusion")

    # (2) the spent prospective block, by result_id
    spent_pro = set(pd.read_parquet(prospective, columns=["result_id"])["result_id"]
                    .dropna().astype(str))
    before = len(uni)
    uni = uni[~uni["result_id"].astype(str).isin(spent_pro)].copy()
    rep.step("after_prospective_subtraction", uni, rows_removed=int(before - len(uni)),
             prospective_result_ids=len(spent_pro))
    if uni["result_id"].astype(str).isin(spent_pro).any():
        raise ArchV2Error("a spent prospective result_id survived the subtraction")

    # (3) L1: every scored ADORE test partition, by result_id
    spent_adore, adore_rep = spent_adore_result_ids(registry, splits)
    before = len(uni)
    hit = uni["result_id"].astype(str).isin(spent_adore)
    uni = uni[~hit].copy()
    rep.step("after_spent_adore_subtraction", uni, rows_removed=int(before - len(uni)),
             spent_adore_partitions=adore_rep["n_partitions"],
             spent_adore_result_ids=adore_rep["n_result_ids"])
    if uni["result_id"].astype(str).isin(spent_adore).any():
        raise ArchV2Error("a scored ADORE test result_id survived the subtraction (L1)")

    # (4) duration floor and presence
    before = len(uni)
    uni = uni[uni["duration_h"].notna()].copy()
    no_duration = int(before - len(uni))
    before = len(uni)
    uni = uni[uni["duration_h"] > MIN_DURATION_H].copy()
    rep.step("after_duration_floor", uni, rows_without_duration=no_duration,
             rows_at_or_below_floor=int(before - len(uni)), min_duration_h=MIN_DURATION_H)
    if (uni["duration_h"] <= MIN_DURATION_H).any():
        raise ArchV2Error("a row at or below the duration floor survived")

    uni["in_adore_domain"] = (uni["chem_structure_class"].astype("string")
                              .isin(ADORE_STRUCTURE_CLASSES).fillna(False))
    rep.counts["universe"] = {
        "rows": int(len(uni)), "cas": int(uni["test_cas"].nunique()),
        "species": int(uni["latin_name"].nunique()),
        "references": int(uni["reference_number"].nunique()),
        "rows_in_adore_domain": int(uni["in_adore_domain"].sum()),
        "rows_out_of_adore_domain": int((~uni["in_adore_domain"]).sum())}
    out = {"steps": rep.counts, "spent_adore": adore_rep}
    if expect:
        for k, v in expect.items():
            got = rep.counts.get(k, {}).get("rows")
            if got != v:
                raise ArchV2Error(f"universe step {k!r}: expected {v} rows, measured {got}")
    return uni, out


# ============================================================== clustering and the hold-out unit

def cluster_chemicals(universe: pd.DataFrame, *, cutoff: float = BUTINA_CUTOFF) -> pd.DataFrame:
    """ECFP4 Butina clusters over one canonical SMILES per CAS. Deterministic in CAS order."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    RDLogger.DisableLog("rdApp.*")
    smi = (universe.groupby("test_cas")["chem_smiles"]
           .agg(lambda s: s.value_counts().index[0]).sort_index())
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    cas, fps, bad = [], [], []
    for c, s in smi.items():
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            bad.append(c)
            continue
        cas.append(c)
        fps.append(gen.GetFingerprint(mol))
    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - x for x in sims)
    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    cid = {cas[m]: k for k, members in enumerate(clusters) for m in members}
    table = pd.DataFrame({"test_cas": cas,
                          "chem_smiles": [smi[c] for c in cas],
                          "parent_key": [PRO.parent_key(smi[c]) for c in cas],
                          "t1_cluster": [cid[c] for c in cas]})
    table.attrs["unparseable_cas"] = sorted(bad)
    return table.sort_values(["t1_cluster", "test_cas"]).reset_index(drop=True)


def holdout_units(table: pd.DataFrame) -> pd.DataFrame:
    """L2: connected components over (``t1_cluster``, ``parent_key`` family).

    Two chemicals are in the same hold-out unit when they share an ECFP4 cluster **or** a parent
    key, transitively. Holding out whole components is what makes 'no parent_key family spans a
    held-out and a retained unit' true by construction rather than by inspection.
    """
    if table["test_cas"].duplicated().any():
        raise ArchV2Error("cluster table must hold one row per CAS")
    cas = table["test_cas"].tolist()
    pos = {c: i for i, c in enumerate(cas)}
    parent = list(range(len(cas)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _, g in table.groupby("t1_cluster")["test_cas"]:
        idx = [pos[c] for c in g]
        for j in idx[1:]:
            union(idx[0], j)
    pk = table["parent_key"].astype("string")
    for _, g in table[pk.notna()].groupby(pk[pk.notna()])["test_cas"]:
        idx = [pos[c] for c in g]
        for j in idx[1:]:
            union(idx[0], j)

    roots = [find(i) for i in range(len(cas))]
    order = {r: k for k, r in enumerate(sorted(set(roots)))}
    out = table.copy()
    out["holdout_unit"] = [order[r] for r in roots]
    # the invariant, asserted
    for col in ("t1_cluster", "parent_key"):
        n = out.dropna(subset=[col]).groupby(col)["holdout_unit"].nunique()
        if (n > 1).any():
            raise ArchV2Error(f"{col} spans more than one hold-out unit: "
                              f"{n[n > 1].head().to_dict()}")
    return out


def assign_bands(duration: pd.Series) -> pd.Series:
    """Duration bands per :data:`BAND_SPEC`; the window band is closed, the rest half-open.

    Built from explicit masks rather than ``pd.cut`` so that the window band and
    :data:`DURATION_WINDOW` are the same set by construction and cannot drift apart.
    """
    d = pd.to_numeric(duration, errors="coerce")
    lo, hi = DURATION_WINDOW
    out = pd.Series(pd.NA, index=d.index, dtype="string")
    out[(d > MIN_DURATION_H) & (d < lo)] = "lt24"
    out[(d >= lo) & (d <= hi)] = WINDOW_BAND
    out[(d > hi) & (d <= 168.0)] = "gt96_168"
    out[(d > 168.0) & (d <= 720.0)] = "gt168_720"
    out[(d > 720.0) & (d <= 2160.0)] = "gt720_2160"
    out[d > 2160.0] = "gt2160"
    return out


def _draw_units(unit_ids: np.ndarray, share: float, seed: int) -> set[int]:
    """Draw a share of UNITS (L4), by permutation under the declared seed."""
    if not 0.0 < share < 1.0:
        raise ArchV2Error(f"hold-out share must be in (0, 1), got {share}")
    uniq = np.sort(np.unique(unit_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    k = int(round(share * len(uniq)))
    if k < 1 or k >= len(uniq):
        raise ArchV2Error(f"share {share} gives {k} of {len(uniq)} units")
    return set(int(x) for x in perm[:k])


def build_split(*, share: float = HOLDOUT_UNIT_SHARE, seed: int = SPLIT_SEED,
                ecotox: Path | str = ECOTOX, prospective: Path | str = PROSPECTIVE,
                registry: Path | str = REGISTRY, splits: Path | str = ADORE_SPLITS,
                cluster_path: Path | str | None = CLUSTER_PATH,
                split_path: Path | str | None = SPLIT_PATH,
                spent_ids_path: Path | str | None = SPENT_IDS_PATH,
                write: bool = True) -> dict:
    """Build and (optionally) emit the sealed label-free split, cluster file and spent-id list.

    L8: this function does the column trim itself and writes exactly the files the amendment pins,
    so the pinned builder is the thing that produced the pinned artefacts. Nothing is written to a
    temporary directory and no target value is read.
    """
    uni, urep = build_universe(ecotox, prospective, registry, splits)
    spent_adore, adore_rep = spent_adore_result_ids(registry, splits)

    table = holdout_units(cluster_chemicals(uni))
    unparseable = table.attrs.get("unparseable_cas", [])
    uni = uni.merge(table[["test_cas", "t1_cluster", "holdout_unit"]], on="test_cas", how="left")
    dropped_unparseable = int(uni["t1_cluster"].isna().sum())
    uni = uni[uni["t1_cluster"].notna()].copy()
    uni["t1_cluster"] = uni["t1_cluster"].astype(int)
    uni["holdout_unit"] = uni["holdout_unit"].astype(int)

    uni["duration_band"] = assign_bands(uni["duration_h"])
    if uni["duration_band"].isna().any():
        raise ArchV2Error("a row fell outside every declared duration band")
    in_window = (uni["duration_band"] == WINDOW_BAND).fillna(False).to_numpy(dtype=bool)
    if not (in_window == uni["duration_h"].between(*DURATION_WINDOW).to_numpy(dtype=bool)).all():
        raise ArchV2Error("the window band and DURATION_WINDOW disagree on a row")
    held = _draw_units(uni["holdout_unit"].to_numpy(), share, seed)
    is_held = uni["holdout_unit"].isin(held).to_numpy(dtype=bool)
    uni["block"] = np.where(is_held & in_window, "T1_test",
                   np.where(~is_held & ~in_window, "T6a_test",
                   np.where(is_held & ~in_window, "T6b_test", "train_pool")))
    # ---- invariants
    pool_units = set(uni.loc[uni["block"] == "train_pool", "holdout_unit"])
    test_units = set(uni.loc[uni["block"].isin(("T1_test", "T6b_test")), "holdout_unit"])
    if pool_units & test_units:
        raise ArchV2Error(f"{len(pool_units & test_units)} hold-out units are in both the pool and "
                          f"a held-unit test block")
    pool_pk = set(table.loc[table["holdout_unit"].isin(pool_units), "parent_key"].dropna())
    held_pk = set(table.loc[table["holdout_unit"].isin(held), "parent_key"].dropna())
    if pool_pk & held_pk:
        raise ArchV2Error(f"{len(pool_pk & held_pk)} parent-key families span the hold-out (L2)")
    if uni["result_id"].astype(str).isin(spent_adore).any():
        raise ArchV2Error("spent ADORE rows reached the split")
    pool_dur = uni.loc[uni["block"] == "train_pool", "duration_h"]
    if len(pool_dur) and not pool_dur.between(*DURATION_WINDOW).all():
        raise ArchV2Error("an out-of-window row reached train_pool")

    # a row's publication is "shared" when its reference_number also appears in the pool. Written
    # into the split so every downstream power computation reads structure, never a label (L6).
    pool_refs = set(uni.loc[uni["block"] == "train_pool", "reference_number"].dropna().astype(str))
    uni["shared_reference"] = uni["reference_number"].astype(str).isin(pool_refs)

    # ---- report: counts and keys only (L6)
    def unit_sizes(frame: pd.DataFrame) -> list[int]:
        return frame.groupby("holdout_unit").size().sort_values(ascending=False).tolist()

    blocks = {}
    for b, g in uni.groupby("block"):
        dom = g[g["in_adore_domain"]]
        blocks[str(b)] = {
            "rows": int(len(g)), "chemicals": int(g["test_cas"].nunique()),
            "clusters": int(g["t1_cluster"].nunique()),
            "holdout_units": int(g["holdout_unit"].nunique()),
            "species": int(g["latin_name"].nunique()),
            "references": int(g["reference_number"].nunique()),
            "rows_in_adore_domain": int(len(dom)),
            "holdout_units_in_adore_domain": int(dom["holdout_unit"].nunique()),
            "largest_unit_rows": int(unit_sizes(g)[0]) if len(g) else 0,
            "duration_min": float(g["duration_h"].min()), "duration_max": float(g["duration_h"].max()),
            "band_rows": {n: int((g["duration_band"] == n).sum()) for n in BAND_NAMES
                          if (g["duration_band"] == n).any()},
            "band_units": {n: int(g.loc[g["duration_band"] == n, "holdout_unit"].nunique())
                           for n in BAND_NAMES if (g["duration_band"] == n).any()},
            "band_claimable": {n: bool(g.loc[g["duration_band"] == n, "holdout_unit"].nunique()
                                       >= MIN_UNITS_FOR_CLAIM)
                               for n in BAND_NAMES if (g["duration_band"] == n).any()},
            "extrapolation_reach_x": round(float(g["duration_h"].max()) / DURATION_WINDOW[1], 1),
        }
    for b in ("T1_test", "T6a_test", "T6b_test"):
        g = uni[uni["block"] == b]
        if not len(g):
            continue
        sh = g["shared_reference"]
        blocks[b]["shared_reference_rows"] = int(sh.sum())
        blocks[b]["shared_reference_row_share"] = round(float(sh.mean()), 4)
        blocks[b]["no_shared_reference_rows"] = int((~sh).sum())
        blocks[b]["no_shared_reference_units"] = int(g.loc[~sh, "holdout_unit"].nunique())

    all_units = uni.groupby("holdout_unit").size()
    report = {
        "generated_by": "sentinel.models.chemtox.arch_v2.build_split",
        "share_of_units_held_out": share, "split_seed": seed,
        "butina_cutoff": BUTINA_CUTOFF, "morgan": [MORGAN_RADIUS, MORGAN_BITS],
        "duration_window": list(DURATION_WINDOW), "min_duration_h": MIN_DURATION_H,
        "bands": {n: d for n, d in BAND_SPEC}, "window_band": WINDOW_BAND,
        "min_units_for_claim": MIN_UNITS_FOR_CLAIM,
        "universe": urep,
        "chemicals_total": int(len(table) + len(unparseable)),
        "chemicals_unparseable": len(unparseable),
        "rows_dropped_unparseable_smiles": dropped_unparseable,
        "n_clusters": int(table["t1_cluster"].nunique()),
        "n_holdout_units": int(table["holdout_unit"].nunique()),
        "units_held_out": len(held),
        "largest_unit_rows": int(all_units.max()),
        "largest_unit_row_share": round(float(all_units.max() / len(uni)), 4),
        "largest_cluster_rows": int(uni.groupby("t1_cluster").size().max()),
        "largest_cluster_row_share": round(float(uni.groupby("t1_cluster").size().max() / len(uni)), 4),
        "singleton_units": int((table.groupby("holdout_unit").size() == 1).sum()),
        "blocks": blocks,
        "label_derived_statistics": "none (L6): no target value is read by this builder",
    }

    if write:
        out = uni[list(SPLIT_COLUMNS)].copy()
        out["duration_band"] = out["duration_band"].astype(str)
        out = out.sort_values("result_id").reset_index(drop=True)
        if set(out.columns) != set(SPLIT_COLUMNS):
            raise ArchV2Error("split column trim failed")
        Path(split_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(split_path, index=False)
        table.to_csv(cluster_path, index=False, lineterminator="\n")
        Path(spent_ids_path).write_text(json.dumps(
            {"rule": adore_rep["rule"], "n_partitions": adore_rep["n_partitions"],
             "n_result_ids": adore_rep["n_result_ids"],
             "per_partition": adore_rep["per_partition"],
             "result_ids": sorted(spent_adore)}, indent=1, sort_keys=True) + "\n")
        report["artefacts"] = {
            str(split_path): sha256_file(Path(split_path)),
            str(cluster_path): sha256_file(Path(cluster_path)),
            str(spent_ids_path): sha256_file(Path(spent_ids_path))}
    return report


# ========================================================== L7  auxiliary / multitask row loader

def auxiliary_rows(target: str, split: pd.DataFrame, *, track: str,
                   ecotox: Path | str = ECOTOX, prospective: Path | str = PROSPECTIVE,
                   registry: Path | str = REGISTRY, splits: Path | str = ADORE_SPLITS,
                   cluster_table: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Auxiliary rows admissible as training signal, with every AUX-LEAK exclusion counted.

    ``target`` is ``"H2"`` (censored acute), ``"H3"`` (chronic) or ``"H4"`` (BCF-family, handled by
    the caller's own frame). ``track`` is ``"T1"`` or ``"T6"``.

    Exclusions, each counted, in order:
      1. T2's reserved classes (as for the primary endpoint);
      2. spent prospective ``result_id``s;
      3. **L1** spent ADORE ``result_id``s;
      4. any row whose ``test_cas`` sits in a held-out hold-out unit — which, because the unit is
         the connected component over (cluster, parent-key family), also removes every row whose
         cluster or parent-key family touches the hold-out;
      5. any row sharing a ``(test_cas, latin_name, duration_h)`` cell with a sealed row;
      6. for ``track == "T6"`` only, any row outside the 24-96 h window, so no long-duration label
         can reach a model whose claim is long-duration extrapolation.
    """
    if track not in ("T1", "T6"):
        raise ArchV2Error(f"track must be 'T1' or 'T6', got {track!r}")
    if target not in ("H2", "H3"):
        raise ArchV2Error(f"auxiliary_rows handles H2 and H3; got {target!r}")

    e = pd.read_parquet(ecotox, columns=_USABLE_COLUMNS)
    if target == "H2":
        sel = (e["endpoint_norm"].isin(ACUTE_ENDPOINTS) & e["censored"].astype(bool)
               & e["chem_smiles"].notna() & e["log10_conc_mol_L"].notna())
    else:
        sel = (e["endpoint_norm"].isin(CHRONIC_ENDPOINTS) & (~e["censored"].astype(bool))
               & e["chem_smiles"].notna() & e["log10_conc_mol_L"].notna())
    aux = e[sel].copy()
    rep = {"target": target, "track": track, "universe_rows": int(len(aux))}

    def cut(frame: pd.DataFrame, name: str, mask) -> pd.DataFrame:
        # the mask is coerced to a real boolean Series on the frame's own index: a float or object
        # mask would be read by pandas as a list of COLUMN labels, which is how a dtype mismatch
        # turns a row filter into a KeyError or, worse, into a silent no-op.
        m = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
        out = frame[~m].copy()
        rep[f"removed_{name}"] = int(len(frame) - len(out))
        return out

    aux = cut(aux, "t2_reserved_classes",
              aux["class"].astype("string").isin(T2_RESERVED_CLASSES))
    spent_pro = set(pd.read_parquet(prospective, columns=["result_id"])["result_id"]
                    .dropna().astype(str))
    aux = cut(aux, "spent_prospective", aux["result_id"].astype(str).isin(spent_pro))
    spent_adore, _ = spent_adore_result_ids(registry, splits)
    aux = cut(aux, "spent_adore", aux["result_id"].astype(str).isin(spent_adore))

    held_units = set(split.loc[split["block"].isin(("T1_test", "T6b_test")), "holdout_unit"]
                     .astype(int))
    # CAS keys are numeric strings, so a CSV round-trip turns them into int64. Normalising both
    # sides to str is what keeps this join from silently matching nothing (and the coverage
    # assertion below is what catches it if it ever does).
    src = cluster_table if cluster_table is not None else split.drop_duplicates("test_cas")
    cas_unit = (src.assign(_k=src["test_cas"].astype(str))
                .drop_duplicates("_k").set_index("_k")["holdout_unit"].astype(int))
    aux_cas = aux["test_cas"].astype(str)
    mapped = aux_cas.map(cas_unit)
    coverage = float(mapped.notna().mean()) if len(aux) else 1.0
    rep["chemistry_mapped_fraction"] = round(coverage, 4)
    if len(aux) and coverage < 0.10:
        raise ArchV2Error(
            f"only {coverage:.1%} of auxiliary rows mapped to a hold-out unit; the CAS join is "
            f"degenerate (dtype mismatch?) and the AUX-LEAK filter would be a silent no-op")
    aux = cut(aux, "held_out_unit_chemistry", mapped.isin(held_units).fillna(False))
    # chemistry absent from the T1/T6 universe has no cluster or parent-key assignment, so whether
    # it touches the hold-out is UNKNOWN; the conservative reading of AUX-LEAK excludes it.
    keep = aux["test_cas"].astype(str).map(cas_unit).notna()
    rep["removed_chemistry_not_in_split"] = int((~keep).sum())
    aux = aux[keep].copy()

    sealed = split[split["block"].isin(("T1_test", "T6a_test", "T6b_test"))]
    # the sealed cell key needs the species, which the split file does not carry; it is rebuilt
    # from the primary frame by result_id so no sealed label is read.
    prim = e.loc[e["result_id"].astype(str).isin(set(sealed["result_id"].astype(str))),
                 ["test_cas", "latin_name", "duration_h"]]
    cells = set(map(tuple, prim.astype(str).to_numpy()))
    key = list(map(tuple, aux[["test_cas", "latin_name", "duration_h"]].astype(str).to_numpy()))
    aux = cut(aux, "same_cell_as_sealed_row", [k in cells for k in key])

    if track == "T6":
        aux = cut(aux, "outside_24_96h_for_T6",
                  ~aux["duration_h"].between(*DURATION_WINDOW).fillna(False))

    rep["admissible_rows"] = int(len(aux))
    rep["admissible_chemicals"] = int(aux["test_cas"].nunique())
    rep["admissible_species"] = int(aux["latin_name"].nunique())
    rep["max_duration_h"] = (float(aux["duration_h"].max()) if aux["duration_h"].notna().any()
                             else None)
    # ---- the assertions the ruling requires
    if len(aux):
        u = aux["test_cas"].astype(str).map(cas_unit)
        if u.isin(held_units).any():
            raise ArchV2Error("held-out chemistry survived the auxiliary filter (AUX-LEAK)")
        k = list(map(tuple, aux[["test_cas", "latin_name", "duration_h"]].astype(str).to_numpy()))
        if any(x in cells for x in k):
            raise ArchV2Error("an auxiliary row shares a cell with a sealed row (AUX-LEAK)")
        if track == "T6" and not aux["duration_h"].between(*DURATION_WINDOW).all():
            raise ArchV2Error("an out-of-window auxiliary row survived on T6 (AUX-LEAK)")
    return aux.drop(columns=["log10_conc_mol_L"], errors="ignore"), rep
