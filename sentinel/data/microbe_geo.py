# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GEO-8 target tables and the qPCR ("QMP") input block for SENTINEL-Microbe.

Specification: ``docs/preregistration/microbe_arch_v2_amendment.md`` §5 (QMP block), §7.1 (targets, transforms,
floors, censoring) and §7.2 (join, split, tables). Built by ``scripts/data/build_microbe_geo.py``.

Targets
-------
The join is the builder's own visit key: :func:`sentinel.data.microbe_wq.collapse_visits` on the event's
``chem_parent`` (the registered ±24 h fallback is already encoded in ``chem_match``). Censoring follows protocol §2.2:
the published value when present, otherwise the record's detection limit (largest over the visit's records, as in
``match_chemistry``), otherwise the floor; log10 targets are floored at the transform. ``<target>_censored`` carries
the flag and ``<target>_imputed`` marks rows without a published value.

Tables (``processed/microbe_wq_geo``)
-------------------------------------
* ``dev_targets.parquet`` — development events and the ``.001`` 2017–18 training events;
* ``protected_labels.parquet`` — 2024 holdout and pre-2017 reserve events. Read only by the scorer under an open
  single-look claim (:func:`load_protected_labels`);
* ``ids.json`` — event ids per block (ids are not labels), used by :func:`load_dev_targets` to refuse protected ids;
* ``counts.json`` — counts per block and variable in the layout of the evidence file of Appendix A.1;
* ``qmp_features.parquet`` — the QMP input block for every event (a community-side input, never chemistry);
* ``qmp_meta.parquet`` — laboratory and batch per event, for the label-free confound check only (never inputs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.data.microbe_wq import collapse_visits, visit_key

DATASET = "microbe_wq_geo"

# target -> (chem_event_wide column, transform, floor); amendment §7.1
GEO = {
    "ph": ("lab_ph", "identity", None),
    "log_cl": ("cl_mgl", "log10", 0.003),
    "log_so4": ("so4_mgl", "log10", 0.006),
    "log_ca": ("ca_mgl", "log10", 0.001),
    "log_dic": ("dic_mgl", "log10", 0.025),
    "do_sat": ("field_do_sat_pct", "identity", None),
    "log_nh4": ("nh4_n_mgl", "log10", 0.002),
    "log_uv254": ("uv_absorbance_254_nm_au", "log10", 0.001),
}
GEO_TARGETS = tuple(GEO)
AUXILIARY = {"field_temp_c": ("field_temp_c", "identity", None)}          # unscored dimension of the master vector
DESCRIPTIVE = {"log_field_sc": ("field_spcond_uscm", "log10", 1.0),      # alternate label source, no inference
               "field_do": ("field_do_mgl", "identity", None)}
ALL_SPECS = {**GEO, **AUXILIARY, **DESCRIPTIVE}

BLOCKS = ("dev", "train1718", "holdout", "reserve")
PROTECTED_BLOCKS = ("holdout", "reserve")
EVIDENCE_BLOCK = {"dev": "002/dev", "holdout": "002/holdout", "train1718": "001/train1718", "reserve": "001/reserve"}

QMP_FEATURES = ("qmp_log_copies", "qmp_log_dna", "qmp_log_volume", "qmp_load")
QMP_FLAGS = tuple(f"{c}_missing" for c in QMP_FEATURES)


class ProtectedRowError(RuntimeError):
    """A protected (holdout / reserve) event was requested from the development target table."""


def event_blocks(release: str, dates, holdout_from: str = "2024-01-01", reserve_before: str = "2017-01-01"):
    """Block of each event: ``.002`` → dev / holdout (protocol §2.3); ``.001`` → train1718 / reserve (review C1)."""
    d = pd.to_datetime(pd.Series(np.asarray(dates)))
    if release == "002":
        return np.where(d < pd.Timestamp(holdout_from), "dev", "holdout")
    if release == "001":
        return np.where(d < pd.Timestamp(reserve_before), "reserve", "train1718")
    raise ValueError(release)


def detection_limits(chem_long: pd.DataFrame, cols) -> dict:
    """var -> Series(visit key -> largest detection limit of the visit's records); as ``match_chemistry``."""
    cl = chem_long.dropna(subset=["detection_limit"])
    cl = cl[cl["var"].isin(list(cols))]
    cl = cl.assign(visit=visit_key(cl["parentSampleID"]))
    return {v: g.groupby("visit")["detection_limit"].max() for v, g in cl.groupby("var")}


def build_targets(samples: pd.DataFrame, chem_wide: pd.DataFrame, chem_long: pd.DataFrame | None,
                  specs: dict | None = None) -> pd.DataFrame:
    """One row per event of ``samples`` (needs ``sample_id``, ``chem_parent``): transformed targets, censoring and
    imputation flags. Non-provisional chemistry records only."""
    specs = specs or ALL_SPECS
    w = chem_wide
    if "provisional" in w:
        w = w[~w["provisional"].fillna(False).astype(bool)]
    w = w.assign(time_utc=pd.to_datetime(w["time_utc"], utc=True))
    cols = sorted({c for c, _, _ in specs.values()})
    missing = [c for c in cols if c not in w]
    if missing:
        raise KeyError(f"chemistry table lacks {missing}")
    c = collapse_visits(w, cols).set_index("parentSampleID")
    dl = detection_limits(chem_long, cols) if chem_long is not None and "detection_limit" in chem_long else {}
    parent = samples["chem_parent"]
    out = pd.DataFrame({"sample_id": samples["sample_id"].to_numpy()})
    for t, (col, how, floor) in specs.items():
        val = parent.map(c[col]).astype(float).to_numpy()
        cen_col = f"{col}__censored"
        cen = (parent.map(c[cen_col]).eq(True).to_numpy(bool) if cen_col in c
               else np.zeros(len(samples), bool))
        imputed = cen & ~np.isfinite(val)
        if imputed.any():
            lim = parent.map(dl[col]).astype(float).to_numpy() if col in dl else np.full(len(samples), np.nan)
            val = np.where(imputed, lim, val)                                  # the record's detection limit
            if floor is not None:
                val = np.where(imputed & ~np.isfinite(val), floor, val)        # then the floor
        if how == "log10":
            val = np.log10(np.maximum(val, floor))
        elif how != "identity":
            raise ValueError(how)
        out[t] = val
        out[f"{t}_censored"] = cen
        out[f"{t}_imputed"] = imputed
    return out


def target_counts(tab: pd.DataFrame, meta: pd.DataFrame, specs: dict | None = None, keys_only: bool = False) -> dict:
    """Counts per variable in the layout of ``microbe_arch_v2_evidence/geo_counts.json`` (keys = source columns).
    ``meta`` is aligned with ``tab`` and has ``site_id``, ``domain_id``, ``chem_match``, ``chem_parent``.

    ``keys_only`` (lead ruling I-04, 2026-11-10; mandatory for the protected blocks): row and key counts only —
    events, events with a chemistry parent, sites, domains. No label column is read, so no per-variable,
    censoring or missing-label count of a protected block is produced."""
    keys = {"n_events": int(len(tab)), "n_with_chem_parent": int(meta["chem_parent"].notna().sum()),
            "sites": int(meta["site_id"].nunique()), "domains": int(meta["domain_id"].nunique())}
    if keys_only:
        return keys
    specs = specs or ALL_SPECS
    rows = {}
    for t, (col, _, _) in specs.items():
        usable = tab[t].notna().to_numpy()
        cen = tab[f"{t}_censored"].to_numpy(bool)
        imp = tab[f"{t}_imputed"].to_numpy(bool)
        has = usable & ~imp
        rows[col] = {"n": int(has.sum()), "n_cens": int(cen.sum()), "n_cens_noval": int((cen & ~has).sum()),
                     "n_cens_val": int((cen & has).sum()), "n_usable": int(usable.sum()),
                     "sites": int(meta.loc[has, "site_id"].nunique()),
                     "domains": int(meta.loc[has, "domain_id"].nunique()),
                     "n_time_match": int((has & (meta["chem_match"] == "time").to_numpy()).sum())}
    return {**keys, "vars": rows}


def build_qmp(samples: pd.DataFrame, qpcr: pd.DataFrame, field_genetic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, meta) per event of ``samples`` (needs ``sample_id``, ``dna_sample_ids`` joined by ``|``).

    16S "bacteria and archaea" records of DP1.20278.001, non-provisional, joined by ``dnaSampleID``; the filtered
    volume by ``geneticSampleID`` (= ``dnaSampleID`` without the trailing ``-DNA<n>``). Each feature is the mean of
    the log10 values over the event's samples; the load index is copies + DNA concentration − volume (log copies per
    mL up to the elution volume, assumed constant)."""
    q = qpcr[(qpcr["targetGene"] == "16S") & ~qpcr["provisional"].fillna(False).astype(bool)]
    if "targetTaxonGroup" in q:
        q = q[q["targetTaxonGroup"].astype(str).str.lower() == "bacteria and archaea"]

    def _log(x):
        x = pd.to_numeric(x, errors="coerce").astype(float)
        return np.log10(x.where(x > 0))

    q = q.assign(_copies=_log(q["meanCopyNumber"]), _dna=_log(q["nucleicAcidConcentration"]),
                 dnaSampleID=q["dnaSampleID"].astype(str))
    by_dna = q.groupby("dnaSampleID")[["_copies", "_dna"]].mean()
    first = q.sort_values("dnaSampleID", kind="stable").groupby("dnaSampleID")[["laboratoryName", "batchID"]].first()
    fg = field_genetic.assign(_vol=_log(field_genetic["geneticFilteredSampleVolume"]),
                              geneticSampleID=field_genetic["geneticSampleID"].astype(str))
    by_gen = fg.groupby("geneticSampleID")["_vol"].mean()

    rows, meta = [], []
    for sid, ids in zip(samples["sample_id"], samples["dna_sample_ids"].fillna("").astype(str)):
        dna_ids = [x for x in ids.split("|") if x]
        a = by_dna.reindex(dna_ids)
        v = by_gen.reindex([x.rsplit("-", 1)[0] for x in dna_ids])
        copies, conc, vol = (float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
                             for x in (a["_copies"].to_numpy(float), a["_dna"].to_numpy(float), v.to_numpy(float)))
        rows.append({"sample_id": sid, "qmp_log_copies": copies, "qmp_log_dna": conc, "qmp_log_volume": vol,
                     "qmp_load": copies + conc - vol})
        m = first.reindex(dna_ids).dropna(how="all")
        meta.append({"sample_id": sid, "n_dna_samples": len(dna_ids), "n_qpcr_records": int(a["_copies"].notna().sum()),
                     "laboratory": str(m["laboratoryName"].iloc[0]) if len(m) else None,
                     "batch": str(m["batchID"].iloc[0]) if len(m) else None})
    feat = pd.DataFrame(rows)
    for c in QMP_FEATURES:
        feat[f"{c}_missing"] = feat[c].isna()
    return feat, pd.DataFrame(meta)


# ------------------------------------------------------------------------------------------------ loaders

def dataset_dir(root: str | Path | None = None) -> Path:
    return Path(root) if root else paths.PROCESSED / DATASET


def load_ids(root: str | Path | None = None) -> dict:
    return json.loads((dataset_dir(root) / "ids.json").read_text())


def load_dev_targets(ids=None, root: str | Path | None = None) -> pd.DataFrame:
    """Development target table (development and ``.001`` 2017–18 events), indexed by ``sample_id``.

    With ``ids``: the rows of those events, in that order. A protected id raises :class:`ProtectedRowError`; an id
    of no block raises ``KeyError``. The protected label file is never opened here."""
    d = dataset_dir(root)
    tab = pd.read_parquet(d / "dev_targets.parquet").set_index("sample_id")
    blocks = load_ids(d)
    protected = set(blocks["holdout"]) | set(blocks["reserve"])
    leaked = protected & set(map(str, tab.index))
    if leaked:
        raise ProtectedRowError(f"{len(leaked)} protected events are in the development target table")
    if ids is None:
        return tab
    ids = pd.Index(np.asarray(ids)).astype(str)
    bad = [i for i in ids if i in protected]
    if bad:
        raise ProtectedRowError(f"{len(bad)} protected (holdout / reserve) events requested from the development "
                                f"target table, e.g. {bad[0]}")
    unknown = ids.difference(tab.index)
    if len(unknown):
        raise KeyError(f"{len(unknown)} events are in no development block, e.g. {unknown[0]}")
    return tab.loc[ids]


def load_protected_labels(root: str | Path | None = None) -> pd.DataFrame:
    """Holdout and reserve labels. Only under an open single-look claim of the lead-written freeze.

    Lead ruling I-03 (2026-11-10): the claim is required for every root that holds real data, however the root
    was passed. Only a declared synthetic test root (``testlook.is_synthetic_root``) is exempt."""
    from sentinel.models.microbe.testlook import require_claim_for_root

    require_claim_for_root(root, "protected_labels.parquet")
    return pd.read_parquet(dataset_dir(root) / "protected_labels.parquet").set_index("sample_id")


def load_qmp(ids=None, root: str | Path | None = None) -> pd.DataFrame:
    """QMP input block (features and missing flags only), indexed by ``sample_id``."""
    f = pd.read_parquet(dataset_dir(root) / "qmp_features.parquet").set_index("sample_id")
    f = f[list(QMP_FEATURES) + list(QMP_FLAGS)]
    return f if ids is None else f.reindex(pd.Index(np.asarray(ids)).astype(str))
