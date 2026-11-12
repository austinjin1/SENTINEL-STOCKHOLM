# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The SENTINEL-Microbe modelling design and its development / holdout guard (protocol §2–3).

The joined dataset ``processed/microbe_wq`` (built by ``scripts/data/build_microbe_wq.py`` from the
NEON products) has three Parquet files:

* ``samples.parquet`` — one row per 16S community *event* (site x collection date x primer set;
  field replicates summed): ``sample_id``, ``site_id``, ``domain_id``, ``aquatic_type``,
  ``latitude``, ``longitude``, ``elevation``, ``collect_date``, ``primer_set``, ``n_reads``,
  ``n_seqs`` and, per target, the raw matched value (``raw_<target>``), its unit and match
  diagnostics; the modelled (transformed) targets are added here by :func:`add_targets`.
* ``tokens.parquet`` — long table ``sample_id``, ``seq_id``, ``count``, taxonomy ranks.
* ``sequences.parquet`` — ``seq_id`` (sha1 of the sequence), ``sequence``, ``length``.

:func:`load_design` returns **development rows only** unless ``allow_holdout=True``; the only
caller allowed to pass that flag is the single-pass final scoring script (protocol §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths

DATASET = "microbe_wq"

# target -> (raw column, transform, floor). Units are fixed by the builder (see datasheet).
TARGETS = {
    "log_tn": ("raw_tn_mgl", "log10", 0.005),          # total nitrogen, mg/L (grab, lab)
    "log_tp": ("raw_tp_mgl", "log10", 0.0005),         # total phosphorus, mg/L (grab, lab)
    "log_doc": ("raw_doc_mgl", "log10", 0.05),         # dissolved organic carbon, mg/L (grab, lab)
    "log_no3": ("raw_no3_mgl", "log10", 0.001),        # nitrate+nitrite as N, mg/L (grab, lab)
    "log_sc": ("raw_sc_uscm", "log10", 1.0),           # specific conductance, uS/cm (sonde daily mean)
    "log_turb": ("raw_turb_fnu", "log10", 0.1),        # turbidity, FNU (sonde daily mean)
    "do": ("raw_do_mgl", "identity", None),            # dissolved oxygen, mg/L (sonde daily mean)
    "log_chla": ("raw_chla_ugl", "log10", 0.05),       # chlorophyll-a, ug/L (sonde daily mean)
}
TARGET_NAMES = tuple(TARGETS)


@dataclass
class Design:
    samples: pd.DataFrame       # index sample_id
    tokens: pd.DataFrame
    sequences: pd.DataFrame
    holdout_rule: dict
    includes_holdout: bool

    @property
    def Y(self) -> pd.DataFrame:
        return self.samples[list(TARGET_NAMES)]


def add_targets(samples: pd.DataFrame) -> pd.DataFrame:
    s = samples.copy()
    for t, (raw, how, floor) in TARGETS.items():
        v = s[raw].astype(float) if raw in s else pd.Series(np.nan, index=s.index)
        if how == "log10":
            v = np.log10(np.maximum(v, floor))
        s[t] = v
    return s


def holdout_mask(samples: pd.DataFrame, rule: dict) -> np.ndarray:
    """Rows reserved for the final test. Rule kinds: ``date_from`` (collect_date >= date)."""
    if rule["kind"] == "date_from":
        return (pd.to_datetime(samples["collect_date"]) >= pd.Timestamp(rule["date"])).to_numpy()
    raise ValueError(f"unknown holdout rule {rule}")


def load_holdout_rule(path: str | Path | None = None) -> dict:
    import yaml

    p = Path(path) if path else paths.REPO_ROOT / "configs" / "microbe" / "common.yaml"
    return dict(yaml.safe_load(p.read_text())["holdout"])


def load_release_holdout_rule(path: str | Path | None = None) -> dict:
    import yaml

    p = Path(path) if path else paths.REPO_ROOT / "configs" / "microbe" / "common.yaml"
    rr = dict(yaml.safe_load(p.read_text())["release_holdout"])
    if rr["kind"] != "date_before":
        raise ValueError(rr)
    return rr


def load_design(*, allow_holdout: bool = False, root: str | Path | None = None,
                holdout_rule: dict | None = None, verify_frozen: bool = True,
                with_release_v1: bool = False) -> Design:
    """Development design (default). ``with_release_v1`` appends the DP1.20141.001 events
    (``source_product``) for the cross-release scheme only."""
    default_root = root is None
    if allow_holdout:
        # real holdout rows (2024 test, .001 pre-2017 reserve): only under an open single-look claim of the
        # lead-written freeze (sentinel.models.microbe.testlook); access closes when predictions are sealed.
        # Lead ruling I-03 (2026-11-10): the claim is required for EVERY root that holds real data, however the
        # root was passed; only a declared synthetic test root is exempt.
        from sentinel.models.microbe.testlook import require_claim_for_root

        require_claim_for_root(root, "samples.parquet")
    root = Path(root) if root else paths.PROCESSED / DATASET
    samples = pd.read_parquet(root / "samples.parquet")
    tokens = pd.read_parquet(root / "tokens.parquet")
    seqs = pd.read_parquet(root / "sequences.parquet")
    rule = holdout_rule or load_holdout_rule()
    samples = add_targets(samples).set_index("sample_id", drop=False)
    samples.index.name = "sample_id"
    if not allow_holdout:
        samples = samples[~holdout_mask(samples, rule)]
        frozen = paths.REPO_ROOT / "docs" / "preregistration" / "microbe_split.json"
        if verify_frozen and default_root and frozen.exists():
            import json

            dev_ids = set(json.loads(frozen.read_text())["development_ids"])
            if set(map(str, samples.index)) != dev_ids:
                raise RuntimeError("development rows differ from the frozen split "
                                   "(docs/preregistration/microbe_split.json) — data or rule changed")
    tokens = tokens[tokens["sample_id"].isin(samples.index)].reset_index(drop=True)
    seqs = seqs[seqs["seq_id"].isin(tokens["seq_id"].unique())].reset_index(drop=True)
    if with_release_v1:
        v1 = load_design(allow_holdout=allow_holdout, root=root.parent / (DATASET + "_v1"), holdout_rule=rule,
                         verify_frozen=False)
        if not allow_holdout:   # cross-release final-test reserve (common.yaml release_holdout; review C1)
            rr = load_release_holdout_rule()
            keep = pd.to_datetime(v1.samples["collect_date"]) >= pd.Timestamp(rr["date"])
            v1.samples = v1.samples[keep.to_numpy()]
            v1.tokens = v1.tokens[v1.tokens["sample_id"].isin(v1.samples.index)]
        if set(v1.samples.index) & set(samples.index):
            raise RuntimeError("sample ids collide between releases")
        samples = pd.concat([samples, v1.samples])
        tokens = pd.concat([tokens, v1.tokens], ignore_index=True)
        seqs = pd.concat([seqs, v1.sequences], ignore_index=True).drop_duplicates("seq_id")
    return Design(samples, tokens, seqs, rule, allow_holdout)
