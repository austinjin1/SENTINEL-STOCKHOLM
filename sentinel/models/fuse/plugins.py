# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Plug-and-play interface for modality encoders from other workstreams (M1–M7).

A modality workstream contributes a :class:`ModalityPlugin`. Fuse calls it with the rows of a tier
table and receives *cross-fitted* features: for every row, the producing model must not have seen
the row's site, its flow group / NEON domain, nor any Fuse test site (docs/preregistration/
fuse_protocol.md §5, the cross-fitting rule). The plugin declares its training sites so this can be
checked mechanically by :func:`check_crossfit`.

Contract
--------
``featurize(rows) -> pandas.DataFrame`` indexed like ``rows`` with
  * ``<name>__present`` (bool) — False where the encoder has no input for that row;
  * feature columns ``<name>__<group>__<feat>`` (grouped into tokens by ``<group>``) or ``<name>__<feat>``;
    values are predictions (e.g. ``<name>__do__mu``, ``<name>__do__sigma``) or embeddings
    (``<name>__emb__e0`` …), NaN where absent;
  * optionally ``<name>__age`` (days between the encoder's input time and the issue time; >= 0).
``training_sites`` — every site whose data the producing model(s) were fit on;
``issued_before_decision`` — True if all inputs predate the row's issue time (the plugin asserts it);
``fold_aware`` — True if the plugin can refit per Fuse development fold (``featurize(rows, exclude_sites=…)``).

The Fuse side (:func:`attach_plugin`) adds the columns to the table and a new
:class:`~sentinel.models.fuse.data.ModalitySpec`; :meth:`FuseModel.add_modality` adds the
tokenizer and modality-specific cross-attention, with the core frozen. When the plugin's modality
is absent for a row, that row's prediction is bit-identical to the model without the plugin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .data import ModalitySpec, group_columns


class CrossFitError(AssertionError):
    pass


@runtime_checkable
class ModalityPlugin(Protocol):
    name: str
    model_registry_id: str | None
    training_sites: set
    fold_aware: bool

    def featurize(self, rows: pd.DataFrame, exclude_sites: set | None = None) -> pd.DataFrame: ...


@dataclass
class FramePlugin:
    """A plugin backed by a precomputed table of cross-fitted outputs (e.g. M3 LODO OOF predictions).

    ``table`` must have a key column ``key`` matching ``rows[row_key]`` and feature columns named per
    the contract; ``fold_of_site`` (optional) records which producer fold predicted each site so the
    check can confirm "never saw this site".
    """
    name: str
    table: pd.DataFrame
    key: str
    row_key: str
    training_sites: set = field(default_factory=set)
    site_col: str = "site"
    model_registry_id: str | None = None
    fold_aware: bool = False
    excluded_sites_of_row: Callable[[pd.Series], set] | None = None

    def featurize(self, rows: pd.DataFrame, exclude_sites: set | None = None) -> pd.DataFrame:
        t = self.table.set_index(self.key)
        cols = [c for c in t.columns if c.startswith(f"{self.name}__")]
        out = t[cols].reindex(rows[self.row_key].to_numpy())
        out.index = rows.index
        pc = f"{self.name}__present"
        if pc not in out:
            out[pc] = out[[c for c in cols if c != pc]].notna().any(axis=1)
        out[pc] = out[pc].fillna(False).astype(bool)
        return out


def check_crossfit(plugin: ModalityPlugin, rows: pd.DataFrame, *, site_col: str = "site",
                   test_sites: set | None = None, per_row_exclusions: pd.Series | None = None) -> dict:
    """Mechanical cross-fitting check (raises :class:`CrossFitError`).

    1. No Fuse test site may be in ``plugin.training_sites`` unless the plugin provides per-row
       exclusions proving the producing model for that row excluded it.
    2. For each row, its own site must not be among the sites the producing model saw: either the
       site is not in ``training_sites`` at all, or ``per_row_exclusions[row]`` (set of sites excluded
       from the model that produced the row) contains it.
    """
    ts = set(test_sites or ())
    trn = set(map(str, plugin.training_sites))
    leak_test = ts & trn
    sites = rows[site_col].astype(str)
    bad = []
    for i, s in sites.items():
        excl = per_row_exclusions.loc[i] if per_row_exclusions is not None else set()
        if s in trn and s not in excl:
            bad.append(s)
        if leak_test and not leak_test <= set(excl):
            raise CrossFitError(f"{plugin.name}: producer trained on Fuse test sites {sorted(leak_test)[:5]}")
    if bad:
        raise CrossFitError(f"{plugin.name}: {len(bad)} rows predicted by a model that saw their site, "
                            f"e.g. {sorted(set(bad))[:5]}")
    return {"plugin": plugin.name, "rows": int(len(rows)), "training_sites": len(trn), "ok": True}


def attach_plugin(df: pd.DataFrame, plugin: ModalityPlugin, *, site_col: str = "site", test_sites: set | None = None,
                  per_row_exclusions: pd.Series | None = None, chunk: int = 6) -> tuple[pd.DataFrame, ModalitySpec, dict]:
    """Verify cross-fitting, join the plugin features, and return the new modality spec."""
    rep = check_crossfit(plugin, df, site_col=site_col, test_sites=test_sites, per_row_exclusions=per_row_exclusions)
    feats = plugin.featurize(df)
    name = plugin.name
    pc = f"{name}__present"
    if pc not in feats:
        raise ValueError(f"plugin {name} must return {pc}")
    clash = [c for c in feats.columns if c in df.columns]
    if clash:
        raise ValueError(f"plugin columns already in table: {clash[:5]}")
    out = pd.concat([df, feats], axis=1)
    cols = [c for c in feats.columns if c not in (pc, f"{name}__age")]
    g, n = group_columns(name, cols, chunk=chunk)
    spec = ModalitySpec(name, cols, g, n, pc, f"{name}__age" if f"{name}__age" in feats else None)
    rep["present_rows"] = int(out[pc].sum())
    return out, spec, rep


def acceptance_test(delta_by_target: dict[str, tuple[float, float, float]], non_inferiority: float = 0.01) -> dict:
    """Plug-and-play acceptance (RESEARCH_PLAN §6): paired blocked ΔCRPS/ΔAUPRC (improvement > 0)
    with CI excluding 0 for >= 1 target, and no target worse than ``non_inferiority`` (relative
    CRPS increase / absolute AUPRC drop, upper CI bound). ``delta_by_target[t] = (est, lo, hi)`` with
    positive = improvement (relative, e.g. 1 − CRPS_new/CRPS_old)."""
    gains = [t for t, (e, lo, hi) in delta_by_target.items() if lo > 0]
    worse = [t for t, (e, lo, hi) in delta_by_target.items() if lo < -non_inferiority]
    return {"accepted": bool(gains) and not worse, "improved_targets": gains, "non_inferiority_failures": worse,
            "margin": non_inferiority}
