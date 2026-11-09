# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Hook for upstream-model features (M1 Sonde / M2 Orbit / M6 HydroMet predictions or embeddings).

RESEARCH_PLAN §8 cross-fitting rule: upstream predictions used as features in a downstream fold must
come from upstream models that never saw that fold's sites or times. Phase 1 uses **no** upstream
features; this module defines the contract that a later provider must satisfy, and the checks that a
downstream driver runs before it may join such features.

A provider is any object with

* ``name`` (str) and ``model_registry_id`` (registry entry of the upstream run),
* ``training_sites`` (set of upstream site keys the upstream model was fit on, incl. tuning data),
* ``training_end`` (pd.Timestamp: last time stamp of upstream training/tuning data),
* ``predict(units: pd.DataFrame) -> pd.DataFrame`` returning features indexed like ``units`` with an
  ``issued_at`` column (when the upstream value would have been available).

:func:`check_crossfit` refuses the join if a downstream evaluation unit maps to an upstream training
site, if the upstream training period overlaps the downstream evaluation period, or if any feature
was issued after the downstream decision time.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class CrossFitError(RuntimeError):
    pass


class UpstreamProvider(Protocol):
    name: str
    model_registry_id: str
    training_sites: set
    training_end: pd.Timestamp

    def predict(self, units: pd.DataFrame) -> pd.DataFrame: ...


def check_crossfit(provider: UpstreamProvider, eval_units: pd.DataFrame, *, site_col: str = "upstream_site_key",
                   decision_col: str = "decision_time", features: pd.DataFrame | None = None) -> dict:
    """Validate that ``provider`` may supply features for the downstream evaluation rows ``eval_units``.

    ``eval_units[site_col]`` = upstream site(s) whose predictions feed the row (nearest/connected site);
    ``eval_units[decision_col]`` = downstream decision time.
    """
    sites = set(eval_units[site_col].dropna().astype(str))
    leak_sites = sites & {str(s) for s in provider.training_sites}
    if leak_sites:
        raise CrossFitError(f"{provider.name}: {len(leak_sites)} downstream evaluation rows use upstream "
                            f"training sites, e.g. {sorted(leak_sites)[:5]}")
    t0 = pd.to_datetime(eval_units[decision_col]).min()
    if pd.Timestamp(provider.training_end) >= t0:
        raise CrossFitError(f"{provider.name}: upstream training ends {provider.training_end} >= first "
                            f"downstream evaluation decision time {t0}")
    if features is not None:
        late = pd.to_datetime(features["issued_at"]) > pd.to_datetime(eval_units[decision_col])
        if late.any():
            raise CrossFitError(f"{provider.name}: {int(late.sum())} features issued after decision time")
    return {"provider": provider.name, "registry_id": provider.model_registry_id, "n_units": int(len(eval_units)),
            "n_upstream_sites": len(sites), "training_end": str(provider.training_end)}


def phase1_no_upstream(feature_columns) -> None:
    """Phase-1 guard: refuse any feature column that looks like an upstream-model output."""
    bad = [c for c in feature_columns if str(c).startswith(("up_", "m1_", "m2_", "m6_", "emb_"))]
    if bad:
        raise CrossFitError(f"phase 1 uses observed covariates only; upstream-looking columns {bad[:5]}")
