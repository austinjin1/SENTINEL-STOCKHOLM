"""The single dispatcher for every declared arch-v2 arm.

New module. ``arch_v2_arms.py`` is one of the four files the lead re-hashes before writing the pins
file, so it is **not edited**: its ``build_arm`` keeps raising ``ArmNotImplemented`` for the learned
arms, and this module is the entry point a runner uses. :func:`build` handles the learned arms and
delegates everything else to ``arch_v2_arms.build_arm``.

:func:`coverage` is the drift check: every entry declared in ``configs/chemtox/arch_v2.yaml`` must
be either buildable here or named in :data:`NOT_BUILDABLE` with a reason. A test asserts the set
difference is empty, so an entry cannot be silently missing from both.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from sentinel.models.chemtox import arch_v2_arms as ARMS

__all__ = ["BuildError", "NOT_BUILDABLE", "LEARNED", "build", "coverage"]


class BuildError(ValueError):
    """An arm could not be built, or is not buildable by design."""


#: Entries that are deliberately not constructed by this dispatcher, with the reason.
NOT_BUILDABLE = {
    "r5_trident_released_errors":
        "no fit: a read of TRIDENT's released per-SMILES 10x10-CV errors, restricted to pool "
        "chemicals until the block's FREEZE exists (R5-NOT-PAIRABLE)",
    "rf_published_adore":
        "the published ADORE configuration, verbatim, built by the existing published-baseline "
        "code path; not re-parameterised here",
    "xgboost_published_adore": "as rf_published_adore",
    "gpboost_published":
        "the two verbatim Zubrod et al. parameter sets from the manifest "
        "chemtox_published_gpboost; gpboost_model.py is NOT edited (S4)",
    "readacross":
        "the existing readacross.py chemical-similarity kNN; shared code, not edited",
    "ensemble_mean": "a post-hoc combination of frozen winners (arch_v2_ensemble.mean_ensemble)",
    "ensemble_decorrelated": "as ensemble_mean (arch_v2_ensemble.select_decorrelated)",
    "ensemble_mean_baseline": "as ensemble_mean, over the strongest simple baselines",
    "ensemble_decorrelated_baseline": "as ensemble_decorrelated, over the simple baselines",
}

#: Learned arms this module constructs.
LEARNED = ("ted", "timehead", "mmoe", "rx_chemprop", "rx_lightgbm", "rx_catboost",
           "rx_lightgbm_dur", "xgboost_emb", "xgboost_emb_dur", "rf_emb", "rf_emb_dur",
           "lightgbm_noemb")


def _rx_gbm(name: str, feature_columns: Sequence[str], **params):
    from sentinel.models.chemtox import arch_v2_rx as RX

    presets = {
        "rx_lightgbm": dict(family="lightgbm", with_embedding=True, with_duration=False),
        "rx_catboost": dict(family="catboost", with_embedding=True, with_duration=False),
        "rx_lightgbm_dur": dict(family="lightgbm", with_embedding=True, with_duration=True),
        "lightgbm_noemb": dict(family="lightgbm", with_embedding=False, with_duration=False),
    }
    cfg = RX.RxGbmConfig(**{**presets[name], **params})
    return RX.build_rx_gbm(cfg, feature_columns)


def _sk_emb(name: str, feature_columns: Sequence[str], **params):
    """``rf_emb`` / ``xgboost_emb`` and their F10 ``*_dur`` variants.

    A ``*_dur`` entry is its parent's configuration with ``duration_h`` added and nothing else, so
    it is built from the parent's parameters and the parent's column list put through
    ``arch_v2_arms.duration_variant`` (F10).
    """
    from sentinel.models.chemtox import arch_v2_embed as E

    cols = list(feature_columns)
    if name.endswith("_dur"):
        if "duration_h" not in cols:
            cols = ARMS.duration_variant(cols)
    elif "duration_h" in cols:
        raise BuildError(f"{name} is the non-duration parent but duration_h is present (F10)")
    if not any(c.startswith(E.EMB_PREFIX) for c in cols):
        raise BuildError(f"{name} is an R4 parity arm and needs the concatenated embedding block")
    if name.startswith("rf_"):
        from sklearn.ensemble import RandomForestRegressor
        defaults = {"n_estimators": 300, "n_jobs": 6, "random_state": 0}
        model = RandomForestRegressor(**{**defaults, **params})
    else:
        from xgboost import XGBRegressor
        defaults = {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
                    "tree_method": "hist", "nthread": 6, "random_state": 0}
        model = XGBRegressor(**{**defaults, **params})
    model._sentinel_feature_columns = cols
    return model


def build(name: str, *, feature_columns: Sequence[str] | None = None, **params):
    """Build any declared arm, or raise with the reason it is not buildable here."""
    if name in NOT_BUILDABLE:
        raise BuildError(f"{name} is not built by this dispatcher: {NOT_BUILDABLE[name]}")
    if name == "ted":
        from sentinel.models.chemtox import arch_v2_ted as TED
        if feature_columns is None:
            raise BuildError("ted needs feature_columns")
        cfg = TED.TedConfig(n_features=len(list(feature_columns)), **params)
        TED.assert_no_time_in_trunk(feature_columns, allow=cfg.leak_time_into_trunk)
        return TED.TED(cfg)
    if name == "timehead":
        from sentinel.models.chemtox import arch_v2_timehead as TH
        if feature_columns is None:
            raise BuildError("timehead needs feature_columns")
        return TH.TimeHead(TH.TimeHeadConfig(n_features=len(list(feature_columns)), **params))
    if name == "mmoe":
        from sentinel.models.chemtox import arch_v2_mmoe as MM
        if feature_columns is None:
            raise BuildError("mmoe needs feature_columns")
        return MM.MMoE(MM.MMoEConfig(n_features=len(list(feature_columns)), **params))
    if name == "rx_chemprop":
        from sentinel.models.chemtox import arch_v2_rx as RX
        return RX.rx_chemprop_params(RX.RxChempropConfig(**params))
    if name in ("rx_lightgbm", "rx_catboost", "rx_lightgbm_dur", "lightgbm_noemb"):
        if feature_columns is None:
            raise BuildError(f"{name} needs feature_columns")
        return _rx_gbm(name, feature_columns, **params)
    if name in ("rf_emb", "rf_emb_dur", "xgboost_emb", "xgboost_emb_dur"):
        if feature_columns is None:
            raise BuildError(f"{name} needs feature_columns")
        return _sk_emb(name, feature_columns, **params)
    return ARMS.build_arm(name, feature_columns=feature_columns, **params)


def coverage(cfg: Mapping | None = None) -> dict:
    """Every declared entry is buildable here, delegated, or named in NOT_BUILDABLE."""
    cfg = cfg if cfg is not None else ARMS.load_config()
    declared: set[str] = set()
    for track in ("T1", "T6"):
        declared |= set(ARMS.entries(cfg, track))
    delegated = set(ARMS._FIXED_BUILDERS) | {"r1_monotone_gbm", "r2_aft_gbm",
                                             "r3_multitask_stack_gbm"}
    accounted = set(LEARNED) | set(NOT_BUILDABLE) | delegated
    missing = sorted(declared - accounted)
    return {"declared": len(declared), "learned": sorted(set(LEARNED) & declared),
            "delegated": sorted(delegated & declared),
            "not_buildable": sorted(set(NOT_BUILDABLE) & declared),
            "unaccounted": missing}
