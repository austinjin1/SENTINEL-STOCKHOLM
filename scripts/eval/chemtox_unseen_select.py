#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stage-2 selection for the ChemTox **unseen-taxa** track (validation folds only).

Implements §4 + §1.4.4 of ``docs/preregistration/chemtox_unseen_taxa_plan.md``:

* folds come from ``unseen_taxa.validation_splits`` — a leave-one-class-out rotation over the four
  pool classes with >= 1,000 rows, with the held-out class's chemicals removed from the training
  side by CAS, InChIKey **and** ``parent_key``;
* every model is fitted through ``unseen_taxa.ChannelBuilder``, which wraps ``TaxonBlindBuilder``
  (trait drop, §1.1) and adds the two pre-declared cross-taxon channels (§1.4);
* ``channels`` is a feature dimension **inside** each family's 32-configuration budget, not extra
  budget, and the plain ADORE comparators get ``channels=none`` while the ``_chan`` variants sample
  it exactly as a candidate may;
* the selection score is the **mean RMSE over the four folds**, with the worse half reported beside
  it and ties broken by the worst fold; all four per-fold scores and the top-5 spread are kept.

**This script reads no test row.** It never imports ``unseen_taxa_guard.load_test_block`` and never
opens ``test_blocks.parquet``; the only parquet it reads is ``training_pool.parquet``. It writes no
FREEZE file: its output is a *candidate* summary for the lead to review.

    python -B scripts/eval/chemtox_unseen_select.py --entries xgboost rf --budget 32
    python -B scripts/eval/chemtox_unseen_select.py --entries xgboost --budget 2 --folds Maxillopoda --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# CPU budget (docs/CONTRIBUTING_V2.md + task rule): at most 6 threads per job. Thread pools read
# these when numpy / xgboost / lightgbm load, so they are set before those imports.
_THREADS = os.environ.get("SENTINEL_THREADS", "6")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.runlog import TrainingRun
from sentinel.models.chemtox import cycle as CYC
from sentinel.models.chemtox import spaces as SPACES
from sentinel.models.chemtox import unseen_taxa as U
from sentinel.models.chemtox import unseen_taxa_guard as G
from sentinel.models.chemtox.data import FeatureBuilder, FeatureSpec

log = logging.getLogger("chemtox_unseen_select")

EXPERIMENT = "chemtox_unseen_taxa_select"
POOL = paths.DATA_ROOT / "processed" / "chemtox_unseen_taxa" / "training_pool.parquet"
TARGET = "result_conc1_mean_mol_log"

#: Channel settings sampled as a feature dimension for a **candidate** (§1.4.4).
CHANNEL_CHOICES: tuple[tuple[str, ...], ...] = ((), ("surrogate",), ("higher_rank",),
                                                ("surrogate", "higher_rank"))

#: Channel settings for a **channel-matched comparator** (review C-A3-5). §1.4.4's
#: ``channels ∈ {none, surrogate, higher_rank, both}`` is written for the candidates; a `_chan`
#: comparator that selects ``()`` is just the channel-free comparator, which makes claim condition
#: (d2) vacuous — the C-M4P-1 failure mode §1.4.3 exists to prevent. So the empty set is excluded.
CHANNEL_CHOICES_MATCHED: tuple[tuple[str, ...], ...] = (
    ("surrogate",), ("higher_rank",), ("surrogate", "higher_rank"))

#: Published **fixed** hyper-parameters (review C-A3-4; plan §1.4.3 "the published configurations
#: likewise", §5.1.1). Taken verbatim from the phase-1 freeze that the closed prospective track
#: used for `rf_published` / `xgboost_published`
#: (`scripts/eval/chemtox_cycle_select.py:PUBLISHED_FALLBACK`), so this track's published
#: comparator is the same published configuration, not a re-tuned tree.
PUBLISHED_PARAMS: dict[str, dict] = {
    "rf": {"n_estimators": 300, "max_depth": 20, "max_features": 0.3},
    "xgboost": {"n_estimators": 800, "eta": 0.03, "max_depth": 6, "min_child_weight": 1,
                "subsample": 0.8},
}
#: features of the published configuration; `chemical` is pinned to the plan's `mw_logp` by
#: `entry_spec`, which is what makes these the `*_mwlogp` published variants §5.1.1 names.
PUBLISHED_FEATURES: dict = {"fingerprint": "MACCS", "published": True}

#: Published GPBoost hyper-parameters, **branch (1) of the lead's 2026-11-09 ruling**: taken
#: verbatim from the authors' own released fitted models in the Bio-QSARs 2.0 repository
#: (https://github.com/syngenta/bio-qsar), whose LightGBM/GPBoost `parameters:` echo block states
#: the complete set. Zubrod, Galic, Vaugeois & Dreier, *Environment International* 186:108607
#: (2024), DOI 10.1016/j.envint.2024.108607, PMID 38593686.
#:
#:   `Fish_without_DEB.json`         sha256 1b17db4c5f18abbd… (7,171,431 bytes)
#:   `Invertebrate_without_DEB.json` sha256 770ecd0d1ed31360… (3,369,787 bytes)
#:
#: The `_without_DEB` variants are the authors' **trait-free** models, which is the matching form
#: for this track (the trait block is dropped, §1.1). Both have 1,000 trees (highest `Tree=999`).
#: Two variants are kept because the authors released taxon-specific models differing in exactly
#: two values; per §5.1.1 the reference is the variant with the best validation score, fixed in the
#: hashed block before the pass. Nothing here is a choice of ours.
#:
#: Disclosed adaptations — THREE, not two (delta review A3.5-1); every one of them makes the
#: published comparator **weaker here than the authors' own model**, i.e. each biases in our
#: candidate's favour, so all three travel with every claim-condition-(b) comparison:
#:   1. grouping reduced to `rank_phylum` (the authors grouped species/genus/group; A3.0);
#:   2. gpboost 1.8.0 against the authors' 1.2.1;
#:   3. the authors' **species one-hot fixed effects are absent** (10 in the fish model, 13 in the
#:      invertebrate model) — §1.1 forbids them on this track. So `_without_DEB` means
#:      "DEB-trait-free", NOT "taxon-free": it is still the matching form for §1.1's trait drop,
#:      but the published estimator is not taxon-free and saying "trait-free" overstates the match.
#: Carried into `summary_index.json` and required beside every (b) comparison (A3.5-1).
PUBLISHED_GPBOOST_ADAPTATIONS: tuple[str, ...] = (
    "grouping reduced to rank_phylum only (authors: species/genus/group)",
    "gpboost 1.8.0 (authors: 1.2.1)",
    "authors' species one-hot fixed effects absent (10 fish / 13 invertebrate), forbidden by §1.1",
)
#: Selection breadth, reported beside (b) so the asymmetry is visible (delta review A3.5-1):
#: the published GPBoost comparator is given the best of 6 settings while `rf_published_chan` /
#: `xgboost_published_chan` get the best of 3. The direction is conservative — it strengthens the
#: bar a claim must clear — but it must be stated, not left implicit.
PUBLISHED_SELECTION_BREADTH: dict[str, int] = {
    "gpboost_published_chan": 6, "rf_published_chan": 3, "xgboost_published_chan": 3,
}

PUBLISHED_GPBOOST: dict[str, dict] = {
    "fish_form": {"objective": "regression", "num_boost_round": 1000, "learning_rate": 0.01,
                  "num_leaves": 64, "max_depth": -1, "min_data_in_leaf": 10, "lambda_l2": 1,
                  "feature_fraction": 1.0, "bagging_fraction": 1.0},
    "invertebrate_form": {"objective": "regression", "num_boost_round": 1000,
                          "learning_rate": 0.01, "num_leaves": 32, "max_depth": -1,
                          "min_data_in_leaf": 10, "lambda_l2": 10, "feature_fraction": 1.0,
                          "bagging_fraction": 1.0},
}

#: A track-local GPBoost space **without** `group_columns` (review C-A3-3). `cycle.GPBOOST_SPACE`
#: samples `group_columns` over forbidden below-class keys (`tax_gs`, `tax_genus`, `tax_family`);
#: leaving it in the sampled dict made the recorded configuration claim species/genus grouping for a
#: fit that actually used `rank_phylum`. Dropping it from the space means the audit trail can only
#: ever show the admitted key.
GPBOOST_SPACE_TRACK: dict = {
    "features": dict(CYC.GPBOOST_SPACE["features"]),
    "params": {k: v for k, v in CYC.GPBOOST_SPACE["params"].items() if k != "group_columns"},
}

# --------------------------------------------------------------------------------- model entries
#
# ``role`` is explicit, never pattern-matched from the name (§5.2). ``channels``:
#   "sample" — drawn inside the 32-configuration budget;
#   "none"   — pinned off (the plain ADORE comparators);
# ``base`` names the search space in ``cycle.SPACES``; ``fixed`` marks the 12 fixed-configuration
# entries. ``blocked`` entries are pre-registered but have no defined training-row rule on this
# track and are refused until the lead rules (see the handoff note).

SEARCHED: dict[str, dict] = {
    # ---- candidates (§5.2)
    "lightgbm":              {"base": "lightgbm", "role": "candidate", "channels": "sample"},
    "catboost":              {"base": "catboost", "role": "candidate", "channels": "sample"},
    "chemeleon_lightgbm":    {"base": "chemeleon_lightgbm", "role": "candidate",
                              "channels": "sample"},
    "gpboost":               {"base": "gpboost", "role": "candidate", "channels": "sample",
                              "seeded_gpboost": True},
    "chemprop":              {"base": "chemprop", "role": "candidate", "channels": "sample",
                              "gpu": True},
    "lightgbm_resid_ice":    {"base": "lightgbm", "role": "candidate", "channels": "sample",
                              "residual": "ice"},
    "lightgbm_resid_surrogate": {"base": "lightgbm", "role": "candidate", "channels": "sample",
                                 "residual": "fish_geomean"},
    # ---- re-tuned ADORE comparators (§5.1.2), no channel
    "rf":                    {"base": "rf", "role": "comparator", "channels": "none"},
    "xgboost":               {"base": "xgboost", "role": "comparator", "channels": "none"},
    # ---- channel-matched comparators (§1.4.3) — mandatory, claim condition (d2).
    # `channels: "matched"` excludes the empty set (C-A3-5). The re-tuned pair gets its own search
    # seed so it cannot coincide with the published pair (C-A3-4).
    "rf_chan":               {"base": "rf", "role": "comparator", "channels": "matched",
                              "seed_offset": 11},
    "xgboost_chan":          {"base": "xgboost", "role": "comparator", "channels": "matched",
                              "seed_offset": 12},
    # The published pair is a FIXED published configuration with the channel setting as the only
    # varied dimension => 3 configurations each, not 32 (C-A3-4, plan §1.4.3).
    "rf_published_chan":     {"base": "rf", "role": "comparator", "channels": "matched",
                              "published_fixed": True},
    "xgboost_published_chan": {"base": "xgboost", "role": "comparator", "channels": "matched",
                               "published_fixed": True},
    # The published cross-species comparator: two verbatim published parameter sets
    # (`PUBLISHED_GPBOOST`) x the 3 channel settings = 6 configurations, no search.
    "gpboost_published_chan": {"base": "gpboost", "role": "comparator", "channels": "matched",
                               "seeded_gpboost": True, "published_gpboost": True},
    # ---- read-across (§5.1.5)
    "readacross":            {"base": "readacross", "role": "comparator", "channels": "none",
                              "track_readacross": True},
}

#: pre-registered but undefined on this track: "multitaxa" meant *extra ADORE rows* on the closed
#: prospective track, and this track's pool already holds every ADORE taxon, so the entry has no
#: training-row rule here. Defining one now would decide a mandatory comparator's training rows
#: after the freeze — the exact C-M4P-1 failure mode. Refused until the lead rules.
BLOCKED_REASON: dict[str, str] = {
    **{n: ("'multitaxa' meant extra ADORE rows on the closed prospective track, and this pool "
           "already holds every ADORE taxon, so the entry has no training-row rule here. A3.1 "
           "aliases it to its plain counterpart; it is never fitted separately.")
       for n in ("rf_multitaxa", "xgboost_multitaxa", "rf_multitaxa_chan",
                 "xgboost_multitaxa_chan")},
}
BLOCKED = tuple(BLOCKED_REASON)

#: A3.1 aliases (review T3): reported in every comparator table as "= rf" / "= xgboost" so the
#: 21 -> 17 comparator change is auditable from the output, not only from the amendment.
ALIASES: dict[str, str] = {"rf_multitaxa": "rf", "xgboost_multitaxa": "xgboost",
                           "rf_multitaxa_chan": "rf_chan",
                           "xgboost_multitaxa_chan": "xgboost_chan"}


# ======================================================================================
# Amendment A5 (2026-11-09) — validation folds must mirror how the TEST block is built.
#
# Defect found by the gate-condition-2 smoke plus an unlogged rf diagnostic, before any logged
# channel result: Channel A was 0.0 % available on all four validation folds (0 of 42,533 / 9,386 /
# 7,085 / 1,191 rows) against 94.6 / 93.1 / 87.6 % on the test blocks, because §4/W-M4U-3's
# chemical blocking and §1.4.4's per-fold surrogate re-fit together guarantee that no validation
# chemical has any pool record. Both replacements live here, in new code: `unseen_taxa.py` is pinned
# by the freeze and is NOT edited.
# ======================================================================================

def validation_splits_classonly(pool: pd.DataFrame,
                                classes: tuple[str, ...] = U.VALIDATION_CLASSES):
    """Leave-one-class-out folds that hold out the **class only** (A5.1).

    `unseen_taxa.validation_splits` additionally removes the held-out class's chemicals from the
    training side by CAS, InChIKey and `parent_key` (W-M4U-3). Its stated rationale is that "a
    salt, counter-ion or stereoisomer cannot make a selection fold easier than the test blocks it
    mirrors" — but plan §1.2 says of the **test** construction: "A test chemical may legitimately
    appear in the training pool... That is realistic and is not leakage of the test rows. So
    chemical novelty is **reported** rather than enforced." So W-M4U-3 forbids in selection exactly
    what test construction allows, which is why fold surrogate coverage was 0 % against test's
    ~90 %. Chemical novelty stays a *reported stratum* here too.

    Yields ``(class_name, train_index, val_index)``. The held-out class is still removed
    completely, and `assert_no_test_classes` still guarantees no insect/mollusc/amphibian row is
    anywhere in the pool.
    """
    cls = pool["class"].astype("string")
    for c in classes:
        val = np.flatnonzero((cls == c).fillna(False).to_numpy())
        if not len(val):
            continue
        yield c, np.flatnonzero((cls != c).fillna(True).to_numpy()), val


class OutOfClassSurrogateFeatures:
    """Channel A computed **out-of-class**, which removes self-inclusion leakage (A5.2).

    `SurrogateFeatures.fit` aggregates the training targets by chemical and `transform` then looks
    the row's own chemical up, so a training row's own target enters its own feature: measured on
    the Malacostraca fold, all **736/736** training rows whose chemical had exactly one pool record
    had `surr_pool_geomean` equal to their own target to machine precision, and the features
    correlated 0.92-0.96 with `y` on training against **0.000** on validation.

    Here a row of class *C* receives surrogate values computed from pool records of **other classes
    only** — never its own record and never its own class. That mirrors the test block exactly,
    where the surrogate can only come from non-test classes. A row whose class is absent from the
    fitting frame (a validation or test row) is served by the full fit, which is likewise "all
    classes present in training".
    """

    def __init__(self, fish_groups: tuple[str, ...] = ("fish",), with_ice: bool = True):
        self.kw = {"fish_groups": fish_groups, "with_ice": with_ice}
        self.names_ = list(U.SURROGATE_FEATURE_NAMES)
        self.full_: U.SurrogateFeatures | None = None
        self.per_class_: dict[str, U.SurrogateFeatures] = {}

    def fit(self, train: pd.DataFrame) -> OutOfClassSurrogateFeatures:
        self.full_ = U.SurrogateFeatures(**self.kw).fit(train)
        cls = train["class"].astype("string")
        self.per_class_ = {}
        for c in sorted(set(cls.dropna())):
            other = train.loc[(cls != c).fillna(True).to_numpy()]
            if len(other):
                self.per_class_[str(c)] = U.SurrogateFeatures(**self.kw).fit(other)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.full_ is None:
            raise RuntimeError("OutOfClassSurrogateFeatures.transform before fit")
        cls = df["class"].astype("string")
        out = np.empty((len(df), len(self.names_)), dtype=float)
        pos = np.arange(len(df))
        done = np.zeros(len(df), dtype=bool)
        for c, fitted in self.per_class_.items():
            m = (cls == c).fillna(False).to_numpy()
            if m.any():
                out[pos[m]] = fitted.transform(df.loc[m])
                done |= m
        if (~done).any():                      # class unseen in training: the test-time case
            out[pos[~done]] = self.full_.transform(df.loc[~done])
        return out

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    def surrogate_prediction(self, df: pd.DataFrame, kind: str = "ice") -> np.ndarray:
        idx = {"fish_geomean": 0, "most_sensitive": 1, "pool_geomean": 2, "ice": 3}[kind]
        return self.transform(df)[:, idx]


class TrackChannelBuilder:
    """`unseen_taxa.ChannelBuilder` with Channel A computed out-of-class (A5.2).

    Channel B (`HigherRankFeatures`) is reused unchanged from the pinned module: it is a pure
    function of the class name and carries no target, so self-inclusion cannot arise.
    """

    def __init__(self, builder, channels: tuple[str, ...] = (), *,
                 fish_groups: tuple[str, ...] = ("fish",)):
        self.base = U.TaxonBlindBuilder(builder)
        self.channels = tuple(channels)
        self.surrogate_ = (OutOfClassSurrogateFeatures(fish_groups=fish_groups)
                           if "surrogate" in channels else None)
        self.higher_ = U.HigherRankFeatures() if "higher_rank" in channels else None
        self.names_: list[str] = []

    def fit(self, train: pd.DataFrame) -> TrackChannelBuilder:
        self.base.fit(train)
        names = list(self.base.names_)
        if self.surrogate_ is not None:
            self.surrogate_.fit(train)
            names += list(self.surrogate_.names_)
        if self.higher_ is not None:
            self.higher_.fit(train)
            names += list(self.higher_.names_)
        self.names_ = names
        U.assert_no_trait_columns(self.names_, what="TrackChannelBuilder")
        U.assert_no_class_or_lower(self.names_, what="TrackChannelBuilder")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = [self.base.transform(df)]
        if self.surrogate_ is not None:
            parts.append(self.surrogate_.transform(df))
        if self.higher_ is not None:
            parts.append(self.higher_.transform(df))
        return np.column_stack(parts).astype(np.float32)

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    @property
    def n_features(self) -> int:
        return len(self.names_)


class ResidualOutOfClassModel:
    """`unseen_taxa.ResidualSurrogateModel` with the base computed out-of-class (A5.2).

    The pinned wrapper fits an inner model on ``y - base`` where ``base`` comes from
    `SurrogateFeatures`, i.e. a training row's base partly contains its own target. Those two
    candidates (`lightgbm_resid_ice`, `lightgbm_resid_surrogate`) were the worst affected by the
    defect: their base was constant on validation rows, so in CV they reduced to plain lightgbm on
    a shifted target while behaving completely differently at test time.
    """

    def __init__(self, inner_factory, params: dict | None = None, *, base_kind: str = "ice",
                 fish_groups: tuple[str, ...] = ("fish",)):
        self.inner = inner_factory(dict(params or {}))
        self.base_kind = base_kind
        self.surrogate = OutOfClassSurrogateFeatures(fish_groups=fish_groups)
        self.fitted_ = False

    def fit(self, X, y, *, train_rows: pd.DataFrame, sample_weight=None, **ctx):
        self.surrogate.fit(train_rows)
        base = self.surrogate.surrogate_prediction(train_rows, self.base_kind)
        self.inner.fit(X, np.asarray(y, dtype=float) - base, train_rows=train_rows,
                       sample_weight=sample_weight, **ctx)
        self.fitted_ = True
        return self

    def predict(self, X, rows: pd.DataFrame | None = None, **ctx) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ResidualOutOfClassModel.predict before fit")
        if rows is None:
            raise ValueError("ResidualOutOfClassModel needs rows")
        base = self.surrogate.surrogate_prediction(rows, self.base_kind)
        return np.asarray(self.inner.predict(X, rows=rows, **ctx), dtype=float) + base


def make_residual_oac(inner_factory, *, base_kind: str = "ice"):
    return lambda params: ResidualOutOfClassModel(inner_factory, params, base_kind=base_kind)


class TrackChemprop:
    """chemprop for this track: a constant placeholder species key (lead ruling 2026-11-09, (a)).

    `ChempropToxModel`'s `ConditionEncoder.fit` reads `rows["tax_gs"]` **unconditionally**
    (`chemprop_model.py:71`), whatever `species_dim` is, and builds a learned embedding per species.
    This track never creates `tax_gs` (T4), so chemprop crashed with `KeyError('tax_gs')` — the
    invariant working as designed, since real species keys would give chemprop a below-class
    channel (§1.1). `chemprop_model.py` is shared with the closed track and is NOT edited; instead,
    as for `TrackReadAcross` (C-A3-1), this wrapper supplies a **constant** placeholder on copies of
    the frames handed to chemprop — never on the pool.

    With the placeholder, `n_species == 2` (placeholder + out-of-vocabulary slot) and training,
    validation and test rows all share **one** species index, so any `species_dim > 0` draw is a
    single learned vector: an **information-free learned bias**, carrying no taxonomy. `species_dim`
    stays sampled over [0, 8, 16, 32], so the sampled space is unchanged (ruling (a)).
    """

    def __init__(self, params: dict | None = None, seed: int = 0):
        from sentinel.models.chemtox.chemprop_model import ChempropToxModel, SPECIES_COL

        self._col = SPECIES_COL
        self.inner = ChempropToxModel(params=dict(params or {}), seed=seed)

    def _prep(self, rows: pd.DataFrame | None) -> pd.DataFrame | None:
        if rows is None:
            return None
        out = rows.copy()
        out[self._col] = CHEMPROP_SPECIES_PLACEHOLDER
        return out

    def fit(self, X, y, *, train_rows: pd.DataFrame, valid_rows: pd.DataFrame | None = None,
            **ctx):
        self.inner.fit(X, y, train_rows=self._prep(train_rows),
                       valid_rows=self._prep(valid_rows), **ctx)
        return self

    def predict(self, X, *, rows: pd.DataFrame | None = None, **ctx) -> np.ndarray:
        return np.asarray(self.inner.predict(X, rows=self._prep(rows), **ctx), dtype=float)

    @property
    def enc_(self):
        return self.inner.enc_


#: constant placeholder species key for chemprop (mirrors `READACROSS_SPECIES_PLACEHOLDER`)
CHEMPROP_SPECIES_PLACEHOLDER = "__dropped__"


def alias_rows() -> list[dict]:
    """The alias lines a comparator table must print (plan A3.1 / review T3, D3).

    The four `*multitaxa*` names are never fitted, so they have no scores of their own; a comparator
    table still has to show them, or the 21 -> 17 comparator change is visible only in the
    amendment. Each row renders as e.g. `rf_multitaxa = rf`.
    """
    return [{"entry": name, "alias_of": target, "render": f"{name} = {target}",
             "role": "comparator", "fitted": False,
             "note": "alias (A3.1): identical rows, configuration and numbers; "
                     "not evaluated as an independent comparator"}
            for name, target in ALIASES.items()]

#: Entries whose estimator needs a taxonomic **grouping** variable. The `prereg_snapshot_v6`
#: caveat admits only `rank_phylum` (3 levels) here, which makes `gpboost_published` an ADAPTED
#: estimator; and `readacross` hard-codes species/group columns the pool does not have, so on this
#: track it degenerates to the plain chemical-similarity kNN §5.1.5 already describes. Both
#: readings change what a **mandatory comparator** is, so they are pre-fit protocol decisions, not
#: engineering choices: refused until the amendment is reviewed.
GROUPING_DEPENDENT = ("gpboost", "gpboost_published_chan", "readacross")

#: Grouping admitted by the v6 caveat — strictly above the held-out class.
ADMITTED_GROUP_COLUMNS = ("rank_phylum",)

#: Columns no track adapter may ever create (review T4). `chemprop`'s `ConditionEncoder` keys a
#: **species embedding** on `tax_gs`, `readacross` matches species on it and
#: `gpboost_model.GROUP_COLUMNS_DEFAULT` groups on it — the column's *absence* is what makes a
#: below-class channel impossible on this track, so it is an invariant with a test, not merely a
#: fixed defect.
FORBIDDEN_COLUMNS = ("tax_gs", "tax_genus", "tax_family", "tax_species")

#: Constant placeholder species key for the track-local read-across adapter (C-A3-1). Mirrors the
#: `to_model_schema` placeholder pattern: the column exists so the estimator runs, but it carries
#: no taxonomic information, so species matching can never succeed.
READACROSS_SPECIES_PLACEHOLDER = "__dropped__"


def assert_no_forbidden_columns(df: pd.DataFrame, *, what: str = "frame") -> None:
    """Refuse any frame that carries a below-class taxonomic key (review T4)."""
    bad = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
    if bad:
        raise AssertionError(
            f"{what}: forbidden below-class taxonomic column(s) {bad}. This track drops the trait "
            "block; a species/genus key would give chemprop a species embedding, read-across a "
            "species match and GPBoost a species random effect. See plan §1.1 / §1.4.2 and the "
            "prereg_snapshot_v6 caveat.")


class TrackReadAcross:
    """Read-across for this track: the plain chemical-similarity kNN §5.1.5 pre-registers.

    `ReadAcrossKNN.fit` reads `rows[SPECIES_COL]` **unconditionally** (`readacross.py:105`),
    before and regardless of `match_species`, so on this track it raised `KeyError('tax_gs')` —
    it did not "degenerate" as amendment A3.2 first claimed (review C-A3-1). `readacross.py` is
    not edited (it is shared with the closed prospective track); instead this wrapper:

    * supplies a **constant** placeholder species key, so the column exists and carries nothing;
    * pins ``match_species=False`` (review C-A3-2), because every *validation* fold's held-out
      class shares its taxon group with training rows (fish for Actinopterygii, crustaceans for
      the other three, 1,145-5,299 rows), whereas a *test* block's `taxon_group` matches no pool
      row at all. Leaving the group level live would make selection easier than test — inverting
      §1.4.4's own honest-analogue principle.

    The result is `widening_ == {species: 0, group: 0, global: N, fallback_mean: k}`, asserted in
    `tests/test_chemtox_unseen_select.py`.
    """

    def __init__(self, params: dict | None = None, seed: int = 0):
        from sentinel.models.chemtox import readacross as RA

        p = dict(params or {})
        p["match_species"] = False                 # pinned, never sampled on this track
        self.inner = RA.ReadAcrossKNN(p, seed) if hasattr(RA, "ReadAcrossKNN") else None
        if self.inner is None:                     # pragma: no cover - API drift guard
            raise RuntimeError("readacross.ReadAcrossKNN not found")
        self._RA = RA

    def _prep(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy()
        out[self._RA.SPECIES_COL] = READACROSS_SPECIES_PLACEHOLDER
        return out

    def fit(self, X, y, *, train_rows: pd.DataFrame, sample_weight=None, **ctx):
        self.inner.fit(X, y, train_rows=self._prep(train_rows), sample_weight=sample_weight, **ctx)
        return self

    def predict(self, X, rows: pd.DataFrame | None = None, **ctx) -> np.ndarray:
        if rows is None:
            raise ValueError("TrackReadAcross needs rows")
        return np.asarray(self.inner.predict(X, rows=self._prep(rows), **ctx), dtype=float)

    @property
    def widening_(self) -> dict:
        return self.inner.widening_


def add_grouping_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supply the ADORE **grouping** column names that `gpboost` and `readacross` require.

    ``gpboost_model.GROUP_COLUMNS_DEFAULT`` is ``("tax_gs", "tax_genus", "tax_group")`` and
    ``readacross`` hard-codes ``SPECIES_COL = "tax_gs"`` / ``GROUP_COL = "tax_group"``, but the
    frozen pool carries only the ECOTOX names.

    **A first version of this adapter derived ``tax_gs`` from ``latin_name`` and ``tax_genus`` from
    ``genus``. That was wrong and has been removed.** The freeze-time caveat recorded with
    ``prereg_snapshot_v6`` states: *"the trait block is dropped on this track, so the pool has no
    ``tax_gs`` and the only grouping variable Channel B admits is ``rank_phylum`` (3 levels); the
    published method groups species/genus/group, so ``gpboost_published`` here is an ADAPTED
    estimator and must be labelled as such in every comparison."* Supplying real species and genus
    identifiers would reintroduce taxonomy at **species and genus rank** — below class rank, which
    §1.1 and §1.4.2 forbid — and would let a random effect absorb species-level structure **during
    training**, changing the estimator, even though an unseen test species collapses to the
    population mean at predict time.

    This adapter therefore supplies **only ranks strictly above the held-out class**:
    ``rank_phylum`` / ``rank_subphylum`` / ``rank_clade``, from ``unseen_taxa.higher_ranks``, which
    is a pure function of the class name. ``tax_gs`` and ``tax_genus`` are **not** created, so any
    model that requires them fails loudly rather than silently regaining taxonomy (see
    ``GROUPING_DEPENDENT``).
    """
    out = df.copy()
    ranks = U.higher_ranks(out)
    for c in U.HIGHER_RANK_NAMES:
        out[c] = ranks[c].to_numpy()
    if "tax_group" not in out.columns and "taxon_group" in out.columns:
        # `tax_group` is the ADORE pool grouping (fish / crustaceans / algae). It is **not** above
        # class rank (review T2): for insects the group label is one-to-one with `Insecta` and for
        # amphibians with `Amphibia` — the same proxy structure delta review D-2 rejected for
        # `rank_subphylum=Hexapoda`. It is admissible here only because (i) it is a
        # `TRAIT_PREFIXES` member, so `TaxonBlindBuilder` deletes it from every feature matrix,
        # (ii) it is never a GPBoost grouping column (`ADMITTED_GROUP_COLUMNS`), and (iii) its
        # test-block values match no training level. Its only use is read-across's widening level
        # (§5.1.5), which `TrackReadAcross` pins off anyway.
        out["tax_group"] = out["taxon_group"]
    assert_no_forbidden_columns(out, what="add_grouping_columns output")
    return out


def entry_spec(features: dict) -> FeatureSpec:
    """``FeatureSpec`` for a sampled configuration, with ``chemical`` pinned to the plan's value."""
    f = dict(features)
    f.pop("chemical", None)                      # §4: chemical="mw_logp" is pre-registered
    return FeatureSpec(chemical="mw_logp", **f)


def _sample_base(base: str, budget: int, seed: int) -> list[dict]:
    """Draw from the right space: track-local GPBoost, `spaces` for chemprop, else `cycle`."""
    if base == "gpboost":                      # C-A3-3: no `group_columns` in the sampled space
        return CYC._sample_space(GPBOOST_SPACE_TRACK, budget, seed) if hasattr(
            CYC, "_sample_space") else _sample_space(GPBOOST_SPACE_TRACK, budget, seed)
    if base == "chemprop":                     # W-A3-3: chemprop's space lives in `spaces`
        return SPACES.sample("chemprop", budget=budget, seed=seed)
    return CYC.sample(base, budget=budget, seed=seed)


def _sample_space(space: dict, budget: int, seed: int) -> list[dict]:
    """Local copy of `cycle.sample`'s draw for a space that is not in `cycle.SPACES`."""
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < budget and attempts < budget * 200:
        attempts += 1
        cfg = {"features": {k: CYC._pick(rng, v) for k, v in space["features"].items()},
               "params": {k: CYC._pick(rng, v) for k, v in space["params"].items()}}
        key = repr(sorted(cfg["features"].items())) + repr(sorted(cfg["params"].items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg)
    if len(out) < budget:
        raise RuntimeError(f"space exhausted at {len(out)} < {budget}")
    return out


def sample_entry(name: str, budget: int, seed: int = 20261109) -> list[dict]:
    """Configurations for one entry, with ``channels`` folded into the same budget.

    Three review fixes live here:

    * **C-A3-4** — a per-entry ``seed_offset``, so two entries over the same base space can never
      draw an identical configuration list (`rf_chan` was 32/32 identical to `rf_published_chan`,
      and the `gpboost` candidate to the `gpboost_published_chan` comparator);
    * **C-A3-4** — a ``published_fixed`` entry is the published configuration with the channel
      setting as its only varied dimension: **3 configurations, not 32**;
    * **C-A3-5** — a channel-matched comparator never draws the empty channel set.

    Deliberately *not* changed: `lightgbm`, `lightgbm_resid_ice` and `lightgbm_resid_surrogate`
    share one draw. All three are **candidates** with different estimators (plain, ICE-residual,
    surrogate-residual), so they produce different numbers; holding the hyper-parameters fixed
    makes the residual base a controlled ablation. C-A3-4's defect was a *comparator* identical to
    a *candidate* (a self-comparison under (b)/(d2)); no comparator now shares a draw with any
    candidate or with another comparator, which `tests/test_chemtox_unseen_select.py` asserts.
    """
    meta = SEARCHED[name]
    mode = meta.get("channels", "none")
    choices = CHANNEL_CHOICES_MATCHED if mode == "matched" else CHANNEL_CHOICES

    if meta.get("published_gpboost"):
        return [{"features": dict(PUBLISHED_FEATURES), "params": dict(prm),
                 "channels": list(ch), "published_form": form}
                for form, prm in PUBLISHED_GPBOOST.items() for ch in choices]
    if meta.get("published_fixed"):
        return [{"features": dict(PUBLISHED_FEATURES),
                 "params": dict(PUBLISHED_PARAMS[meta["base"]]),
                 "channels": list(ch)} for ch in choices]

    entry_seed = seed + int(meta.get("seed_offset", 0))
    cfgs = _sample_base(meta["base"], budget, entry_seed)
    rng = np.random.default_rng(entry_seed + 1)
    for c in cfgs:
        if mode == "none":
            c["channels"] = []
        else:
            c["channels"] = list(choices[int(rng.integers(0, len(choices)))])
        if meta["base"] == "chemprop":
            c["params"]["multitask"] = False          # W-A3-1 / C-M4U-4: never the fish head
    return cfgs


#: Lightning accelerator for chemprop, set from ``--accelerator``. Applied inside `build_model` on
#: a local copy only, so the RECORDED params stay equal to the sampled ones (the merge drift-check
#: compares them) while the device is recorded in the TrainingRun config instead.
_ACCELERATOR = "cpu"


def build_model(name: str, params: dict, seed: int):
    """Model factory for an entry, honouring the seeded-GPBoost and residual wrappers."""
    meta = SEARCHED[name]
    params = dict(params)
    U.assert_taxon_agnostic(meta["base"], params)        # W-A3-1: §5.2's C-M4U-4 enforcement
    if meta["base"] == "chemprop":
        params = CYC.attach_device("chemprop", params, _ACCELERATOR)
    if meta.get("seeded_gpboost"):
        # Pin the random-effect grouping to the only variable the v6 caveat admits. Never
        # species/genus: that is below class rank and this track drops the trait block. The
        # sampled space no longer carries `group_columns` at all (C-A3-3), so this sets rather
        # than overwrites, and the recorded configuration cannot misreport the grouping.
        params["group_columns"] = list(ADMITTED_GROUP_COLUMNS)
        factory = U.make_seeded_gpboost(seed=seed, threads=int(_THREADS))
    elif meta.get("track_readacross"):
        factory = lambda p: TrackReadAcross(p, seed)     # C-A3-1 / C-A3-2
    elif meta["base"] == "chemprop":
        factory = lambda p: TrackChemprop(p, seed)       # lead ruling 2026-11-09, option (a)
    else:
        factory = CYC.make_model(meta["base"], seed)
    if meta.get("residual"):
        factory = make_residual_oac(factory, base_kind=meta["residual"])
    return factory(params)


def run_entry(name: str, cfgs: list[dict], pool: pd.DataFrame, folds: list[str], *,
              seed: int = 0, out_dir: Path, indices: list[int] | None = None,
              out_name: str | None = None) -> dict:
    """Fit the given configurations of one entry on every fold; return the summary.

    ``indices`` are the **global** configuration indices of ``cfgs`` within the entry's full sampled
    list, so a shard records the same ``config_index`` a single-process run would, and a merge can
    verify that every index is covered exactly once.
    """
    if indices is None:
        indices = list(range(len(cfgs)))
    if len(indices) != len(cfgs):
        raise ValueError("indices and cfgs differ in length")
    meta = SEARCHED[name]
    y_all = pd.to_numeric(pool[TARGET], errors="coerce").to_numpy(dtype=float)
    splits = [(c, tr, va) for c, tr, va in validation_splits_classonly(pool) if c in folds]
    if not splits:
        raise SystemExit(f"no validation fold matched {folds}")

    rows: list[dict] = []
    for i, cfg in zip(indices, cfgs):
        spec = entry_spec(cfg["features"])
        channels = tuple(cfg.get("channels") or ())
        per_fold: list[dict] = []
        # What the fits actually used, which is what gets recorded (C-A3-3): the sampled dict is
        # never written to the audit trail unmodified.
        effective_params = dict(cfg["params"])
        widening: dict | None = None
        t0 = time.time()
        for cls, tr, va in splits:
            train_rows = pool.iloc[tr].reset_index(drop=True)
            val_rows = pool.iloc[va].reset_index(drop=True)
            y_tr, y_va = y_all[tr], y_all[va]
            ok = np.isfinite(y_tr)
            train_rows, y_tr = train_rows.loc[ok].reset_index(drop=True), y_tr[ok]

            builder = TrackChannelBuilder(FeatureBuilder(spec), channels)
            Xtr = builder.fit_transform(train_rows)
            Xva = builder.transform(val_rows)
            U.assert_no_trait_columns(builder.names_, what=f"{name}/{cls}")
            U.assert_no_class_or_lower(builder.names_, what=f"{name}/{cls}")

            model = build_model(name, cfg["params"], seed)
            model.fit(Xtr, y_tr, train_rows=train_rows, valid_rows=val_rows,
                      feature_names=builder.names_, builder=builder, sample_weight=None)
            if meta.get("seeded_gpboost"):
                # C-A3-3: prove the fit used the admitted grouping, not a sampled one.
                used = tuple(model.inner_.group_columns_)
                if used != ADMITTED_GROUP_COLUMNS:
                    raise AssertionError(
                        f"{name}/{cls}: GPBoost grouped on {used}, expected "
                        f"{ADMITTED_GROUP_COLUMNS} (prereg_snapshot_v6 caveat)")
                effective_params["group_columns"] = list(ADMITTED_GROUP_COLUMNS)
            if meta.get("track_readacross"):
                # C-A3-1 / C-A3-2: §5.1.5's counters, and proof the species/group levels are dead.
                w = dict(model.widening_)
                if w.get("species") or w.get("group"):
                    raise AssertionError(
                        f"{name}/{cls}: read-across widened to species/group ({w}); this track "
                        "pre-registers the plain chemical-similarity kNN (§5.1.5)")
                widening = w
                effective_params["match_species"] = False
            p = np.asarray(model.predict(Xva, rows=val_rows), dtype=float)
            m = np.isfinite(y_va) & np.isfinite(p)
            rmse = float(np.sqrt(np.mean((y_va[m] - p[m]) ** 2))) if m.any() else float("nan")
            per_fold.append({"fold": cls, "rmse": rmse, "n_train": int(len(train_rows)),
                             "n_val": int(m.sum()), "n_features": builder.n_features,
                             # A5.1-a: reported beside each fold's score, so the fold-vs-test
                             # availability mismatch is visible rather than inferred.
                             "surrogate_available_val": float(
                                 U.surrogate_available(val_rows, train_rows).mean())})
            log.info("%s cfg%02d %-18s rmse=%.4f (train %d)", name, i, cls, rmse,
                     len(train_rows))

        agg = aggregate_folds(per_fold)
        rows.append({
            "entry": name, "role": meta["role"], "config_index": i,
            "features": spec.to_dict(), "params": effective_params, "channels": list(channels),
            "alias_of": meta.get("alias_of"), "widening": widening,
            "published_form": cfg.get("published_form"),
            "per_fold": per_fold, **agg, "seconds": time.time() - t0,
        })

    summary = summarise(name, rows, [c for c, _, _ in splits], seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / (out_name or f"{name}.json")).write_text(json.dumps(summary, indent=2, default=str))
    return summary


def summarise(name: str, rows: list[dict], folds: list[str], seed: int) -> dict:
    """Selection summary over a set of configuration rows (single run or merged shards)."""
    meta = SEARCHED[name]
    rows = sorted(rows, key=lambda r: r["config_index"])
    ranked = sorted([r for r in rows if np.isfinite(r["mean_rmse"])],
                    key=lambda r: (r["mean_rmse"], r["worst_rmse"]))
    top5 = [r["mean_rmse"] for r in ranked[:5]]
    return {
        "entry": name, "role": meta["role"], "base": meta["base"],
        "n_configurations": len(rows), "n_folds": len(folds),
        "folds": list(folds), "seed": seed,
        "best": ranked[0] if ranked else None,
        "top5_mean_rmse": top5,
        "top5_spread": float(max(top5) - min(top5)) if len(top5) > 1 else 0.0,
        "n_failed": len(rows) - len(ranked),
        "configs": rows,
    }


def aggregate_folds(per_fold: list[dict]) -> dict:
    """The per-configuration score over its folds — the SAME arithmetic `run_entry` uses."""
    finite = [f["rmse"] for f in per_fold if np.isfinite(f["rmse"])]
    return {
        "mean_rmse": float(np.mean(finite)) if finite else float("nan"),
        "worst_rmse": float(np.max(finite)) if finite else float("nan"),
        "worse_half": (float(np.mean(sorted(finite)[len(finite) // 2:])) if finite else float("nan")),
        "n_folds_ok": len(finite),
    }


def merge_fold_jobs(name: str, out_dir: Path, budget: int) -> dict:
    """Merge one-configuration-x-one-fold job files into the entry summary (2 h cap, chemprop).

    A chemprop configuration takes 20-122 min over the four folds (3-seed draws are the slow ones),
    so a configuration can sit at the cap; a single fold is estimated at <= ~30 min. Each job runs
    `run_entry` for ONE configuration on ONE fold, so the fit and the fold RMSE are computed exactly
    as in a single-process run. This assembles them, recomputing each configuration's score with
    `aggregate_folds` (the same arithmetic as `run_entry`), and refuses unless every configuration
    has all four folds exactly once and matches the re-sampled configuration at its index.
    """
    expected = sample_entry(name, budget)
    files = sorted(out_dir.glob(f"{name}.cfg*.fold-*.json"))
    if not files:
        raise SystemExit(f"fold-merge {name}: no fold job files in {out_dir}")
    by_cfg: dict[int, dict] = {}
    seeds = set()
    for f in files:
        d = json.loads(f.read_text())
        seeds.add(d["seed"])
        for r in d["configs"]:
            slot = by_cfg.setdefault(r["config_index"], {"row": r, "folds": {}})
            for pf in r["per_fold"]:
                if pf["fold"] in slot["folds"]:
                    raise SystemExit(f"fold-merge {name}: config {r['config_index']} fold "
                                     f"{pf['fold']} appears twice")
                slot["folds"][pf["fold"]] = pf
    if len(seeds) != 1:
        raise SystemExit(f"fold-merge {name}: jobs disagree on seed {seeds}")
    if sorted(by_cfg) != list(range(len(expected))):
        missing = sorted(set(range(len(expected))) - set(by_cfg))
        raise SystemExit(f"fold-merge {name}: configurations missing {missing}")
    rows = []
    for i, slot in sorted(by_cfg.items()):
        if sorted(slot["folds"]) != sorted(U.VALIDATION_CLASSES):
            raise SystemExit(f"fold-merge {name}: config {i} has folds {sorted(slot['folds'])}, "
                             "not all four")
        r, exp = slot["row"], expected[i]
        rec = {k: v for k, v in r["params"].items() if k not in _EFFECTIVE_ONLY_KEYS}
        want = {k: v for k, v in exp["params"].items() if k not in _EFFECTIVE_ONLY_KEYS}
        if (json.dumps(rec, sort_keys=True, default=str) != json.dumps(want, sort_keys=True, default=str)
                or list(r["channels"]) != list(exp["channels"])):
            raise SystemExit(f"fold-merge {name}: config {i} does not match the re-sampled one")
        per_fold = [slot["folds"][c] for c in U.VALIDATION_CLASSES]      # canonical fold order
        rows.append({**r, "per_fold": per_fold, **aggregate_folds(per_fold)})
    summary = summarise(name, rows, list(U.VALIDATION_CLASSES), next(iter(seeds)))
    summary["merged_from"] = [f.name for f in files]
    summary["granularity"] = "one configuration x one fold per job"
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


#: sampled keys that `run_entry` legitimately adds to the recorded params (C-A3-3 / C-A3-1)
_EFFECTIVE_ONLY_KEYS = ("group_columns", "match_species")


def merge_shards(name: str, out_dir: Path, budget: int) -> dict:
    """Merge one entry's shard files, verifying integrity before anything is logged.

    Refuses unless the shards cover **every** global configuration index of the entry's freshly
    re-sampled list **exactly once**, every shard used the same folds, and each recorded
    configuration equals the re-sampled one at its index (so a shard cannot have run a drifted
    list). Only then is the merged `<entry>.json` written.
    """
    expected = sample_entry(name, budget)
    files = sorted(out_dir.glob(f"{name}.shard*of*.json"))
    if not files:
        raise SystemExit(f"merge {name}: no shard files in {out_dir}")
    rows, folds_seen, seeds = [], set(), set()
    for f in files:
        d = json.loads(f.read_text())
        folds_seen.add(tuple(d["folds"]))
        seeds.add(d["seed"])
        rows.extend(d["configs"])
    if len(folds_seen) != 1 or len(seeds) != 1:
        raise SystemExit(f"merge {name}: shards disagree on folds {folds_seen} or seed {seeds}")
    folds = list(next(iter(folds_seen)))
    if sorted(folds) != sorted(U.VALIDATION_CLASSES):
        raise SystemExit(f"merge {name}: shards did not run all four folds ({folds})")
    idx = [r["config_index"] for r in rows]
    if sorted(idx) != list(range(len(expected))):
        missing = sorted(set(range(len(expected))) - set(idx))
        dup = sorted(i for i in set(idx) if idx.count(i) > 1)
        raise SystemExit(f"merge {name}: index coverage wrong (missing {missing}, duplicated {dup})")
    for r in rows:
        exp = expected[r["config_index"]]
        rec = {k: v for k, v in r["params"].items() if k not in _EFFECTIVE_ONLY_KEYS}
        want = {k: v for k, v in exp["params"].items() if k not in _EFFECTIVE_ONLY_KEYS}
        if (json.dumps(rec, sort_keys=True, default=str) != json.dumps(want, sort_keys=True, default=str)
                or list(r["channels"]) != list(exp["channels"])):
            raise SystemExit(f"merge {name}: config {r['config_index']} does not match the "
                             "re-sampled configuration — refusing to merge a drifted list")
    summary = summarise(name, rows, folds, next(iter(seeds)))
    summary["merged_from"] = [f.name for f in files]
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", nargs="+", required=True,
                    help=f"model entries to search; known: {sorted(SEARCHED)}")
    ap.add_argument("--budget", type=int, default=CYC.DEFAULT_BUDGET)
    ap.add_argument("--folds", nargs="*", default=list(U.VALIDATION_CLASSES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="skip the registry/TrainingRun write (development only)")
    ap.add_argument("--log-registry", action="store_true")
    ap.add_argument("--allow-grouping-dependent", action="store_true",
                    help="run the GROUPING_DEPENDENT entries; only after the lead's reviewed "
                         "amendment on the grouping variable")
    ap.add_argument("--shard", type=int, default=0,
                    help="run only configurations with global index %% shards == shard")
    ap.add_argument("--shards", type=int, default=1,
                    help="split one entry's SAME sampled list across processes (2 h cap)")
    ap.add_argument("--merge", action="store_true",
                    help="merge an entry's shard files (integrity-checked), then log")
    ap.add_argument("--config-index", type=int, default=None,
                    help="fold job: run ONE configuration (global index) on the given --folds")
    ap.add_argument("--fold-merge", action="store_true",
                    help="merge one-config-x-one-fold job files (integrity-checked), then log")
    ap.add_argument("--accelerator", default="cpu", choices=("cpu", "gpu"),
                    help="chemprop only; run GPU jobs through `python -m sentinel.core.gpu`")
    a = ap.parse_args(argv)
    global _ACCELERATOR
    _ACCELERATOR = a.accelerator
    if not (0 <= a.shard < a.shards):
        raise SystemExit(f"--shard {a.shard} out of range for --shards {a.shards}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bad = [e for e in a.entries if e in BLOCKED]
    if bad:
        raise SystemExit(
            f"entries {bad} are pre-registered but have no defined training-row rule on this "
            "track ('multitaxa' meant extra ADORE rows, and this pool already holds every ADORE "
            "taxon). Defining one now would set a mandatory comparator's training rows after the "
            "freeze. Awaiting the lead's ruling.")
    unknown = [e for e in a.entries if e not in SEARCHED]
    if unknown:
        raise SystemExit(f"unknown entries {unknown}; known: {sorted(SEARCHED)}")
    grouped = [e for e in a.entries if e in GROUPING_DEPENDENT]
    if grouped and not a.allow_grouping_dependent:
        raise SystemExit(
            f"entries {grouped} need a taxonomic grouping variable. The prereg_snapshot_v6 caveat "
            f"admits only {ADMITTED_GROUP_COLUMNS} (3 levels), making gpboost_published an ADAPTED "
            "estimator, and readacross's species/group columns do not exist in the frozen pool. "
            "Both change what a mandatory comparator is, so they are pre-fit protocol decisions. "
            "Awaiting the reviewed amendment; pass --allow-grouping-dependent once it is approved.")

    # Registry gate, checked BEFORE any compute (it used to be checked only afterwards, so a
    # mis-specified run burned hours before being refused). A number logged under the rotation's
    # split must be the full pre-registered budget over all four folds (W-A3-2); a shard is never
    # logged on its own — only a verified merge is.
    if a.log_registry:
        # A fold job is refused FIRST. (A first version nested this check inside a block guarded by
        # `config_index is None`, so it could never fire; its own test then ran a real chemprop fit
        # on CPU instead of being refused. Caught by the test, killed by its timeout, nothing written.)
        if a.config_index is not None:
            raise SystemExit("--log-registry refused for a fold job: log only a verified --fold-merge")
        if sorted(a.folds) != sorted(U.VALIDATION_CLASSES) or a.budget != CYC.DEFAULT_BUDGET:
            raise SystemExit(
                f"--log-registry refused: budget={a.budget} folds={a.folds}. A logged number must "
                "be the full pre-registered budget over all four folds (review W-A3-2).")
        if a.shards > 1 and not a.merge:
            raise SystemExit("--log-registry refused for a single shard: log only a verified --merge")

    # The freeze must still verify before any selection fit (plan §7 / W-D-1).
    pins = G.verify_freeze()
    log.info("freeze verified: plan %s module %s", pins["plan_sha256"][:16],
             pins["module_sha256"][:16])
    for b in ("insects", "molluscs", "amphibians"):
        if G.registry_rows(b):
            raise SystemExit(f"registry already holds test rows for {b}; refusing to run")

    pool = add_grouping_columns(U.to_model_schema(pd.read_parquet(POOL)))
    U.assert_no_test_classes(pool, what="training pool")
    log.info("pool %d rows, folds %s", len(pool), a.folds)

    out_dir = Path(a.out) if a.out else (paths.RUNS_ROOT / "chemtox" / "unseen_taxa" /
                                         "stage2_select")
    summaries = {}
    for name in a.entries:
        if a.fold_merge:
            s = merge_fold_jobs(name, out_dir, a.budget)
            summaries[name] = s
            b = s["best"]
            log.info("== %s FOLD-MERGED from %d job files: best mean RMSE %.4f (%d configs)",
                     name, len(s["merged_from"]), b["mean_rmse"] if b else float("nan"),
                     s["n_configurations"])
            continue
        if a.config_index is not None:
            all_cfgs = sample_entry(name, a.budget)
            i = a.config_index
            if len(a.folds) != 1:
                raise SystemExit("a fold job takes exactly one --folds value")
            out_name = f"{name}.cfg{i:02d}.fold-{a.folds[0]}.json"
            if a.smoke:
                s = run_entry(name, [all_cfgs[i]], pool, a.folds, seed=a.seed, out_dir=out_dir,
                              indices=[i], out_name=out_name)
            else:
                # every run goes through TrainingRun / TRAINING_LOG.md, fold jobs included
                with TrainingRun("M4 ChemTox", f"{EXPERIMENT}/{name}", config={
                        "entry": name, "config_index": i, "fold": a.folds[0], "seed": a.seed,
                        "role": SEARCHED[name]["role"], "track": "unseen_taxa",
                        "granularity": "one configuration x one fold",
                        "accelerator": a.accelerator,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "plan_sha256": pins["plan_sha256"]},
                        plan_ref="docs/preregistration/chemtox_unseen_taxa_plan.md §4, A7",
                        artifacts_dir=str(out_dir)) as run:
                    s = run_entry(name, [all_cfgs[i]], pool, a.folds, seed=a.seed,
                                  out_dir=out_dir, indices=[i], out_name=out_name)
                    run.metric("fold_rmse", s["configs"][0]["per_fold"][0]["rmse"])
            log.info("== %s fold job cfg%02d %s rmse %.4f", name, i, a.folds[0],
                     s["configs"][0]["per_fold"][0]["rmse"])
            continue
        if a.merge:
            s = merge_shards(name, out_dir, a.budget)
            summaries[name] = s
            b = s["best"]
            log.info("== %s MERGED from %d shard files: best mean RMSE %.4f (%d configs, %d failed)",
                     name, len(s["merged_from"]), b["mean_rmse"] if b else float("nan"),
                     s["n_configurations"], s["n_failed"])
            continue
        all_cfgs = sample_entry(name, a.budget)
        indices = [i for i in range(len(all_cfgs)) if i % a.shards == a.shard]
        cfgs = [all_cfgs[i] for i in indices]
        out_name = (f"{name}.shard{a.shard}of{a.shards}.json" if a.shards > 1 else None)
        if a.smoke:
            s = run_entry(name, cfgs, pool, a.folds, seed=a.seed, out_dir=out_dir,
                          indices=indices, out_name=out_name)
        else:
            with TrainingRun("M4 ChemTox", f"{EXPERIMENT}/{name}", config={
                    "entry": name, "budget": a.budget, "folds": a.folds, "seed": a.seed,
                    "role": SEARCHED[name]["role"], "track": "unseen_taxa",
                    "shard": a.shard, "shards": a.shards, "config_indices": indices,
                    "accelerator": a.accelerator,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "plan_sha256": pins["plan_sha256"]},
                    plan_ref="docs/preregistration/chemtox_unseen_taxa_plan.md §4",
                    artifacts_dir=str(out_dir)) as run:
                s = run_entry(name, cfgs, pool, a.folds, seed=a.seed, out_dir=out_dir,
                              indices=indices, out_name=out_name)
                if s["best"]:
                    run.metric("mean_rmse_best", s["best"]["mean_rmse"])
                    run.metric("worst_fold_rmse_best", s["best"]["worst_rmse"])
                run.metric("n_configurations", s["n_configurations"])
                run.metric("n_failed", s["n_failed"])
        summaries[name] = s
        b = s["best"]
        log.info("== %s best mean RMSE %.4f (channels=%s, top5 spread %.4f, %d failed)",
                 name, b["mean_rmse"] if b else float("nan"),
                 b["channels"] if b else "-", s["top5_spread"], s["n_failed"])

    if a.log_registry and not a.smoke and (a.shards == 1 or a.merge or a.fold_merge) \
            and a.config_index is None:
        from sentinel.core import registry
        for name, s in summaries.items():
            if not s["best"]:
                continue
            for metric, value in (("rmse", s["best"]["mean_rmse"]),
                                  ("rmse_worst_fold", s["best"]["worst_rmse"])):
                registry.log_result(
                    workstream="M4 ChemTox", experiment=EXPERIMENT, model=name,
                    metric=metric, value=value,
                    split="unseen_taxa/validation_rotation", target="log10_lc50_mol_L",
                    is_baseline=(s["role"] == "comparator"),
                    config={"channels": s["best"]["channels"], "params": s["best"]["params"]},
                    artifacts=[str(out_dir / f"{name}.json")],
                    notes="stage-2 selection candidate (validation only; no test row read)",
                    extra={"n_configurations": s["n_configurations"],
                           "top5_spread": s["top5_spread"], "folds": s["folds"],
                           "role": s["role"], "plan_sha256": pins["plan_sha256"],
                           "fold_scheme": "A5 class-only, out-of-class Channel A",
                           "per_fold": {f["fold"]: {"rmse": f["rmse"],
                                                    "surrogate_available_val":
                                                        f.get("surrogate_available_val")}
                                        for f in s["best"]["per_fold"]},
                           "merged_from": s.get("merged_from"),
                           "published_gpboost_adaptations":
                               (list(PUBLISHED_GPBOOST_ADAPTATIONS)
                                if SEARCHED[name].get("published_gpboost") else None),
                           "selection_breadth": PUBLISHED_SELECTION_BREADTH.get(name),
                           "stage": "selection_candidate"})

    idx_path = out_dir / "summary_index.json"
    try:
        prior = json.loads(idx_path.read_text()).get("entries", {}) if idx_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        prior = {}
    if (a.shards == 1 or a.merge or a.fold_merge) and a.config_index is None:
        prior.update({k: {kk: vv for kk, vv in v.items() if kk != "configs"}
                      for k, v in summaries.items()})
    idx_path.write_text(json.dumps(
        {"entries": prior,
         "aliases": alias_rows(),
         "published_gpboost_adaptations": PUBLISHED_GPBOOST_ADAPTATIONS,
         "plan_sha256": pins["plan_sha256"]},
        indent=2, default=str))
    print(json.dumps({k: (v["best"]["mean_rmse"] if v["best"] else None)
                      for k, v in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
