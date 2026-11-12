"""Arms and comparators for the M4 arch-v2 tracks T1 and T6: the entry registry and the
non-neural comparators.

New module, no I/O on project rows. ``configs/chemtox/arch_v2.yaml`` is the single source of truth
for the entry inventory, the caps and the closed condition-(b) list; this module loads it,
validates it against the counts ruling §5.5 fixes, and builds the comparators that need no new
research code. It imports ``arch_v2`` only for frozen constants.

Governing text in precedence order: ``chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``),
``chemtox_arch_v2_addendum_01.md``, the stamped amendment.

What is here
------------
* :func:`load_config`, :func:`entries`, :func:`validate_caps` — the inventory and its arithmetic,
  so a drift in the declared 384 / 128 / 26 / 12 / 550 fails a test rather than a review.
* :func:`duration_variant` — **ruling F10**: a ``*_dur`` entry is its parent's configuration with
  ``duration_h`` added **and nothing else**. Implemented as a column-list transform so "nothing
  else" is checkable.
* The fixed baselines of the **closed F3 list** that need no new modelling:
  :class:`GlobalMean`, :class:`GroupMean` (per species, per chemical, per species x band),
  :class:`HaberRule`, :class:`NearestDurationCarryForward`.
* :func:`monotone_duration_gbm` (**R1**), :func:`aft_gbm` (**R2**),
  :func:`multitask_stack_gbm` (**R3**) — the mandatory references, as constructors over already
  loaded frames.
* :func:`assert_parity` — the mechanical parity check of amendment §4.1: identical row counts,
  identical sorted feature names and their hash, identical auxiliary-target sets.

What is NOT here yet
--------------------
The learned arms ``ted``, ``timehead``, ``mmoe``, ``rx_chemprop`` and the embedding block. They are
declared in the config and in :data:`NEURAL_ARMS`; :func:`build_arm` raises
:class:`ArmNotImplemented` for them rather than returning a stub, so nothing can accidentally be
reported as having run. They arrive in their own modules with their own synthetic tests.

Nothing here may be fitted on pool rows before the lead writes the pins file (ruling S2): the rows
come from ``arch_v2_guard.load_training_pool()``, which refuses until then. Every constructor below
takes frames the caller has already obtained through the guard.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from sentinel.models.chemtox import arch_v2 as A

__all__ = ["ArmError", "ArmNotImplemented", "CONFIG_PATH", "NEURAL_ARMS", "TARGET",
           "load_config", "entries", "entry_roles", "validate_caps", "condition_b_list",
           "duration_variant", "feature_fingerprint", "assert_parity",
           "GlobalMean", "GroupMean", "HaberRule", "NearestDurationCarryForward",
           "monotone_duration_gbm", "aft_gbm", "multitask_stack_gbm", "build_arm"]

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "chemtox" / "arch_v2.yaml"
TARGET = "log10_conc_mol_L"
DURATION = "duration_h"

#: Declared in the config, implemented elsewhere. ``build_arm`` refuses them loudly.
NEURAL_ARMS = ("ted", "timehead", "mmoe", "rx_chemprop", "rx_lightgbm", "rx_catboost",
               "rx_lightgbm_dur", "xgboost_emb", "xgboost_emb_dur", "rf_emb", "rf_emb_dur",
               "lightgbm_noemb", "ensemble_mean", "ensemble_decorrelated",
               "ensemble_mean_baseline", "ensemble_decorrelated_baseline",
               "rf_published_adore", "xgboost_published_adore", "gpboost_published",
               "readacross", "r5_trident_released_errors")


class ArmError(ValueError):
    """A pre-registered rule about an arm or the inventory was violated."""


class ArmNotImplemented(NotImplementedError):
    """The arm is declared in the configuration but its code does not exist yet."""


# ------------------------------------------------------------------------- the inventory and caps

def load_config(path: Path | str | None = None) -> dict:
    """Load ``configs/chemtox/arch_v2.yaml``. The only file this module reads."""
    import yaml

    p = Path(path) if path is not None else CONFIG_PATH
    if not p.is_file():
        raise ArmError(f"no arch-v2 configuration at {p}")
    cfg = yaml.safe_load(p.read_text())
    if not isinstance(cfg, dict) or "entries" not in cfg or "caps" not in cfg:
        raise ArmError(f"{p} is not an arch-v2 configuration")
    return cfg


def entries(cfg: Mapping, track: str) -> dict[str, dict]:
    """Every entry of one track, searched and fixed, with its spec."""
    if track not in ("T1", "T6"):
        raise ArmError(f"track must be 'T1' or 'T6', got {track!r}")
    block = cfg["entries"][track]
    out: dict[str, dict] = {}
    for kind in ("searched", "fixed"):
        for name, spec in (block.get(kind) or {}).items():
            if name in out:
                raise ArmError(f"{track}: entry {name!r} appears in both searched and fixed")
            out[name] = {**spec, "kind": kind, "track": track, "entry": name}
    return out


def entry_roles(cfg: Mapping, track: str) -> dict[str, str]:
    return {n: s["role"] for n, s in entries(cfg, track).items()}


def validate_caps(cfg: Mapping | None = None) -> dict:
    """Re-derive every declared count from the entry table. Raises on the first disagreement.

    This is the drift check ruling §5.5 asks for: the five figures (384, 128, 26, 12, 550) and the
    per-entry caps (32 on T1, 16 on T6) are recomputed, not trusted.
    """
    cfg = cfg if cfg is not None else load_config()
    caps = cfg["caps"]
    got: dict = {}
    for track, per_entry in (("T1", 32), ("T6", 16)):
        es = entries(cfg, track)
        searched = {n: s for n, s in es.items() if s["kind"] == "searched"}
        fixed = {n: s for n, s in es.items() if s["kind"] == "fixed"}
        for n, s in searched.items():
            if int(s["configs"]) != per_entry:
                raise ArmError(
                    f"{track}/{n}: {s['configs']} configurations, but the per-entry cap is "
                    f"{per_entry} and no entry may exceed it (§5.5)")
        s_total = sum(int(s["configs"]) for s in searched.values())
        f_total = sum(int(s["configs"]) for s in fixed.values())
        declared = caps[track]
        if len(searched) != int(declared["searched_entries"]):
            raise ArmError(f"{track}: {len(searched)} searched entries, declared "
                           f"{declared['searched_entries']}")
        if s_total != int(declared["searched_total"]):
            raise ArmError(f"{track}: searched total {s_total}, declared "
                           f"{declared['searched_total']}")
        if f_total != int(declared["fixed_configurations"]):
            raise ArmError(f"{track}: fixed configurations {f_total}, declared "
                           f"{declared['fixed_configurations']}")
        got[track] = {"searched_entries": len(searched), "searched_total": s_total,
                      "fixed_configurations": f_total, "entries": len(es)}
    grand = (got["T1"]["searched_total"] + got["T6"]["searched_total"]
             + got["T1"]["fixed_configurations"] + got["T6"]["fixed_configurations"]
             + int(caps["ablations_fixed"]))
    if grand != int(caps["grand_total"]):
        raise ArmError(f"grand total {grand}, declared {caps['grand_total']}")
    n_entries = got["T1"]["entries"] + got["T6"]["entries"]
    if n_entries != int(caps["entries_total"]):
        raise ArmError(f"{n_entries} entries, declared {caps['entries_total']}")
    got["grand_total"] = grand
    got["entries_total"] = n_entries
    got["ablations_fixed"] = int(caps["ablations_fixed"])
    return got


def condition_b_list(cfg: Mapping | None = None) -> list[str]:
    """The CLOSED condition-(b) list of ruling F3, checked against the entry tables.

    Every name on the list must be a declared entry on at least one track, and no candidate or
    published-form arm may appear on it.
    """
    cfg = cfg if cfg is not None else load_config()
    listed = list(cfg["condition_b"]["closed_list"])
    known: dict[str, set[str]] = {}
    for track in ("T1", "T6"):
        for n, s in entries(cfg, track).items():
            known.setdefault(n, set()).add(s["role"] if not s.get("published_arm")
                                           else "published_arm")
    unknown = [n for n in listed if n not in known]
    if unknown:
        raise ArmError(f"condition (b) names entries that are not declared: {unknown}")
    bad = [n for n in listed if known[n] & {"candidate", "published_arm", "cross_reference"}]
    if bad:
        raise ArmError(
            f"condition (b) must exclude candidates and published-form arms (F3); found {bad}")
    if len(set(listed)) != len(listed):
        raise ArmError("the condition (b) list has duplicates")
    return listed


# --------------------------------------------------------------------------------- F10  *_dur

def duration_variant(parent_columns: Iterable[str], *, duration_column: str = DURATION
                     ) -> list[str]:
    """Ruling F10: the parent's columns **plus the duration feature, and nothing else**.

    Returns the parent's column list with ``duration_column`` appended if absent. Raises if the
    parent already carries it, because then the variant would not differ from its parent and the
    entry would be a duplicate configuration inside the cap.
    """
    cols = list(parent_columns)
    if duration_column in cols:
        raise ArmError(
            f"the parent already has {duration_column!r}: a *_dur variant would be identical to "
            f"its parent, which would be a duplicate configuration inside the cap (F10)")
    return cols + [duration_column]


def feature_fingerprint(columns: Iterable[str]) -> dict:
    """Sorted feature names and their hash: the parity record of amendment §4.1."""
    cols = sorted(str(c) for c in columns)
    h = hashlib.sha256("\n".join(cols).encode()).hexdigest()
    return {"n_features": len(cols), "feature_sha256": h, "features": cols}


def assert_parity(records: Mapping[str, Mapping]) -> dict:
    """The mechanical parity check of §4.1, over one fold.

    ``records`` maps an entry name to ``{"n_rows", "features", "aux_targets"}``. Every entry that
    condition (g) is read against must agree on all three; a disagreement raises. Published-form
    arms are exempt from the feature check by ruling §4.3 (they stay verbatim without the
    embedding) and must declare ``{"published_form": True}``.
    """
    if not records:
        raise ArmError("assert_parity needs at least two entries")
    parity_set = {n: r for n, r in records.items() if not r.get("published_form")}
    if len(parity_set) < 2:
        raise ArmError("fewer than two parity-bound entries supplied")
    rows = {n: int(r["n_rows"]) for n, r in parity_set.items()}
    if len(set(rows.values())) != 1:
        raise ArmError(f"row counts differ across parity-bound entries: {rows}")
    fps = {n: feature_fingerprint(r["features"])["feature_sha256"] for n, r in parity_set.items()}
    if len(set(fps.values())) != 1:
        differing = sorted({v[:12] for v in fps.values()})
        raise ArmError(f"feature sets differ across parity-bound entries: {differing} "
                       f"({ {n: v[:12] for n, v in fps.items()} })")
    aux = {n: tuple(sorted(r.get("aux_targets") or ())) for n, r in parity_set.items()}
    if len(set(aux.values())) != 1:
        raise ArmError(f"auxiliary-target sets differ across parity-bound entries: {aux}")
    return {"n_rows": next(iter(rows.values())),
            "feature_sha256": next(iter(fps.values())),
            "aux_targets": list(next(iter(aux.values()))),
            "entries_checked": sorted(parity_set),
            "published_form_exempt": sorted(set(records) - set(parity_set))}


# ------------------------------------------------------------- the fixed baselines of the F3 list

@dataclass
class GlobalMean:
    """``trivial_global_mean``: predict the training mean everywhere."""

    value_: float | None = None

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "GlobalMean":
        v = np.asarray(list(y), dtype=float)
        if v.size == 0:
            raise ArmError("GlobalMean.fit on no rows")
        self.value_ = float(np.mean(v))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.value_ is None:
            raise ArmError("GlobalMean is not fitted")
        return np.full(len(X), self.value_, dtype=float)


@dataclass
class GroupMean:
    """``trivial_per_species_mean`` / ``per_chemical_mean`` / ``per_species_x_band_mean``.

    Falls back to the global training mean for a group unseen in training — the declared fallback of
    amendment §4.5.
    """

    by: tuple[str, ...]
    table_: dict | None = None
    fallback_: float | None = None

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "GroupMean":
        missing = [c for c in self.by if c not in X.columns]
        if missing:
            raise ArmError(f"GroupMean needs columns {missing}")
        v = pd.Series(np.asarray(list(y), dtype=float), index=X.index)
        key = list(self.by)
        self.table_ = v.groupby([X[c].astype(str) for c in key]).mean().to_dict()
        self.fallback_ = float(v.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.table_ is None:
            raise ArmError("GroupMean is not fitted")
        keys = list(zip(*[X[c].astype(str) for c in self.by])) if len(self.by) > 1 \
            else list(X[self.by[0]].astype(str))
        return np.array([self.table_.get(k, self.fallback_) for k in keys], dtype=float)


@dataclass
class HaberRule:
    """``haber_rule``: C x t is constant, the crude regulatory rule T6 is motivated against.

    On the log10 scale ``log10 C = a - log10 t``, so the only fitted quantity is the intercept
    ``a``, estimated as the training mean of ``log10 C + log10 t``. Pure extrapolation in duration
    by construction, which is the point: it is the floor T6's candidates must beat.
    """

    intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "HaberRule":
        if DURATION not in X.columns:
            raise ArmError(f"HaberRule needs {DURATION!r}")
        t = pd.to_numeric(X[DURATION], errors="coerce").to_numpy(dtype=float)
        v = np.asarray(list(y), dtype=float)
        ok = np.isfinite(t) & (t > 0) & np.isfinite(v)
        if not ok.any():
            raise ArmError("HaberRule.fit: no row with a positive duration and a finite target")
        self.intercept_ = float(np.mean(v[ok] + np.log10(t[ok])))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.intercept_ is None:
            raise ArmError("HaberRule is not fitted")
        t = pd.to_numeric(X[DURATION], errors="coerce").to_numpy(dtype=float)
        t = np.where(np.isfinite(t) & (t > 0), t, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self.intercept_ - np.log10(t)
        return np.where(np.isfinite(out), out, self.intercept_)


@dataclass
class NearestDurationCarryForward:
    """``nearest_duration_carry_forward``: what a tree actually does past its last split.

    Per chemical x species, carry the training value at the nearest observed duration; fall back to
    the chemical's mean, then the global mean.
    """

    keys: tuple[str, ...] = ("test_cas", "latin_name")
    table_: dict | None = None
    by_chem_: dict | None = None
    fallback_: float | None = None

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "NearestDurationCarryForward":
        need = [*self.keys, DURATION]
        missing = [c for c in need if c not in X.columns]
        if missing:
            raise ArmError(f"NearestDurationCarryForward needs {missing}")
        v = np.asarray(list(y), dtype=float)
        df = X[need].copy()
        df["_y"] = v
        self.table_ = {}
        for k, g in df.groupby([df[c].astype(str) for c in self.keys]):
            t = pd.to_numeric(g[DURATION], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(t)
            if ok.any():
                self.table_[k if isinstance(k, tuple) else (k,)] = (t[ok], g["_y"].to_numpy()[ok])
        self.by_chem_ = df.groupby(df[self.keys[0]].astype(str))["_y"].mean().to_dict()
        self.fallback_ = float(np.mean(v))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.table_ is None:
            raise ArmError("NearestDurationCarryForward is not fitted")
        t_new = pd.to_numeric(X[DURATION], errors="coerce").to_numpy(dtype=float)
        out = np.empty(len(X), dtype=float)
        keys = list(zip(*[X[c].astype(str) for c in self.keys]))
        chem = list(X[self.keys[0]].astype(str))
        for i, k in enumerate(keys):
            hit = self.table_.get(k)
            if hit is not None and np.isfinite(t_new[i]):
                ts, ys = hit
                out[i] = ys[int(np.argmin(np.abs(ts - t_new[i])))]
            else:
                out[i] = self.by_chem_.get(chem[i], self.fallback_)
        return out


# ------------------------------------------------------------------ R1, R2, R3 — the references

def monotone_duration_gbm(feature_columns: Sequence[str], *, duration_column: str = DURATION,
                          direction: int = -1, **params):
    """**R1**: LightGBM with a monotone constraint on continuous duration.

    ``direction = -1`` encodes "log10 LC50 is non-increasing in exposure duration": longer exposure
    cannot raise the lethal concentration. The library is fixed to LightGBM by ruling OPEN-CHOICES.
    Returns an unfitted estimator; the caller supplies rows obtained through the guard.
    """
    from lightgbm import LGBMRegressor

    cols = list(feature_columns)
    if duration_column not in cols:
        raise ArmError(f"R1 needs {duration_column!r} among its features")
    if direction not in (-1, 1):
        raise ArmError("direction must be -1 or +1")
    constraints = [direction if c == duration_column else 0 for c in cols]
    defaults = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
    model = LGBMRegressor(monotone_constraints=constraints, **{**defaults, **params})
    model._sentinel_feature_columns = cols          # for the parity record
    model._sentinel_monotone = dict(zip(cols, constraints))
    return model


def aft_gbm(*, distribution: str = "normal", **params):
    """**R2**: XGBoost ``survival:aft`` for the interval-censored acute rows.

    Returns ``(booster_params, num_boost_round)``; the caller builds the ``DMatrix`` with
    ``label_lower_bound`` / ``label_upper_bound`` from the censored frame and calls
    ``xgboost.train``. Kept as parameters rather than an estimator because AFT has no sklearn
    wrapper that accepts interval bounds.
    """
    if distribution not in ("normal", "logistic", "extreme"):
        raise ArmError(f"unsupported AFT distribution {distribution!r}")
    p = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
         "aft_loss_distribution": distribution, "aft_loss_distribution_scale": 1.0,
         "tree_method": "hist", "learning_rate": 0.05, "max_depth": 6, "nthread": 6}
    p.update(params)
    return p, int(p.pop("num_boost_round", 300))


def multitask_stack_gbm(feature_columns: Sequence[str], *, endpoint_column: str = "endpoint_kind",
                        **params):
    """**R3**: one GBM over acute + chronic rows with an endpoint indicator and per-endpoint offsets.

    The indicator is a required feature, so a chronic row's gradient can only reach an acute
    prediction through shared *features* — which is exactly the limitation R3 exists to demonstrate.
    """
    from lightgbm import LGBMRegressor

    cols = list(feature_columns)
    if endpoint_column not in cols:
        raise ArmError(f"R3 needs the endpoint indicator {endpoint_column!r} among its features")
    defaults = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63, "verbose": -1}
    model = LGBMRegressor(**{**defaults, **params})
    model._sentinel_feature_columns = cols
    model._sentinel_endpoint_column = endpoint_column
    return model


_FIXED_BUILDERS = {
    "trivial_global_mean": lambda: GlobalMean(),
    "trivial_per_species_mean": lambda: GroupMean(by=("latin_name",)),
    "trivial_per_chemical_mean": lambda: GroupMean(by=("test_cas",)),
    "trivial_per_species_x_band_mean": lambda: GroupMean(by=("latin_name", "duration_band")),
    "haber_rule": lambda: HaberRule(),
    "nearest_duration_carry_forward": lambda: NearestDurationCarryForward(),
}


def build_arm(name: str, *, feature_columns: Sequence[str] | None = None, **params):
    """Construct a declared arm, or refuse loudly if its code does not exist yet."""
    if name in _FIXED_BUILDERS:
        return _FIXED_BUILDERS[name]()
    if name == "r1_monotone_gbm":
        if feature_columns is None:
            raise ArmError("R1 needs feature_columns")
        return monotone_duration_gbm(feature_columns, **params)
    if name == "r2_aft_gbm":
        return aft_gbm(**params)
    if name == "r3_multitask_stack_gbm":
        if feature_columns is None:
            raise ArmError("R3 needs feature_columns")
        return multitask_stack_gbm(feature_columns, **params)
    if name in NEURAL_ARMS:
        raise ArmNotImplemented(
            f"{name!r} is declared in configs/chemtox/arch_v2.yaml but its code does not exist "
            f"yet; it arrives in its own module with its own synthetic tests. Nothing may report "
            f"it as having run.")
    raise ArmError(f"unknown arm {name!r}")
