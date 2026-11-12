"""Statistics for the M4 arch-v2 looks: paired intervals, Holm, the MDE gate, outcome wording.

New module. It touches no file and no row: every function takes arrays or small dicts, so the whole
of rulings F5, F6, F7, F8 and F1 can be unit-tested on synthetic numbers without the guard being
open. ``arch_v2.py`` and ``arch_v2_power.py`` are byte-frozen by ruling B1 and are not imported for
anything but their constants.

Governing text, in precedence order: ``chemtox_arch_v2_lead_rulings.md`` (sha256 ``553be413…``),
``chemtox_arch_v2_addendum_01.md``, then the stamped amendment. The configuration is
``configs/chemtox/arch_v2.yaml``.

What is implemented, ruling by ruling
-------------------------------------
**F6** — the interval that carries (a) and (b) is the **seed-mean paired difference**: the mean over
seeds 0, 1, 2 of the candidate minus the mean over the same seeds of the comparator, on identical
rows, seed *s* paired with seed *s*. Two-sided 95 % percentile cluster bootstrap. The Holm p-value
of a leg is ``2 * min(share <= 0, share >= 0)`` with each share computed as ``(count + 1)/(B + 1)``.
``B`` is **10,000** for every interval that enters a Holm family and is never lowered.
``all_seeds_better`` is TRUE only if the point difference favours the candidate for **each** of
seeds 0, 1 and 2 against **each** of the (a) and (b) comparators, on the full block.

**F5** — legs (a) and (b) must exclude 0 under **both** clusterings (``holdout_unit`` and
``reference_number``). Condition (h) additionally requires the difference to keep its sign on the
``no shared reference`` stratum and to exceed that stratum's own MDE; a candidate that passes
everything else and fails (h) carries no label and is reported as *"ahead on the full block, not
separable from publication structure"*.

**F7** — the governing MDE is the **analytic** value; the bootstrap figure is never a gate, and
every write-up that quotes an MDE states both figures and the unverified-coverage caveat.

**F1** — no §15.1 label is available on T1_test, T6a_test or T6b_test. Conditions (a)-(j) are still
computed and reported in full as the pre-registered descriptive reading, and
:func:`outcome_label` returns the fixed F8 wording — never a SOTA label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["B_HOLM_FAMILY", "SEEDS", "CLUSTERINGS", "DESCRIPTIVE_ONLY_BLOCKS", "OUTCOME",
           "GOVERNING_MDE_ICC_0_5", "NO_SHARED_REFERENCE_MDE_ICC_0_5", "ScorerError",
           "micro_rmse", "seed_mean_difference", "paired_cluster_bootstrap", "holm_p_value",
           "holm_adjust", "all_seeds_better", "mde_gate", "LegResult", "evaluate_leg",
           "outcome_label", "describe_mde"]

#: F6: never lowered.
B_HOLM_FAMILY = 10_000

#: F6: the three seeds, and the pairing is by index.
SEEDS = (0, 1, 2)

#: F5: both are required for legs (a) and (b).
CLUSTERINGS = ("holdout_unit", "reference_number")

#: F1: all three blocks are descriptive only; no route to a §15.1 label exists.
DESCRIPTIVE_ONLY_BLOCKS = ("T1_test", "T6a_test", "T6b_test")

#: F10 — the pinned ``t1t6_mde_v2.json`` governs wherever these differ. Analytic, icc 0.5, rho 0.9,
#: the stricter of the full block and its in-domain subset.
GOVERNING_MDE_ICC_0_5 = {"T1_test": 0.0959, "T6a_test": 0.1027, "T6b_test": 0.1033}
NO_SHARED_REFERENCE_MDE_ICC_0_5 = {"T1_test": 0.1255, "T6a_test": 0.1061, "T6b_test": 0.1039}

#: F8 — the outcome wording, fixed before any look.
OUTCOME = {
    "all_met": "exceeds the pre-registered MDE on an under-powered block; no SOTA label",
    "ahead_unmet": "not resolvable on this block",
    "behind": "no gain",
    "behind_significant": "worse than {comparator}",
    "shared_reference": "ahead on the full block, not separable from publication structure",
    "none_ahead": ("no candidate is ahead of the (a) comparator on any block: RESEARCH_PLAN §15.5 "
                   "applies and the next track must be harder or larger"),
}


class ScorerError(ValueError):
    """A pre-registered statistical rule was violated by the caller."""


# ------------------------------------------------------------------------------------- the metric

def micro_rmse(residual: Iterable[float]) -> float:
    """Micro-RMSE on a block's rows: the primary metric (PRIMARY-METRIC-AND-CI)."""
    r = np.asarray(list(residual), dtype=float)
    if r.size == 0:
        raise ScorerError("micro_rmse on an empty residual vector")
    if not np.isfinite(r).all():
        raise ScorerError("micro_rmse: non-finite residual")
    return float(np.sqrt(np.mean(r * r)))


def _stack(per_seed: Mapping[int, Sequence[float]], *, what: str) -> np.ndarray:
    """Validate a seed -> residual-vector mapping and return it as a (3, n) array."""
    missing = [s for s in SEEDS if s not in per_seed]
    if missing:
        raise ScorerError(f"{what}: missing seeds {missing}; F6 requires seeds {list(SEEDS)}")
    extra = [s for s in per_seed if s not in SEEDS]
    if extra:
        raise ScorerError(f"{what}: unexpected seeds {extra}")
    rows = [np.asarray(list(per_seed[s]), dtype=float) for s in SEEDS]
    n = {r.size for r in rows}
    if len(n) != 1:
        raise ScorerError(f"{what}: seeds disagree on row count {sorted(n)}")
    out = np.vstack(rows)
    if not np.isfinite(out).all():
        raise ScorerError(f"{what}: non-finite residual")
    return out


# ---------------------------------------------------------------------------------- F6  the point

def seed_mean_difference(cand: Mapping[int, Sequence[float]],
                         comp: Mapping[int, Sequence[float]]) -> float:
    """Seed-mean paired difference: mean-over-seeds RMSE(candidate) - mean-over-seeds RMSE(comparator).

    Negative favours the candidate, since lower micro-RMSE is better. Seed *s* of the candidate is
    paired with seed *s* of the comparator (F6), and both must cover the same rows.
    """
    a = _stack(cand, what="candidate")
    b = _stack(comp, what="comparator")
    if a.shape != b.shape:
        raise ScorerError(
            f"candidate and comparator must be scored on identical rows; got {a.shape} and "
            f"{b.shape} (F6: paired on identical rows)")
    return float(np.mean([micro_rmse(a[i]) for i in range(a.shape[0])])
                 - np.mean([micro_rmse(b[i]) for i in range(b.shape[0])]))


def all_seeds_better(cand: Mapping[int, Sequence[float]],
                     comps: Mapping[str, Mapping[int, Sequence[float]]]) -> bool:
    """TRUE only if the candidate wins on EVERY seed against EVERY given comparator (F6).

    ``comps`` maps a comparator name to its own seed -> residual mapping; the (a) and (b)
    comparators are both passed, and both must be beaten on all three seeds.
    """
    if not comps:
        raise ScorerError("all_seeds_better needs at least the (a) and (b) comparators")
    a = _stack(cand, what="candidate")
    for name, comp in comps.items():
        b = _stack(comp, what=f"comparator {name}")
        if a.shape != b.shape:
            raise ScorerError(f"comparator {name}: row count differs from the candidate's")
        for i in range(a.shape[0]):
            if not micro_rmse(a[i]) < micro_rmse(b[i]):
                return False
    return True


# ------------------------------------------------------------------- F6  the bootstrap and Holm p

def paired_cluster_bootstrap(cand: Mapping[int, Sequence[float]],
                             comp: Mapping[int, Sequence[float]],
                             cluster: Sequence, *, B: int = B_HOLM_FAMILY, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Two-sided percentile cluster bootstrap of the seed-mean paired difference.

    Clusters are resampled with replacement; within a resample the same rows are used for the
    candidate and the comparator, for every seed (F6: paired on identical rows).
    """
    a = _stack(cand, what="candidate")
    b = _stack(comp, what="comparator")
    if a.shape != b.shape:
        raise ScorerError("candidate and comparator must cover identical rows")
    g = np.asarray(list(cluster))
    if g.size != a.shape[1]:
        raise ScorerError(f"cluster vector has {g.size} entries for {a.shape[1]} rows")
    if B < B_HOLM_FAMILY:
        raise ScorerError(
            f"B = {B} is below the pre-registered {B_HOLM_FAMILY} for an interval entering a Holm "
            f"family; F6 forbids lowering it")
    if not 0.0 < alpha < 1.0:
        raise ScorerError(f"alpha must be in (0, 1), got {alpha}")

    uniq, inv = np.unique(g, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(uniq.size)]
    rng = np.random.default_rng(seed)
    sqa, sqb = a * a, b * b
    draws = rng.integers(0, uniq.size, size=(B, uniq.size))
    diffs = np.empty(B)
    for k in range(B):
        take = np.concatenate([idx[i] for i in draws[k]])
        ra = np.sqrt(sqa[:, take].mean(axis=1)).mean()
        rb = np.sqrt(sqb[:, take].mean(axis=1)).mean()
        diffs[k] = ra - rb
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    point = seed_mean_difference(cand, comp)
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "favours_candidate": bool(point < 0.0),
            "B": int(B), "n_clusters": int(uniq.size), "n_rows": int(a.shape[1]),
            "seeds": list(SEEDS), "alpha": alpha,
            "p_value": holm_p_value(diffs, B=B),
            "method": "two-sided 95% percentile cluster bootstrap of the seed-mean paired "
                      "difference, paired on identical rows (F6)"}


def holm_p_value(diffs: Iterable[float], *, B: int | None = None) -> float:
    """``2 * min(share <= 0, share >= 0)``, each share as ``(count + 1)/(B + 1)`` (F6).

    Capped at 1.0, since twice a share can exceed 1 when the distribution straddles 0.
    """
    d = np.asarray(list(diffs), dtype=float)
    if d.size == 0:
        raise ScorerError("holm_p_value on an empty bootstrap distribution")
    n = int(B) if B is not None else int(d.size)
    lo = (int(np.sum(d <= 0.0)) + 1) / (n + 1)
    hi = (int(np.sum(d >= 0.0)) + 1) / (n + 1)
    return float(min(1.0, 2.0 * min(lo, hi)))


def holm_adjust(p_by_leg: Mapping[str, float], *, m: int | None = None) -> dict[str, float]:
    """Holm step-down adjustment over a family whose size ``m`` is fixed in writing (F6, §8.3).

    ``m`` defaults to the number of legs passed, but the pre-registered family size is larger than
    the number of legs a single candidate contributes (T1: m = 14, T6: m = 6 per block), so the
    caller passes it explicitly and the adjusted values are monotone non-decreasing.
    """
    if not p_by_leg:
        raise ScorerError("holm_adjust on an empty family")
    size = int(m) if m is not None else len(p_by_leg)
    if size < len(p_by_leg):
        raise ScorerError(f"m = {size} is smaller than the {len(p_by_leg)} legs supplied")
    order = sorted(p_by_leg, key=lambda k: p_by_leg[k])
    out, running = {}, 0.0
    for rank, key in enumerate(order):
        adj = min(1.0, (size - rank) * p_by_leg[key])
        running = max(running, adj)
        out[key] = running
    return out


# --------------------------------------------------------------------------------- F7  the gate

def mde_gate(block: str, difference: float, *, stratum: str = "all",
             mde: Mapping[str, float] | None = None) -> dict:
    """Does the point difference exceed the block's governing ANALYTIC MDE? (F7, condition (f))

    ``difference`` is the seed-mean paired difference; its magnitude is compared, and the sign is
    reported separately. The bootstrap figure is never consulted here.
    """
    table = dict(mde) if mde is not None else (
        GOVERNING_MDE_ICC_0_5 if stratum == "all" else NO_SHARED_REFERENCE_MDE_ICC_0_5)
    if block not in table:
        raise ScorerError(f"no governing MDE for block {block!r} (stratum {stratum!r})")
    threshold = float(table[block])
    return {"block": block, "stratum": stratum, "governing_mde": threshold,
            "difference": float(difference),
            "magnitude_exceeds_mde": bool(abs(float(difference)) > threshold),
            "favours_candidate": bool(float(difference) < 0.0),
            "source": "analytic value of amendment §6.2; the bootstrap check is not a gate (F7)"}


def describe_mde(block: str, *, analytic: float | None = None,
                 bootstrap: float | None = None) -> str:
    """The sentence F7 requires wherever an MDE is quoted."""
    a = GOVERNING_MDE_ICC_0_5.get(block) if analytic is None else analytic
    b = "not quoted as a gate" if bootstrap is None else f"{bootstrap:.4f}"
    return (f"{block}: governing analytic MDE {a:.4f} at icc 0.5, rho 0.9 (the gate); bootstrap "
            f"check {b}. The percentile bootstrap's coverage on heavy-tailed unit sizes has not "
            f"been verified (F7).")


# ------------------------------------------------------------------------ a leg, and the wording

@dataclass
class LegResult:
    """One comparator leg of condition (a) or (b), under both clusterings (F5)."""

    leg: str
    comparator: str
    point: float
    by_clustering: dict = field(default_factory=dict)

    @property
    def excludes_zero_under_both(self) -> bool:
        if set(self.by_clustering) != set(CLUSTERINGS):
            return False
        return all(v["excludes_zero"] for v in self.by_clustering.values())

    @property
    def favours_candidate(self) -> bool:
        return self.point < 0.0

    def as_dict(self) -> dict:
        return {"leg": self.leg, "comparator": self.comparator, "point": self.point,
                "excludes_zero_under_both_clusterings": self.excludes_zero_under_both,
                "favours_candidate": self.favours_candidate,
                "by_clustering": self.by_clustering,
                "p_values": {k: v.get("p_value") for k, v in self.by_clustering.items()}}


def evaluate_leg(leg: str, comparator: str, cand: Mapping[int, Sequence[float]],
                 comp: Mapping[int, Sequence[float]],
                 clusters: Mapping[str, Sequence], *, B: int = B_HOLM_FAMILY,
                 seed: int = 0) -> LegResult:
    """Run the paired bootstrap under BOTH pre-registered clusterings (F5)."""
    if leg not in ("a", "b"):
        raise ScorerError(f"leg must be 'a' or 'b', got {leg!r}")
    missing = [c for c in CLUSTERINGS if c not in clusters]
    if missing:
        raise ScorerError(f"leg {leg}: missing clusterings {missing}; F5 requires both")
    by = {c: paired_cluster_bootstrap(cand, comp, clusters[c], B=B, seed=seed)
          for c in CLUSTERINGS}
    return LegResult(leg=leg, comparator=comparator,
                     point=seed_mean_difference(cand, comp), by_clustering=by)


def outcome_label(block: str, *, conditions: Mapping[str, bool], point: float,
                  comparator: str, behind_ci_excludes_zero: bool = False) -> dict:
    """The F8 wording. Never returns a §15.1 label: under F1 these blocks cannot carry one.

    ``conditions`` maps the condition letters ("a".."j") to whether each is met.
    """
    if block not in DESCRIPTIVE_ONLY_BLOCKS:
        raise ScorerError(
            f"{block!r} is not one of the three descriptive-only blocks {DESCRIPTIVE_ONLY_BLOCKS}; "
            f"a different block needs its own stamped track (lead point (f))")
    unmet = sorted(k for k, v in conditions.items() if not v)
    favours = point < 0.0
    if not favours:
        wording = (OUTCOME["behind_significant"].format(comparator=comparator)
                   if behind_ci_excludes_zero else OUTCOME["behind"])
        kind = "behind_significant" if behind_ci_excludes_zero else "behind"
    elif unmet == ["h"]:
        wording, kind = OUTCOME["shared_reference"], "shared_reference"
    elif unmet:
        wording, kind = OUTCOME["ahead_unmet"], "ahead_unmet"
    else:
        wording, kind = OUTCOME["all_met"], "all_met"
    return {"block": block, "kind": kind, "wording": wording, "point": float(point),
            "conditions_unmet": unmet, "sota_label": False,
            "label_available": False,
            "why_no_label": ("F1: T1_test, T6a_test and T6b_test are descriptive only; a powered "
                             "claim needs a new, separately stamped track (lead point (f))"),
            "equal_prominence_required": True}
