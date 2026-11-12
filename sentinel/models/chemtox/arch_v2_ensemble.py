"""The arch-v2 ensembles and their MATCHED BASELINE controls.

New module, no project row read: every function takes out-of-fold prediction vectors the caller
already produced from pool rows obtained through the guard.

Four entries, all fixed-configuration (amendment §5.2):

* ``ensemble_mean`` — the unweighted mean of the frozen candidate winners;
* ``ensemble_decorrelated`` — members chosen greedily to maximise
  ``mean pairwise (1 - corr(residuals))`` subject to each member being within
  :data:`MEMBER_MARGIN` of the best single model, the set fixed **before** the look;
* ``ensemble_mean_baseline`` and ``ensemble_decorrelated_baseline`` — the same two rules applied to
  the **strongest simple baselines**, with the same member count and the same margin.

The matched baseline controls exist because amendment §4.1 binds the project to ensembling parity
and the first text enumerated no control: without them condition (g) would read an ensemble
candidate against unensembled baselines, which is exactly the C-M4P-1 failure. **No §15.1 label is
available for an ensemble candidate on a block where the matched baseline ensemble was not built
and reported** — and on T6 no ensemble entry is declared at all, so no ensemble claim exists there
(§5.3). :func:`require_matched_control` enforces both.

Disclosed, per the stamped amendment: the 0.05 margin is about five times T1's row-level MDE and
about half its governing unit-level gate, so a member may differ from the best single model by more
than a claimable difference. The margin stays fixed at 0.05 and is reported beside every ensemble
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = ["EnsembleError", "MEMBER_MARGIN", "NO_ENSEMBLE_TRACKS", "MARGIN_DISCLOSURE",
           "rmse", "residual_correlation", "mean_ensemble", "select_decorrelated",
           "require_matched_control"]


class EnsembleError(ValueError):
    """A pre-registered rule about the ensembles was violated."""


#: Fixed before any look and never changed.
MEMBER_MARGIN = 0.05

#: §5.3: T6 declares no ensemble entry, so no ensemble claim is available there.
NO_ENSEMBLE_TRACKS = ("T6",)

MARGIN_DISCLOSURE = (
    "The 0.05 ensemble-membership margin is about five times T1's row-level MDE (0.0097) and about "
    "half its governing unit-level gate (0.0959), so members may differ from the best single model "
    "by more than a claimable difference at the row-level reading. The margin is fixed at 0.05 and "
    "is reported beside every ensemble result.")


def rmse(residual: Sequence[float]) -> float:
    r = np.asarray(list(residual), dtype=float)
    if r.size == 0:
        raise EnsembleError("rmse on an empty residual vector")
    if not np.isfinite(r).all():
        raise EnsembleError("rmse: non-finite residual")
    return float(np.sqrt(np.mean(r * r)))


def _matrix(residuals: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    if len(residuals) < 2:
        raise EnsembleError("an ensemble needs at least two members")
    names = sorted(residuals)
    rows = [np.asarray(list(residuals[n]), dtype=float) for n in names]
    sizes = {r.size for r in rows}
    if len(sizes) != 1:
        raise EnsembleError(f"members disagree on row count {sorted(sizes)}")
    m = np.vstack(rows)
    if not np.isfinite(m).all():
        raise EnsembleError("non-finite residual in a member")
    return names, m


def residual_correlation(residuals: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Pairwise Pearson correlation of the members' residuals. Zero variance -> correlation 0."""
    names, m = _matrix(residuals)
    sd = m.std(axis=1)
    out = np.zeros((len(names), len(names)), dtype=float)
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                out[i, j] = 1.0
            elif sd[i] == 0 or sd[j] == 0:
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.corrcoef(m[i], m[j])[0, 1])
    return out


def mean_ensemble(predictions: Mapping[str, Sequence[float]]) -> np.ndarray:
    """``ensemble_mean``: the unweighted mean of the members' predictions."""
    names, m = _matrix(predictions)
    return m.mean(axis=0)


def select_decorrelated(residuals: Mapping[str, Sequence[float]], *,
                        margin: float = MEMBER_MARGIN,
                        max_members: int | None = None) -> dict:
    """Greedy decorrelation selection, subject to the eligibility margin.

    Eligible members are those within ``margin`` RMSE of the best single model. The first member is
    the best single model; each further member is the eligible candidate that maximises the mean
    pairwise ``1 - corr(residuals)`` of the resulting set. Deterministic: ties break on the name, so
    the member set is reproducible and can be fixed before the look.
    """
    if margin < 0:
        raise EnsembleError("margin must be non-negative")
    names, m = _matrix(residuals)
    scores = {n: rmse(m[i]) for i, n in enumerate(names)}
    best = min(scores, key=lambda n: (scores[n], n))
    eligible = sorted(n for n in names if scores[n] <= scores[best] + margin)
    corr = residual_correlation(residuals)
    index = {n: i for i, n in enumerate(names)}
    chosen = [best]
    cap = len(eligible) if max_members is None else max(1, int(max_members))
    while len(chosen) < cap:
        rest = [n for n in eligible if n not in chosen]
        if not rest:
            break
        def objective(cand: str) -> tuple[float, str]:
            members = chosen + [cand]
            pairs = [1.0 - corr[index[a], index[b]]
                     for i, a in enumerate(members) for b in members[i + 1:]]
            return (-float(np.mean(pairs)) if pairs else 0.0, cand)
        chosen.append(min(rest, key=objective))
    members = sorted(chosen)
    pairs = [1.0 - corr[index[a], index[b]]
             for i, a in enumerate(members) for b in members[i + 1:]]
    return {"members": members, "n_members": len(members),
            "eligible": eligible, "excluded_by_margin": sorted(set(names) - set(eligible)),
            "best_single": best, "best_single_rmse": scores[best],
            "margin": float(margin),
            "mean_pairwise_decorrelation": float(np.mean(pairs)) if pairs else 0.0,
            "member_rmse": {n: scores[n] for n in members},
            "disclosure": MARGIN_DISCLOSURE}


def require_matched_control(track: str, *, candidate_members: Sequence[str],
                            baseline_members: Sequence[str] | None) -> dict:
    """Ensembling parity: the matched baseline ensemble must exist, with the same member count.

    Raises on T6, where no ensemble entry is declared, and whenever the control is missing or has a
    different member count.
    """
    if track in NO_ENSEMBLE_TRACKS:
        raise EnsembleError(
            f"no ensemble entry is declared on {track} (§5.3), so no ensemble claim is available "
            f"there; the clause has nothing to attach to")
    if not candidate_members:
        raise EnsembleError("the candidate ensemble has no members")
    if not baseline_members:
        raise EnsembleError(
            "the matched baseline ensemble was not built: condition (g) would read an ensemble "
            "candidate against unensembled baselines, which is the C-M4P-1 failure. No label is "
            "available for an ensemble candidate on this block.")
    if len(baseline_members) != len(candidate_members):
        raise EnsembleError(
            f"member counts differ: candidate {len(candidate_members)}, matched baseline "
            f"{len(baseline_members)}. Ensembling parity requires the same member count.")
    return {"track": track, "n_members": len(candidate_members),
            "candidate_members": sorted(candidate_members),
            "baseline_members": sorted(baseline_members),
            "label_available_for_ensemble": False,
            "why": "F1: all three blocks are descriptive only",
            "disclosure": MARGIN_DISCLOSURE}
