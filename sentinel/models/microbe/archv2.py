# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Architecture round v2 of SENTINEL-Microbe: one place for the model list, the gates and the out-of-fold loop.

Governing texts: ``docs/preregistration/microbe_arch_v2_amendment.md`` (stamped 2026-11-10) and
``docs/preregistration/microbe_arch_v2_lead_rulings.md`` (wins where they differ).

* ``FAMILIES``: P = the 8 primary targets, G = GEO-8.
* :func:`attach_master` adds the GEO-8 and auxiliary columns of the DEVELOPMENT target table to a development
  design (the protected label file is never opened here; a protected id raises).
* :func:`gate` is the single admission check of every development run of this round: amendment stamp, released
  scheme, family rule of the +QMP twins, and the deviations-file release for everything the stamp did not release.
* :func:`fit_predict` / :func:`oof` dispatch to ``niche``, ``wa``, ``v4mr`` and ``baselines``.
* :func:`calibration` is the diagnostic of lead ruling A1 (reading in ``microbe_arch_v2_deviations_01.md``).
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from sentinel.data.microbe_geo import GEO_TARGETS
from sentinel.models.microbe import baselines as B
from sentinel.models.microbe import niche as NI
from sentinel.models.microbe import stamp as ST
from sentinel.models.microbe import v4mr as VM
from sentinel.models.microbe import wa as WA
from sentinel.models.microbe.design import TARGET_NAMES

FAMILIES = {"P": tuple(TARGET_NAMES), "G": tuple(GEO_TARGETS)}
RELEASED_SCHEMES = ("lodo",)                 # lead stamp message 2026-11-10: step 1 is released on lodo only
EXPERIMENT = "microbe_arch_v2_cv"
MANIFESTS = ["microbe_wq", "microbe_wq_geo"]

# Counted slots (cap 8; #5 retired). Slot #8 is conditional on the lead's written D1-gate evaluation.
SLOTS = {"niche_s": 6, "xgb_v4mr": 7, "xgb_v4mr_stack": 8}
# Closed ablation / control list (rulings B8, A1): counted, lodo only, never selectable.
ABLATIONS = {"niche_s_a0": "A0", "niche_s_tax": "A1"}
CONTROLS = ("xgb_nichez_context", "xgb_context_qmp")
# GEO entries G1..G9 on lodo (G5 = the slot-#8 model, G10 = one +QMP twin chosen mechanically).
GEO_ENTRIES = {"G1": "niche_s", "G3": "xgb_v4mr", "G5": None, "G7": "xgb_v4clr", "G8": "xgb_v4clr_context",
               "G9": "ens_ridge_v4clr+xgb_v4clr"}
# Baseline tables of ``baselines.py`` that this round may run (on family G; family P exists on the old track).
BASE_TABLES = ("global_mean", "context_ridge", "context_rf", "ridge_clr", "rf_clr", "xgb_clr", "rf_clr_tuned",
               "xgb_clr_tuned", "tabpfn_clr", "rf_clr_context", "xgb_clr_context", "ridge_v4clr", "xgb_v4clr",
               "xgb_v4clr_context")
# Names that need a lead-stamped deviations file (stamp.RELEASE_KEYS) before any development number.
RELEASE_KEY = {"niche_s_a0": "A0", "xgb_v4clr_mo": "XGB_V4CLR_MO", "niche_s_qmp": "NICHE_S_QMP",
               "niche_s_doy": "EA", "niche_geo_phys": "D2", "niche_geo_phys_a2": "D2", "niche_geo_phys_a3": "D2",
               "xgb_chained_physics": "D2"}
NOT_ELIGIBLE_HM3 = ("xgb_v4mr", "xgb_v4mr_stack")     # ruling B12: taxonomy labels, not eligible as M* for H-M3


class GateError(RuntimeError):
    """A development run of architecture round v2 that the stamp or the rulings do not allow."""


# ------------------------------------------------------------------------------------------- model registry

def backend(name: str) -> str:
    if name in NI.NAMES:
        return "niche"
    if name in WA.NAMES:
        return "wa"
    if name in VM.ALL:
        return "v4mr"
    if name in BASE_TABLES:
        return "base"
    raise KeyError(f"{name!r} is not a model of architecture round v2")


def is_qmp(name: str) -> bool:
    return name.endswith("_qmp")


def stochastic(name: str) -> bool:
    b = backend(name)
    if b == "niche":
        return name == NI.CONTROL                     # the inversion has no stochastic step; the control is XGB
    if b == "wa":
        return False
    return name in (VM.STOCHASTIC if b == "v4mr" else B.STOCHASTIC)


def needs_gpu(name: str) -> bool:
    b = backend(name)
    return name in (VM.NEEDS_GPU if b == "v4mr" else B.NEEDS_GPU if b == "base" else ())


def label(name: str) -> str:
    for table in (NI.LABELS, WA.LABELS, VM.LABELS, B.LABELS):
        if name in table:
            return table[name]
    if is_qmp(name) and name[:-4] in {**VM.LABELS, **B.LABELS}:
        return {**VM.LABELS, **B.LABELS}[name[:-4]] + "+QMP"
    return name


def role(name: str) -> str:
    if name in SLOTS:
        return f"slot#{SLOTS[name]}"
    if name in ABLATIONS:
        return f"ablation {ABLATIONS[name]}"
    if name in CONTROLS:
        return "control"
    if is_qmp(name):
        return "+QMP twin"
    return "comparator"


def is_baseline(name: str) -> bool:
    """Registry flag: everything that is not a counted slot model, an ablation of one or a +QMP twin of one."""
    base = name[:-4] if is_qmp(name) else name
    return base not in SLOTS and base not in ABLATIONS and base != "niche_s"


def release_key(name: str) -> str | None:
    return RELEASE_KEY.get(name)


def qmp_base(name: str) -> str:
    """The base model of a +QMP twin (``<base>_qmp``); the name itself for every other model."""
    return name[:-4] if is_qmp(name) else name


def release_keys(name: str) -> tuple:
    """Every release key a run of ``name`` needs: its own and, for a +QMP twin, its base model's (dev01 lead ruling
    D1: no +QMP twin runs before its base model's own gate is satisfied)."""
    return tuple(dict.fromkeys(k for k in (release_key(qmp_base(name)), release_key(name)) if k))


def gate(name: str, scheme: str, family: str, *, registry_path=None, repo_root=None, confound_done: bool = True,
         slot8_row: str | None = None, g10_row: str | None = None) -> dict:
    """Admission check of one development run. Raises ``stamp.StampError`` or :class:`GateError`.

    Order: amendment stamp; deviations-file releases (the model's own and, for a +QMP twin, its base model's);
    released scheme; family rules; +QMP rules; slot #8 (also for the +QMP twin of the slot-#8 model); the logged
    G10 choice for ``niche_s_qmp`` (dev01 lead rulings D1, D8)."""
    what = f"development run {name} / {scheme} / family {family}"
    st = {"stamp": ST.require_stamp(what, registry_path, repo_root)}
    for key in release_keys(name):
        st["release"] = ST.require_release(what, key, registry_path, repo_root)
        st.setdefault("releases", {})[key] = st["release"]
    backend(name)
    if family not in FAMILIES:
        raise GateError(f"{what}: unknown family (P, G)")
    if scheme not in RELEASED_SCHEMES:
        raise GateError(f"{what}: only {RELEASED_SCHEMES} is released (lead stamp message, step 1)")
    if is_qmp(name):
        if family != "G":
            raise GateError(f"{what}: +QMP models run on the GEO track only (amendment §5)")
        if not confound_done:
            raise GateError(f"{what}: the label-free QMP confound check has not been written (amendment §5)")
    base = qmp_base(name)
    if base in SLOT8_AFTER_GATE:
        if not slot8_row:
            raise GateError(f"{what}: slot #8 needs the registry row id of the lead's written D1-gate evaluation")
        st["slot8_row"] = require_d1_gate_row(slot8_row, base, registry_path)
    if name in G10_ONLY:
        if not g10_row:
            raise GateError(f"{what}: needs the registry row id of the logged G10 choice (ruling B8, dev01 D8)")
        st["g10_row"] = require_g10_choice_row(g10_row, name, registry_path)
    return st


D1_GATE_EXPERIMENT = "microbe/d1_gate_evaluation"   # written by the lead only; no M3 code logs this experiment
SLOT8_AFTER_GATE = {"xgb_v4mr_stack": "fail"}       # amendment §2.6: gate fails -> D3b; gate passes -> D2
G10_CHOICE_EXPERIMENT = "microbe/g10_choice"        # ruling B8: the mechanical G10 choice, logged once before the fit
G10_ONLY = {"niche_s_qmp": "G1"}                    # dev01 D8: run only if the logged G10 choice names this entry


def _registry_rows(experiment: str, registry_path=None) -> list:
    import json
    from pathlib import Path

    from sentinel.core import paths

    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    rows = []
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if experiment not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("experiment") == experiment:
                    rows.append(r)
    return rows


def require_g10_choice_row(row_id: str, name: str, registry_path=None) -> dict:
    """The registry row of the mechanical G10 choice (ruling B8; dev01 lead ruling D8). The row must carry
    experiment ``microbe/g10_choice``, ``extra.g10_entry`` (the G1-G9 entry whose +QMP twin is G10) and
    ``extra.rows`` (the registry row ids the choice was computed from), it must be the only row of that experiment
    (the choice is made once), and the entry must be the one ``name`` is the twin of."""
    rows = _registry_rows(G10_CHOICE_EXPERIMENT, registry_path)
    if len(rows) > 1:
        raise GateError(f"{len(rows)} rows of {G10_CHOICE_EXPERIMENT} in the registry: G10 is chosen once (ruling B8)")
    if not rows or rows[0].get("id") != row_id:
        raise GateError(f"registry row {row_id} is not the logged G10 choice ({G10_CHOICE_EXPERIMENT}); {name} waits "
                        "for it (dev01 lead ruling D8)")
    extra = rows[0].get("extra") or {}
    entry, src = extra.get("g10_entry"), extra.get("rows")
    if entry not in GEO_ENTRIES or not isinstance(src, (list, tuple)) or not src:
        raise GateError(f"registry row {row_id} carries no extra.g10_entry among {tuple(GEO_ENTRIES)} with the "
                        "extra.rows it was computed from")
    if entry != G10_ONLY[name]:
        raise GateError(f"{name} is fitted only if the logged G10 choice names {G10_ONLY[name]}; row {row_id} names "
                        f"{entry}")
    return {"id": row_id, "experiment": G10_CHOICE_EXPERIMENT, "timestamp": rows[0].get("timestamp"),
            "g10_entry": entry, "rows": list(src)}


def require_d1_gate_row(row_id: str, name: str, registry_path=None) -> dict:
    """The registry row of the lead's written D1-gate evaluation (ruling B6: the lead evaluates the gate once, from
    registry rows; M3 does not). The row must carry experiment ``microbe/d1_gate_evaluation``,
    ``extra.written_by`` = "lead" and ``extra.d1_gate`` = "pass" or "fail", and the outcome must be the one that
    opens slot #8 for ``name`` (form agreed with the lead, 2026-11-10)."""
    import json
    from pathlib import Path

    from sentinel.core import paths

    reg = Path(registry_path) if registry_path else paths.REGISTRY_PATH
    if reg.is_file():
        with reg.open() as fh:
            for line in fh:
                if row_id not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("id") != row_id:
                    continue
                if r.get("experiment") != D1_GATE_EXPERIMENT:
                    raise GateError(f"registry row {row_id} is not a D1-gate evaluation (experiment "
                                    f"{r.get('experiment')!r}, need {D1_GATE_EXPERIMENT!r})")
                if (r.get("extra") or {}).get("written_by") != "lead":
                    raise GateError(f"registry row {row_id} does not carry extra.written_by = 'lead'")
                outcome = (r.get("extra") or {}).get("d1_gate")
                if outcome not in ("pass", "fail"):
                    raise GateError(f"registry row {row_id} carries no extra.d1_gate = 'pass' | 'fail'")
                if outcome != SLOT8_AFTER_GATE[name]:
                    raise GateError(f"{name} is the slot-#8 model only if the D1 gate is evaluated as "
                                    f"{SLOT8_AFTER_GATE[name]!r}; row {row_id} says {outcome!r}")
                return {"id": row_id, "workstream": r.get("workstream"), "experiment": r.get("experiment"),
                        "timestamp": r.get("timestamp"), "d1_gate": outcome, "written_by": "lead"}
    raise GateError(f"registry row {row_id} not found: slot #8 waits for the lead's written D1-gate evaluation")


# ------------------------------------------------------------------------------------------------- design

def attach_master(design, root=None):
    """Copy of a DEVELOPMENT design whose ``samples`` carry the GEO-8 and auxiliary target columns.

    ``microbe_geo.load_dev_targets`` refuses any protected id, so a design that includes holdout rows cannot be
    extended here."""
    from sentinel.data import microbe_geo as G

    if getattr(design, "includes_holdout", False):
        raise G.ProtectedRowError("attach_master is for development designs only")
    need = [c for c in NI.MASTER if c not in TARGET_NAMES]
    tab = G.load_dev_targets(design.samples.index, root)
    d = copy.copy(design)
    s = design.samples.copy()
    for c in need:
        s[c] = tab[c].to_numpy(float)
    d.samples = s
    return d


# ------------------------------------------------------------------------------------------- fit / predict

def fit_predict(name: str, design, train_ids, test_ids, *, targets, seed: int = 0, n_jobs: int = 6,
                params: dict | None = None, device: str = "cpu") -> pd.DataFrame:
    b = backend(name)
    mod = {"niche": NI, "wa": WA, "v4mr": VM, "base": B}[b]
    return mod.fit_predict(name, design, train_ids, test_ids, targets=list(targets), seed=seed, n_jobs=n_jobs,
                           params=params, device=device)


def fold_diag(name: str) -> dict | None:
    """Label-free diagnostics of the most recent fold of ``name`` (None for models that keep none)."""
    b = backend(name)
    if b == "niche" and NI.LAST.get("name") == name:
        return dict(NI.LAST["result"].diag)
    if b == "wa" and WA.LAST.get("name") == name:
        return {"n_units": WA.LAST["n_units"], "targets": WA.LAST["targets"]}
    return None


def oof(name: str, design, folds, *, targets, seeds=(0,), n_jobs: int = 6, device: str = "cpu",
        params: dict | None = None, on_fold=None) -> tuple[dict, list]:
    """({seed: out-of-fold predictions with a ``fold`` column}, per-fold diagnostics).

    Same predictions as ``cv.oof_baseline`` called once per seed; the loop is fold-outer / seed-inner so that the
    per-fold caches (features, stacking inputs, the niche fit) serve every seed. ``on_fold(i, fold, diag)`` is
    called after each fold with label-free diagnostics only."""
    seeds = [int(s) for s in seeds]
    parts, diags = {s: [] for s in seeds}, []
    for i, f in enumerate(folds):
        d = None
        for s in seeds:
            p = fit_predict(name, design, f.train, f.test, targets=targets, seed=s, n_jobs=n_jobs, params=params,
                            device=device)
            p["fold"] = i
            parts[s].append(p)
            if d is None:
                d = fold_diag(name)
        if d is not None:
            diags.append({"fold": i, "name": f.name, **d})
        if on_fold is not None:
            on_fold(i, f, d)
    return {s: pd.concat(v) for s, v in parts.items()}, diags


def ensemble(parts: list[pd.DataFrame], targets) -> pd.DataFrame:
    """Unweighted mean of member out-of-fold predictions (protocol §10.1); members must cover the same rows."""
    idx = parts[0].index
    for p in parts[1:]:
        if not p.index.sort_values().equals(idx.sort_values()):
            raise ValueError("members were scored on different rows")
    return sum(p[list(targets)].reindex(idx) for p in parts) / len(parts)


# -------------------------------------------------------------------------------------------- diagnostics

def calibration(y, pred, domain) -> dict:
    """Calibration diagnostic of lead ruling A1 for one target.

    ``slope_within_domain``: OLS slope of observed on predicted after subtracting the domain means of both (rows
    with both values finite; < 1 means the predictions are over-dispersed within domains, > 1 under-dispersed).
    ``sd_ratio``: SD(predicted) / SD(observed), pooled over the same rows. ``sd_ratio_within_domain``: the same
    after subtracting domain means."""
    y, p = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, g = y[ok], p[ok], np.asarray(domain).astype(str)[ok]
    out = {"n": int(ok.sum()), "slope_within_domain": float("nan"), "sd_ratio": float("nan"),
           "sd_ratio_within_domain": float("nan"), "slope_pooled": float("nan")}
    if out["n"] < 10:
        return out
    yc = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
    pc = p - pd.Series(p).groupby(g).transform("mean").to_numpy()
    if (pc * pc).sum() > 0:
        out["slope_within_domain"] = float((pc * yc).sum() / (pc * pc).sum())
    if y.std() > 0:
        out["sd_ratio"] = float(p.std() / y.std())
        pm = p - p.mean()
        if (pm * pm).sum() > 0:
            out["slope_pooled"] = float((pm * (y - y.mean())).sum() / (pm * pm).sum())
    if yc.std() > 0:
        out["sd_ratio_within_domain"] = float(pc.std() / yc.std())
    return out
