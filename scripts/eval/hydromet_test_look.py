#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""The ONLY entry to M6 test data (pre-test review 2026-11-09, C1–C3). One run per look, ever.

    python -B scripts/eval/hydromet_test_look.py --look M6/test  [--gpu]                 # primary (FREEZE_hydromet.json)
    python -B scripts/eval/hydromet_test_look.py --look M6N/test [--gpu]                 # track N (same freeze)
    python -B scripts/eval/hydromet_test_look.py --look M6R/test --freeze-file docs/preregistration/FREEZE_hydromet_trackR.json
    python -B scripts/eval/hydromet_test_look.py --look <look> --rescore-from-saved       # crash recovery only

Models, Holm family size, comparators, artifacts and data pins come from ``freeze["looks"][<look>]``; nothing
about the look is selectable on the command line. Order: load freeze (split + scoring-code hash) → verify data
pins → claim (registry refusal + O_EXCL marker + claim-time registry row) → predict with the frozen artifacts →
seal predictions (0444 + SHA256SUMS) → score + register → SCORED marker. See ``sentinel.models.hydromet.testlook``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from sentinel.core import paths
from sentinel.models.hydromet import runner

DEFAULT_FREEZE = {"M6/test": "FREEZE_hydromet.json", "M6/test_b": "FREEZE_hydromet_test_b.json",
                  "M6N/test": "FREEZE_hydromet.json", "M6R/test": "FREEZE_hydromet_trackR.json"}


def _track_n():
    spec = importlib.util.spec_from_file_location("hydromet_track_n", paths.REPO_ROOT / "scripts" / "eval" / "hydromet_track_n.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_cfg(artifact: Path) -> dict:
    for d in [artifact.parent, artifact.parent.parent, artifact.parent.parent.parent]:
        if (d / "config.json").is_file():
            return json.loads((d / "config.json").read_text())
    raise FileNotFoundError(f"no config.json above {artifact}")


def _groups(artifacts: dict) -> dict:
    """{(track_r fold | None, static_ablation, upstream_forcing): {name: path}} from the frozen artifact keys."""
    out = {}
    for key, path in artifacts.items():
        fold, name = (key.split("/", 1)[0][len("trackr_"):], key.split("/", 1)[1]) if key.startswith("trackr_") else (None, key)
        c = _run_cfg(Path(path))
        if c.get("_smoke") or (fold and c.get("track_r") != fold) or (not fold and c.get("track_r")):
            raise ValueError(f"{key}: artifact run config does not match its key (smoke / track-R fold)")
        out.setdefault((fold, c.get("static_ablation"), c.get("upstream_forcing")), {})[name] = Path(path)
    return out


def predict(look: str, blk: dict, ff: str | None, pred_dir: Path, device: str, log) -> None:
    from sentinel.models.hydromet import pipeline
    from sentinel.models.hydromet import tracks as TR

    block, roles = blk["block"], tuple(blk["roles"])
    base = runner.load_config("configs/hydromet/common.yaml", smoke=False)
    for (fold, abl, ups), arts in _groups(blk["artifacts"]).items():
        tag = f"{fold or 'all'}_{abl or 'full'}_{ups or 'site'}"
        d = pred_dir / (fold or "") if fold else pred_dir
        if look == "M6N/test":
            data = _track_n().build_data(block, roles, allow_test=bool(ff), freeze_file=ff, static_ablation=abl,
                                         upstream=ups, log=log)
            si = np.arange(len(data.sites))
        else:
            cfg = dict(base, static_ablation=abl, upstream_forcing=ups, track_r=fold)
            args = Namespace(eval_blocks=block, split=None, allow_test=bool(ff), freeze_file=ff)
            data, _, _ = runner.prepare(args, cfg, pred_dir.parent / f"prep_{tag}", training=False, log=log, roles=roles)
            si = data.site_idx(list(roles))
            if fold:
                fo = TR.load_folds()["fold_of"]
                si = np.array([i for i in si if fo.get(data.sites[i]) == fold], dtype=int)
        ti = np.flatnonzero(data.block_mask(block))
        pipeline.predict_artifacts(data, arts, si, ti, block, d, lag=0, access=data.access, device=device, log=log,
                                   track_r_fold=fold)


def score(look: str, blk: dict, ff: str | None, pred_dir: Path, out: Path, log, rehearsal: bool = False) -> dict:
    from sentinel.models.hydromet import pipeline

    block, roles = blk["block"], tuple(blk["roles"])
    if look == "M6N/test":
        tn = _track_n()
        data = tn.build_data(block, roles, allow_test=bool(ff), freeze_file=ff, log=log)
        si, ti = np.arange(len(data.sites)), np.flatnonzero(data.block_mask(block))
        preds, sources = pipeline.load_prediction_set([pred_dir], data, si, ti, block, smoke=False, access=data.access)
        prim = tn.score_track_n(data, preds, sources, roles, block, blk["comparators"], blk["primary_models"], out,
                                log=log, experiment=None if rehearsal else look, n_hypotheses=int(blk["n_hypotheses"]))
        return {"n_hypotheses": int(len(prim)), "won": int(prim["won"].sum()) if "won" in prim else 0}
    cfg = dict(runner.load_config("configs/hydromet/common.yaml", smoke=False), track_r="test" if look == "M6R/test" else None)
    args = Namespace(eval_blocks=block, split=None, allow_test=bool(ff), freeze_file=ff)
    data, _, _ = runner.prepare(args, cfg, out, training=False, log=log, roles=roles)
    si, ti = data.site_idx(list(roles)), np.flatnonzero(data.block_mask(block))
    dirs = [pred_dir] + sorted(p for p in pred_dir.iterdir() if p.is_dir())
    preds, sources = pipeline.load_prediction_set(dirs, data, si, ti, block, smoke=False, access=data.access)
    res = pipeline.score(data, preds, si, ti, block, roles, out, n_boot=int(cfg.get("n_boot", 2000)),
                         min_days=int(cfg.get("min_days_per_site", 60)), primary=True, access=data.access, log=log,
                         fixed_comparators=blk["comparators"], primary_models=blk["primary_models"])
    prim = res["primary"]
    if prim is None or len(prim) != int(blk["n_hypotheses"]):
        raise RuntimeError(f"{look}: Holm family {0 if prim is None else len(prim)} != frozen {blk['n_hypotheses']}")
    if rehearsal:
        return {"n_hypotheses": int(len(prim)), "rehearsal": True}
    n = runner.register_summary(res["summary"], cfg, data.split_info, experiment=look, block=block,
                                artifacts=list(sources.values()), roles=roles)
    n += runner.register_pooled_and_primary(res["pooled"], prim, cfg, data.split_info, experiment=look, block=block,
                                            roles=roles, artifacts=list(sources.values()))
    n += runner.register_secondary(res.get("secondary"), cfg, data.split_info, experiment=look, block=block, roles=roles,
                                   artifacts=list(sources.values()))
    log(f"[registry] {n} entries ({look})")
    return {"n_hypotheses": int(len(prim)), "won": int(prim["won"].sum()),
            "label_model_won": prim.loc[prim["model"] == blk.get("label_model"), "won"].tolist()}


def rehearse(a) -> int:
    """Same predict + score code as the look, on val-role sites × val block (not protected), from the candidate
    record's look block. The primary rehearsal must reproduce the registered val score of the same artifacts."""
    from sentinel.models.hydromet import testlook as TL

    rec = json.loads(Path(a.rehearse_on_val).read_text())
    blk = dict(TL.look_block(rec, a.look), block="val", roles=["val"])
    out = paths.RUNS_ROOT / "hydromet" / "test_look_rehearsal" / a.look.replace("/", "_")
    if out.exists():
        raise SystemExit(f"{out} exists — remove it to re-run the rehearsal")
    out.mkdir(parents=True)
    log = runner.make_logger(out / "log.txt")
    if a.gpu:
        runner.setup_gpu("hydromet_test_look_rehearsal")
    predict(a.look, blk, None, out / "predictions", "cuda" if a.gpu else "cpu", log)
    summary = score(a.look, blk, None, out / "predictions", out / "score", log, rehearsal=True)
    log(f"[{a.look}] REHEARSAL (val) done: {summary}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--look", required=True, choices=list(DEFAULT_FREEZE))
    ap.add_argument("--freeze-file", default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--rescore-from-saved", action="store_true")
    ap.add_argument("--rehearse-on-val", default=None, metavar="CANDIDATE_JSON",
                    help="dry run of the whole look on val-role sites x val block from the candidate record: no claim, "
                         "no protected data, no registry; outputs under $SENTINEL_RUNS/hydromet/test_look_rehearsal/")
    a = ap.parse_args()
    runner.env_threads(6)
    if a.rehearse_on_val:
        return rehearse(a)
    from sentinel.models.hydromet import data as D
    from sentinel.models.hydromet import guard
    from sentinel.models.hydromet import testlook as TL

    ff = a.freeze_file or str(paths.REPO_ROOT / "docs" / "preregistration" / DEFAULT_FREEZE[a.look])
    split, _ = D.load_split()
    freeze = guard.load_freeze(ff, split=split)           # split + scoring-code hash
    blk = TL.look_block(freeze, a.look)
    TL.verify_data_pins(freeze, a.look)                    # before the claim: a data mismatch never burns the look
    out = TL.look_dir(a.look)
    pred_dir = out / "predictions"
    if a.rescore_from_saved:
        TL.check_sealed(a.look)
        log = runner.make_logger(out / "log_rescore.txt")
    else:
        TL.claim(a.look, freeze)
        log = runner.make_logger(out / "log.txt")
        if a.gpu:
            runner.setup_gpu("hydromet_test_look")
        predict(a.look, blk, ff, pred_dir, "cuda" if a.gpu else "cpu", log)
        TL.seal_predictions(a.look, pred_dir)
    summary = score(a.look, blk, ff, pred_dir, out / "score", log)
    TL.mark_scored(a.look, summary)
    log(f"[{a.look}] scored once: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
