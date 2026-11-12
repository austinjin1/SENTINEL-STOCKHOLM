# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tabular comparators of M7 architecture round v2 (amendment §5; rulings F2, F4, DS-06): fitted on the v2 training
rows, scored on the national validation blocks and on ``val_p2``. Needs the lead's stamp, except ``--train-smoke``.

    python -B scripts/train/litev2_comparators.py --group tabular            # S1–S9, P1 forms, P2a, P2b; anchors (CPU)
    python -m sentinel.core.gpu --name litev2_s10 --min-free-mb 10000 -- \
        python -B scripts/train/litev2_comparators.py --group tabpfn         # S10 (one GPU)
    python -B scripts/train/litev2_comparators.py --group stacked --embeddings <dir>   # S11
    python -B scripts/train/litev2_comparators.py --group tabular --train-smoke        # training rows only

``--train-smoke`` never loads a validation row: the training rows of anchor fold 0 stand in for an evaluation block,
the models are fitted on the other four folds, nothing is scored, printed as a metric or registered; it checks
shapes, finiteness and run time. The anchor tables (out-of-fold for training rows, refitted model for evaluation
rows) are written by ``--group tabular`` to ``$SENTINEL_RUNS/lite/litev2_anchors`` for the neural arms.

One anchor artefact (ruling Q5; lead order 2026-11-10 20:20 EST, A3): ``--group tabular`` fits the anchor model once,
on the training rows, and registers its sha256 before a validation row is loaded (``anchor_store.build``); the
anchors of an evaluation block and comparator S6 are the same table of predictions of that file. ``--group tabpfn``
on a GPU is logged in the GPU-hour ledger (``launch.gpu_job``, block ``tabpfn``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.models.lite import baselines as B  # noqa: E402
from sentinel.models.litev2 import anchor_store as AS  # noqa: E402
from sentinel.models.litev2 import harness as H  # noqa: E402
from sentinel.models.litev2 import labels2b as L  # noqa: E402
from sentinel.models.litev2 import launch as LA  # noqa: E402
from sentinel.models.litev2 import rows as RW  # noqa: E402
from sentinel.models.litev2 import stamp  # noqa: E402
from sentinel.models.litev2 import tabular as T  # noqa: E402

S9_REGISTERED = ("secchi", "tss", "chla")           # registry row lite/s9_targets (addendum section viii)
NOTES = ("M7 architecture round v2 comparator: fitted on the v2 training rows (national-2b train + train_p2), "
         "scored on validation water bodies; label-side input in P2a/P2b: days between image and sample")


teacher_table = H.teacher_table          # chip-mean teacher μ of the training rows (cached, F4-masked)


def per_target_gbm(Ftr: pd.DataFrame, tr: pd.DataFrame, Fev: pd.DataFrame) -> dict:
    return {t: B.gbm(Ftr, tr[f"y_{t}"].to_numpy(float), Fev)[0] for t in L.TARGETS if tr[f"y_{t}"].notna().any()}


def tabular_group(tr, ftr, ev, fev, teacher, log, anc: dict | None = None,
                  a_ev: pd.DataFrame | None = None) -> tuple[dict, dict, dict]:
    """Predictions ``{comparator: {target: array}}`` for the evaluation frame, run information, anchor outputs.
    ``anc`` / ``a_ev``: the registered anchor artefact and its predictions for the evaluation frame (S6 is that
    table). Without them (training smoke) the anchors are fitted here, in memory, once, and S6 is again the anchor
    table of the evaluation frame."""
    Y = tr[list(RW.Y_COLS)]
    wb = tr["water_body"].to_numpy()
    preds: dict = {}
    info: dict = {}
    t0 = time.time()
    preds.update(T.frozen_list(tr, ftr["X"], ev, fev["X"]))
    log(f"S1–S5 and P1 forms done ({time.time() - t0:.0f} s)")
    if anc is None:
        anc = T.anchors(ftr["X"], Y, wb)
        a_ev = T.predict_parity(anc["model"], fev["X"])
    if a_ev is None or len(a_ev) != len(ev):
        raise RuntimeError("the anchors of the evaluation frame are missing or misaligned")
    preds["S6_gbm_parity"] = AS.s6_predictions(a_ev)                # S6 = the evaluation anchors (one artefact)
    info["S6_gbm_parity"] = anc["info"]
    log(f"S6 and anchors done ({time.time() - t0:.0f} s)")
    m7 = T.fit_parity(ftr["X"], Y, T.parity_columns(), pooled=False)
    preds["S7_gbm_parity_unit"] = T.parity_predictions(m7, fev["X"])
    info["S7_gbm_parity_unit"] = {"n_fit": m7["n_fit"]}
    preds["S8_gbm_parity_mt"], info["S8_gbm_parity_mt"] = T.fit_predict_s8(ftr["X"], Y, wb, teacher, fev["X"],
                                                                         first_stage=anc)
    log(f"S7, S8 done ({time.time() - t0:.0f} s)")
    aft = T.aft_targets(tr)
    if tuple(sorted(aft)) != tuple(sorted(S9_REGISTERED)):
        raise RuntimeError(f"S9 target list {sorted(aft)} differs from the registered one {sorted(S9_REGISTERED)}")
    cols = T.parity_columns()
    preds["S9_gbm_parity_aft"], info["S9_gbm_parity_aft"] = {}, {"n_censored": aft}
    for t in aft:
        preds["S9_gbm_parity_aft"][t], info["S9_gbm_parity_aft"][t] = T.fit_predict_aft(
            ftr["X"][cols].to_numpy(float), tr[f"y_{t}"], tr[f"cens_{t}"], tr[f"lim_{t}"], fev["X"][cols].to_numpy(float))
    log(f"S9 done ({time.time() - t0:.0f} s)")
    preds["P2a_cyfi"] = per_target_gbm(ftr["cyfi"], tr, fev["cyfi"])
    Ptr = pd.concat([ftr["cyfi"], ftr["sc"]], axis=1)
    Pev = pd.concat([fev["cyfi"], fev["sc"]], axis=1)
    preds["P2b_cyfi_parity"] = per_target_gbm(Ptr, tr, Pev)
    info["P2a_cyfi"] = {"n_features": int(ftr["cyfi"].shape[1]), "label_side_feature": "cy_dt_label_days"}
    info["P2b_cyfi_parity"] = {"n_features": int(Ptr.shape[1]), "label_side_feature": "cy_dt_label_days"}
    log(f"P2a, P2b done ({time.time() - t0:.0f} s)")
    return preds, info, anc


def tabpfn_cap() -> int:
    """The row cap registered by the memory test (registry experiment ``lite/tabpfn_memtest``)."""
    from sentinel.core.registry import read_registry

    rows = [r for r in read_registry() if r.get("workstream") == R.WORKSTREAM
            and r.get("experiment") == "lite/tabpfn_memtest"]
    if not rows:
        raise RuntimeError("the TabPFN memory test is not registered (run scripts/train/litev2_tabpfn_memtest.py)")
    cap = int(rows[-1]["extra"]["cap"])
    if cap not in (10_000, 8_000, 6_000):
        raise RuntimeError(f"registered TabPFN cap {cap} is not one of 10,000 / 8,000 / 6,000")
    return cap


def tabpfn_group(tr, ftr, ev, fev, device, log) -> tuple[dict, dict]:
    cap = tabpfn_cap()
    cols = T.parity_columns()
    preds, info = {}, {"cap": cap}
    for t in L.TARGETS:                                   # FNU and NTU as in S7: separate models, NTRU unused
        y = tr[f"y_{t}"].to_numpy(float)
        if not np.isfinite(y).any():
            continue
        t0 = time.time()
        preds[t], info[t] = T.fit_predict_tabpfn(ftr["X"][cols].to_numpy(), y, tr["water_body"].to_numpy(),
                                                 fev["X"][cols].to_numpy(), cap=cap, device=device)
        log(f"S10 {t}: {info[t]['n_fit']} rows fitted ({time.time() - t0:.0f} s)")
    return {"S10_tabpfn": preds}, {"S10_tabpfn": info}


def stacked_group(tr, ftr, ev, fev, emb_dir, block, log) -> tuple[dict, dict]:
    """S11: seed-matched — for seed s the embedding of the seed-s network and the GBM of seed s; mean of the three.
    ``emb_dir`` holds ``emb_train_seed<s>.parquet`` and ``emb_<block>_seed<s>.parquet`` (uid + e000…)."""
    preds_s, info = [], {}
    for s in T.SEEDS:
        etr = pd.read_parquet(Path(emb_dir) / f"emb_train_seed{s}.parquet").set_index("uid")
        eev = pd.read_parquet(Path(emb_dir) / f"emb_{block}_seed{s}.parquet").set_index("uid")
        p, info[f"seed{s}"] = T.fit_predict_stacked(ftr["X"], tr[list(RW.Y_COLS)], etr.loc[tr["uid"]].to_numpy(float),
                                                    fev["X"], eev.loc[ev["uid"]].to_numpy(float), seeds=(s,))
        preds_s.append(p)
        log(f"S11 seed {s} done")
    return ({"S11_gbm_stacked": {t: np.mean([p[t] for p in preds_s], axis=0) for t in preds_s[0]}},
            {"S11_gbm_stacked": info})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, choices=["tabular", "tabpfn", "stacked"])
    ap.add_argument("--train-smoke", action="store_true", help="training rows only; nothing scored or registered")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embeddings", default=None, help="directory of pooled embeddings (group stacked)")
    ap.add_argument("--gpu-budget-h", type=float, default=0.75,
                    help="GPU-hour budget per evaluation block of group tabpfn (ledger; the job stops at it)")
    a = ap.parse_args(argv)
    smoke = a.train_smoke
    if not smoke:
        stamp.require_stamp(f"v2 comparators (group {a.group}) on the validation blocks")
    tr = RW.train_rows()
    ftr = H.feature_store(tr, "train", None)
    cfg = {"group": a.group, "label_version": L.VERSION, "gbm_params": B.GBM_PARAMS, "seeds": list(T.SEEDS),
           "n_train_rows": int(len(tr)), "train_smoke": smoke, "aft_params": T.AFT_PARAMS, "aft_rounds": T.AFT_ROUNDS,
           "features": str(ftr["path"])}
    run_dir = R.run_directory(f"litev2_comparators_{a.group}", smoke)
    with R.training_run(f"litev2_comparators_{a.group}", cfg, run_dir, smoke) as run:
        log = lambda s: (print(s, flush=True), run.note(s))[0]   # noqa: E731
        teacher, tinfo = teacher_table(tr) if a.group == "tabular" else (None, {})
        anc = None
        if a.group == "tabular" and not smoke:                     # before any validation row is loaded
            anc = AS.first_stage(tr) if AS.registered() else AS.build(tr, ftr["X"])
            log(f"anchor artefact sha256 {anc['info']['anchor_sha256']}")
        if smoke:
            fold = T.folds(tr["water_body"])
            te = fold == 0
            sub = lambda d, m: {k: (v.loc[m].reset_index(drop=True) if isinstance(v, pd.DataFrame) else v)   # noqa: E731
                                for k, v in d.items()}
            blocks = {"train_fold0": (tr.loc[te].reset_index(drop=True), sub(ftr, te))}
            tr, ftr = tr.loc[~te].reset_index(drop=True), sub(ftr, ~te)
            teacher = None if teacher is None else teacher.loc[~te].reset_index(drop=True)
        else:
            blocks = {}
            for b in RW.EVAL_BLOCKS:
                ev = RW.eval_rows(b)
                blocks[b] = (ev, H.feature_store(ev, b, None))
        summary = {"teacher": tinfo, "blocks": {}}
        for b, (ev, fev) in blocks.items():
            log(f"block {b}: {len(tr)} training rows, {len(ev)} evaluation rows")
            if a.group == "tabular":
                a_ev = None if smoke else AS.eval_anchors(b, ev["uid"], fev["X"])
                preds, info, _ = tabular_group(tr, ftr, ev, fev, teacher, log, anc, a_ev)
                if not smoke:
                    info["S6_gbm_parity"] = {**info["S6_gbm_parity"], "anchor_table_sha256": AS.table_sha256(b)}
            elif a.group == "tabpfn":
                if str(a.device).startswith("cuda"):
                    with LA.gpu_job(f"litev2_s10_{b}", "tabpfn", a.gpu_budget_h):
                        preds, info = tabpfn_group(tr, ftr, ev, fev, a.device, log)
                else:
                    preds, info = tabpfn_group(tr, ftr, ev, fev, a.device, log)
            else:
                if smoke or not a.embeddings:
                    raise SystemExit("group stacked needs --embeddings and has no train smoke")
                if Path(a.embeddings).name != LA.init_run_dir(True).name:   # S11: the config-3 re-run only (A2)
                    raise SystemExit("--embeddings is not the directory of the corrected-label re-run of config 3")
                preds, info = stacked_group(tr, ftr, ev, fev, a.embeddings, b, log)
            bsum = {}
            for name, p in preds.items():
                bad = [t for t, v in p.items() if len(v) != len(ev) or not np.isfinite(v).all()]
                if bad:
                    raise RuntimeError(f"{name}: non-finite or misaligned predictions for {bad}")
                if smoke:
                    bsum[name] = {"targets": sorted(p), "n_rows": int(len(ev))}
                    continue
                scores = H.score_block(b, ev, p)
                f = run_dir / f"pred_{b}_{name}.parquet"
                H.save_predictions(f, ev, p)
                H.log_block(b, name, scores, cfg=cfg, artifacts=[f], is_baseline=True, notes=NOTES,
                            extra={"info": info.get(name, {})})
                bsum[name] = scores
                for t, s in scores.items():
                    run.metric(f"{b}/{name}/{t}/{s['primary']}", s["value"])
                print(b, name, {t: round(s["value"], 3) for t, s in scores.items()}, flush=True)
            summary["blocks"][b] = {"scores" if not smoke else "shapes": bsum, "info": info}
        R.write_json(run_dir / "summary.json", summary)
    print("done", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
