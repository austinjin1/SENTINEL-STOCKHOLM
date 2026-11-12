# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Neural arms and comparators of M7 architecture round v2 (amendment §4.3–§4.5, §5; rulings F5, F8): ANCHOR-Lite,
ANCHOR-Set, ablations A1 / A3 / A4 / A5, their no-KD twins, the DenseNet121 comparators P3a / P3b, and the pooled
embedding of the leader for comparator S11. Every stage needs the lead's stamp.

    G="python -m sentinel.core.gpu --min-free-mb 10000"
    $G --name litev2_p3a_lr1e-4_s0 -- python -B scripts/train/litev2_train.py --arm P3a_densenet \
        --backbone-lr 1e-4 --seeds 0 --stage fit                       # one seed per job (80 epochs)
    python -B ... --arm P3a_densenet --backbone-lr 1e-4 --stage eval   # all three seeds → registry rows
    $G --name litev2_embed -- python -B ... --arm embed                                # S11 embeddings
    $G --name litev2_anchor_lite -- python -B ... --arm anchor_lite --stage fit        # refuses before the comparators
    python -B ... --arm anchor_lite --stage eval                                       # refuses before the comparators

Stages. ``fit`` reads training rows only (v2 training rows, out-of-fold anchors, KD tensors with the Test-S pairings
removed) and trains for the fixed number of epochs; one checkpoint and one registry row (split ``train``) per seed;
a seed that already has a checkpoint is never trained again. ``eval`` loads the validation blocks once, scores the
3-seed ensemble (mean μ; σ² = mean σ² + variance of μ) and every seed on the national validation block and on
``val_p2`` and writes the registry rows; it refuses a second evaluation of the same name. One training run serves
both tracks. A candidate's ``eval`` calls the comparators-first gate for both blocks before any validation row is
loaded (amendment §5).

Lead order of 2026-11-10 20:20 EST. (A4) A candidate's ``fit`` calls the same gate first: no candidate fit starts
before every comparator validation row of both blocks is registered. (A2) The ANCHOR-Lite family starts, seed for
seed, from the corrected-label re-run of configuration 3 (``--no-kd`` twins: from the re-run of its no-KD twin), and
``--arm embed`` reads the same re-run checkpoints; the directory is found by ``launch.init_run_dir`` and every file
is checked against its registered sha256 (``launch.init_checkpoint``) — a missing, unregistered or as-built
checkpoint is a hard error, there is no fallback. ``--init-run`` is optional and only checked against that
directory. (A7) Every stage on a GPU is one job of the GPU-hour ledger (``launch.gpu_job``) and stops at its budget.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pickle
import time
from pathlib import Path

from sentinel.models.lite import runner as R

R.cpu_threads(6)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.litev2 import anchor_store as AS  # noqa: E402
from sentinel.models.litev2 import fit as FT  # noqa: E402
from sentinel.models.litev2 import harness as H  # noqa: E402
from sentinel.models.litev2 import labels2b as L  # noqa: E402
from sentinel.models.litev2 import launch as LA  # noqa: E402
from sentinel.models.litev2 import nets as NN  # noqa: E402
from sentinel.models.litev2 import rows as RW  # noqa: E402
from sentinel.models.litev2 import stamp  # noqa: E402
from sentinel.models.litev2 import tabular as T  # noqa: E402

FIT_EXPERIMENT = "litev2_fit"
EMBED_DIR = "litev2_embeddings"
DENSENET_FILE = paths.RAW / "torchvision_densenet121_imagenet1k_v1" / "densenet121-a639ec97.pth"
TRAIN_CACHE, VAL_CACHE = "v2train", "v2val"
A4_SHA_FILE = "a4_anchor_sha256.json"
NOTES = ("M7 architecture round v2: fitted on the v2 training rows (national-2b train + train_p2) for a fixed number "
         "of epochs; no validation row is used in training")


def run_name(a) -> str:
    if a.arm.startswith("P3"):
        return f"{a.arm}_lr{a.backbone_lr}"
    return a.arm + ("_nokd" if a.no_kd else "")


def arm_cfg(a) -> dict:
    over = {}
    if a.arm.startswith("P3"):
        if a.backbone_lr not in H.DENSENET_LRS:
            raise SystemExit(f"--backbone-lr must be one of {H.DENSENET_LRS} for the DenseNet comparators (F8)")
        over["lr_backbone"] = float(a.backbone_lr)
    elif a.backbone_lr is not None:
        raise SystemExit("--backbone-lr is a DenseNet option; the other arms use the registered rates")
    if a.no_kd:
        if FT.ARMS[a.arm]["net"] != "anchor":
            raise SystemExit("--no-kd applies to the ANCHOR-Lite family (the no-KD twins of ruling F5)")
        over["kd_weight"] = 0.0
    return FT.arm_config(a.arm, **over)


def leader_state(kd: bool, seed: int, init_run: str | None = None) -> dict:
    """Seed-matched warm-start weights: the corrected-label re-run of configuration 3 (``kd``) or of its no-KD
    twin. ``launch.init_checkpoint`` raises when the re-run checkpoint is missing, is not registered, differs from
    its registered sha256, or when ``init_run`` names another directory. There is no fallback."""
    f = LA.init_checkpoint(kd, seed, init_run=init_run)
    return torch.load(f, map_location="cpu", weights_only=True)["state_dict"]


def init_name(kd: bool, seeds, init_run: str | None = None) -> str:
    """Name of the re-run directory the job starts from, after every seed's checkpoint has been checked."""
    for s in seeds:
        LA.init_checkpoint(kd, s, init_run=init_run)
    return LA.init_run_dir(kd).name


def load_anchor_table(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    """The §4.2 anchor table of the registered anchor artefact (``anchor_store``), aligned with ``rows`` by uid."""
    return AS.load_table(name, rows["uid"])[list(L.TARGETS)].reset_index(drop=True)


def gpu_block(a) -> str:
    """Block of the addendum's compute table a GPU stage of this job is logged under."""
    if a.arm == "embed":
        return "s11_embed"
    if a.arm.startswith("P3"):
        return "densenet"
    return "slot_twins" if a.no_kd else "v2_networks"


SEED_BUDGET_H = {"densenet": 2.0, "v2_networks": 0.5, "slot_twins": 0.5}     # per fitted seed; a job stops at ≤ 2 h
EVAL_BUDGET_H, EMBED_BUDGET_H = 0.25, 0.5


def gpu_budget(a) -> float:
    if a.gpu_budget_h is not None:
        return float(a.gpu_budget_h)
    if a.arm == "embed":
        return EMBED_BUDGET_H
    h = SEED_BUDGET_H[gpu_block(a)] * len(a.seeds) if a.stage in ("fit", "both") else 0.0
    return min(LA.JOB_CAP_H, h + (EVAL_BUDGET_H if a.stage in ("eval", "both") else 0.0))


def reduced_labels(tr: pd.DataFrame) -> pd.DataFrame:
    """A4: the four original targets, uncensored values only — every recovered label column is empty."""
    Y = tr[list(RW.Y_COLS)].copy()
    for c in Y.columns:
        if c[2:] not in L.ORIG:
            Y[c] = np.nan
    return Y


def a4_anchors(run_dir: Path, tr: pd.DataFrame, ftr: dict) -> tuple[pd.DataFrame, dict]:
    """A4 anchors refitted on the reduced rows (§4.5): out-of-fold for the training rows; the refitted model is kept
    in the run directory for the evaluation stage."""
    f_oof, f_model = run_dir / "a4_anchors_train.parquet", run_dir / "a4_anchor_model.pkl"
    if f_oof.is_file() and f_model.is_file():
        oof = pd.read_parquet(f_oof)
        if oof["uid"].tolist() != tr["uid"].astype(str).tolist():
            raise RuntimeError("the stored A4 anchors do not match the training rows")
        return oof[list(L.ORIG)], a4_model(run_dir)
    anc = T.anchors(ftr["X"], reduced_labels(tr), tr["water_body"].to_numpy())
    anc["train"].assign(uid=tr["uid"].astype(str).to_numpy()).to_parquet(f_oof, index=False)
    f_model.write_bytes(pickle.dumps(anc["model"]))
    R.write_json(run_dir / A4_SHA_FILE, {"a4_anchor_model.pkl": hashlib.sha256(f_model.read_bytes()).hexdigest()})
    return anc["train"][list(L.ORIG)], anc["model"]


def a4_model(run_dir: Path) -> dict:
    """The A4 anchor model of this run directory, checked against the sha256 written when it was fitted."""
    f = run_dir / "a4_anchor_model.pkl"
    want = json.loads((run_dir / A4_SHA_FILE).read_text())["a4_anchor_model.pkl"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != want:
        raise RuntimeError(f"{f} differs from the sha256 written at its fit")
    return pickle.loads(f.read_bytes())


def train_tensors(cfg: dict, run_dir: Path, log) -> tuple[FT.Tensors, pd.DataFrame, dict]:
    tr = RW.train_rows()
    ftr = H.feature_store(tr, "train", None)
    chips = RW.chips(tr, cache_name=TRAIN_CACHE)
    info = {"n_train_rows": int(len(tr)), "features": str(ftr["path"])}
    anchors = None
    if cfg["net"] != "densenet" and cfg["use_anchor"]:
        anchors = a4_anchors(run_dir, tr, ftr)[0] if cfg["reduced"] else load_anchor_table("train", tr)
    tens = FT.build_tensors(tr, chips, ftr["sc"], anchors, cfg["targets"], reduced=cfg["reduced"],
                            with_set=cfg["net"] == "set")
    if cfg["net"] == "anchor" and cfg["kd_weight"] > 0:
        kd, masked = H.kd_tensors(tr, chips)
        info["kd"] = FT.attach_kd(tens, kd, masked)
        log(f"KD tensors: {info['kd']}")
    if cfg["net"] == "densenet" and cfg["parity"]:
        tm, tinfo = H.teacher_table(tr)
        tens.teacher = torch.from_numpy(tm.to_numpy(np.float32))
        info["teacher"] = tinfo
    info["n_terms"] = {t: int(torch.isfinite(tens.y[:, j]).sum() + (tens.cens[:, j] != 0).sum())
                       for j, t in enumerate(cfg["targets"])}
    info["n_ntru"] = int(torch.isfinite(tens.y_ntru).sum())
    return tens, tr, info


# ------------------------------------------------------------------------------------------------ stages
def stage_fit(a, cfg: dict, name: str, run_dir: Path) -> None:
    from sentinel.core.registry import log_result

    H.require_gate(name)                       # A4: a candidate is not fitted before the comparators' validation rows
    todo = [s for s in a.seeds if not (run_dir / f"model_seed{s}.pt").exists()]
    if not todo:
        print("every requested seed already has a checkpoint; nothing is trained again")
        return
    kd_init = not a.no_kd
    init = init_name(kd_init, todo, a.init_run) if cfg["net"] == "anchor" else None      # A2: before any row is loaded
    with R.training_run(f"litev2_fit_{name}", {**cfg, "seeds": todo, "init_run": init}, run_dir, False) as run:
        log = lambda s: (print(s, flush=True), run.note(s))[0]   # noqa: E731
        tens, tr, dinfo = train_tensors(cfg, run_dir, log)
        weights = NN.densenet_state(DENSENET_FILE) if cfg["net"] == "densenet" else None
        for s in todo:
            t0 = time.time()
            lead = leader_state(kd_init, s, a.init_run) if cfg["net"] == "anchor" else None
            model, hist, info = FT.train(cfg, tens, seed=s, device=a.device, leader_state=lead,
                                         densenet_weights=weights, log=lambda m, s=s: log(f"seed {s} {m}"))
            ck = run_dir / f"model_seed{s}.pt"
            if ck.exists():
                raise RuntimeError(f"{ck} appeared during training; not overwritten")
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "seed": s, "hist": hist, "info": info,
                        "init_run": init, "label_version": L.VERSION}, ck)
            rec = {"seed": s, "seconds": round(time.time() - t0, 1), "n_parameters": NN.count_parameters(model),
                   "init": info, "data": dinfo, "hist": hist, "init_run": init}
            R.write_json(run_dir / f"fit_seed{s}.json", rec)
            run.metric(f"seed{s}/train_loss", hist[-1]["loss"])
            log_result(workstream=R.WORKSTREAM, experiment=FIT_EXPERIMENT, model=H.MODEL_PREFIX + name,
                       metric="train_loss_final", value=float(hist[-1]["loss"]), split="train", n=int(len(tr)),
                       config=R.public_config(cfg), artifacts=[str(ck)], notes=NOTES,
                       extra={"written_by": "M7", "seed": s, "epochs": cfg["epochs"], "init_run": init,
                              "seconds": rec["seconds"], "label_version": L.VERSION, "data": dinfo})
            del model
            if str(a.device).startswith("cuda"):
                torch.cuda.empty_cache()


def _load_model(cfg: dict, ck: Path, device: str):
    d = torch.load(ck, map_location="cpu", weights_only=True)
    if d["cfg"]["arm"] != cfg["arm"] or float(d["cfg"]["lr_backbone"]) != float(cfg["lr_backbone"]) \
            or float(d["cfg"]["kd_weight"]) != float(cfg["kd_weight"]):
        raise RuntimeError(f"{ck} was trained with another configuration")
    model = FT.build_model(cfg)
    model.load_state_dict(d["state_dict"], strict=True)
    return model.to(device).eval()


def eval_tensors(cfg: dict, block: str, run_dir: Path) -> tuple[FT.Tensors, pd.DataFrame]:
    ev = RW.eval_rows(block)
    fev = H.feature_store(ev, block, None)
    chips = RW.chips(ev, cache_name=VAL_CACHE if block == "national_val" else None)
    anchors = None
    if cfg["net"] != "densenet" and cfg["use_anchor"]:
        if cfg["reduced"]:
            anchors = T.predict_parity(a4_model(run_dir), fev["X"])[list(L.ORIG)].reset_index(drop=True)
        else:
            anchors = load_anchor_table(block, ev)[list(cfg["targets"])]
    rows = ev
    if cfg["reduced"]:                                             # the label columns are not used in prediction
        rows = ev.assign(**{c: np.nan for c in RW.Y_COLS if c[2:] not in L.ORIG})
    return FT.build_tensors(rows, chips, fev["sc"], anchors, cfg["targets"], reduced=cfg["reduced"],
                            with_set=cfg["net"] == "set"), ev


def stage_eval(a, cfg: dict, name: str, run_dir: Path) -> None:
    from sentinel.core.registry import read_registry

    candidate = H.is_candidate(name)
    H.require_gate(name)                                           # before any validation row is loaded (§5)
    cks = [run_dir / f"model_seed{s}.pt" for s in a.seeds]
    if [c for c in cks if not c.is_file()]:
        raise SystemExit(f"missing checkpoints: {[c.name for c in cks if not c.is_file()]}")
    done = [r for r in read_registry() if r.get("workstream") == R.WORKSTREAM
            and r.get("model") == H.MODEL_PREFIX + name and r.get("experiment") in H.EXPERIMENT.values()]
    if done:
        raise SystemExit(f"{name} already has {len(done)} validation rows in the registry; it is evaluated once")
    with R.training_run(f"litev2_eval_{name}", {**cfg, "seeds": list(a.seeds)}, run_dir, False) as run:
        summary = {"name": name, "seeds": list(a.seeds), "candidate": candidate, "blocks": {}}
        for b in RW.EVAL_BLOCKS:
            tens, ev = eval_tensors(cfg, b, run_dir)
            mus, sds, per_seed = [], [], {}
            for s, ck in zip(a.seeds, cks):
                model = _load_model(cfg, ck, a.device)
                p = FT.predict(model, cfg, tens, device=a.device)
                mus.append(p["mu"])
                sds.append(p["sd"])
                per_seed[f"seed{s}"] = H.score_block(b, ev, FT.prediction_dict(cfg, p["mu"]),
                                                     FT.prediction_dict(cfg, p["sd"]))
                H.save_predictions(run_dir / f"pred_{b}_{name}_seed{s}.parquet", ev, FT.prediction_dict(cfg, p["mu"]),
                                   FT.prediction_dict(cfg, p["sd"]))
                del model
            mu, sd = FT.ensemble(mus, sds)
            preds, sig = FT.prediction_dict(cfg, mu), FT.prediction_dict(cfg, sd)
            if any(not np.isfinite(v).all() for v in preds.values()):
                raise RuntimeError(f"{name}: non-finite predictions on {b}")
            scores = H.score_block(b, ev, preds, sig)
            f = run_dir / f"pred_{b}_{name}.parquet"
            H.save_predictions(f, ev, preds, sig)
            H.log_block(b, name, scores, cfg=cfg, artifacts=[f, *cks], is_baseline=not candidate, notes=NOTES,
                        extra={"seeds": list(a.seeds), "ensemble": "mean mu; var = mean var + var of mu",
                               "per_seed": {k: {t: v["value"] for t, v in sc.items()} for k, sc in per_seed.items()},
                               "init_run": LA.init_run_dir(not a.no_kd).name if cfg["net"] == "anchor" else None})
            for t, sc in scores.items():
                run.metric(f"{b}/{name}/{t}/{sc['primary']}", sc["value"])
            summary["blocks"][b] = {"ensemble": scores, "per_seed": per_seed, "n_rows": int(len(ev))}
            print(b, name, {t: round(sc["value"], 3) for t, sc in scores.items()}, flush=True)
        R.write_json(run_dir / "summary.json", summary)


@torch.no_grad()
def _embed(model: NN.AnchorLite, tens: FT.Tensors, device: str, batch: int = 64) -> np.ndarray:
    """Centre-window-pooled decoder output (48) of the leader — the vector its own chip-level pooling uses."""
    out = []
    st = model.student
    for i in range(0, tens.n, batch):
        dn = tens.dn[i:i + batch].to(device).float()
        x = (dn / 255.0 - st.x_mean.view(1, -1, 1, 1)) / st.x_std.view(1, -1, 1, 1)
        top = model.decoder(x).float()
        out.append((top * tens.pool_w[i:i + batch].to(device).unsqueeze(1)).sum((2, 3)).cpu().numpy())
    return np.concatenate(out)


def stage_embed(a) -> None:
    """Pooled embeddings of the leader for S11: training rows and both validation blocks, one file per seed."""
    init = init_name(True, a.seeds, a.init_run)                    # A2: the config-3 re-run, checked, no fallback
    out_dir = paths.run_dir("lite", f"{EMBED_DIR}/{init}")
    sets = {"train": RW.train_rows(), **{b: RW.eval_rows(b) for b in RW.EVAL_BLOCKS}}
    for nm, rows in sets.items():
        chips = RW.chips(rows, cache_name={"train": TRAIN_CACHE, "national_val": VAL_CACHE}.get(nm))
        sc = H.feature_store(rows, nm, chips)["sc"]
        tens = FT.build_tensors(rows.assign(**{c: np.nan for c in RW.Y_COLS}), chips, sc, None, L.TARGETS,
                                reduced=True)                      # no label value is carried into the tensors
        for s in a.seeds:
            f = out_dir / f"emb_{nm}_seed{s}.parquet"
            if f.exists():
                continue
            m = NN.AnchorLite()
            m.load_leader(leader_state(True, s, a.init_run))
            e = _embed(m.to(a.device).eval(), tens, a.device)
            pd.DataFrame(e, columns=[f"e{j:03d}" for j in range(e.shape[1])]).assign(
                uid=rows["uid"].astype(str).to_numpy()).to_parquet(f, index=False)
            print(f"{f.name}: {e.shape}", flush=True)
    R.write_json(out_dir / "meta.json", {"init_run": init, "seeds": list(a.seeds), "width": 48,
                                         "definition": "centre-window-pooled stride-4 decoder output of the leader",
                                         "n_rows": {k: int(len(v)) for k, v in sets.items()}})
    print("embeddings in", out_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=[*FT.ARMS, "embed"])
    ap.add_argument("--stage", default="both", choices=["fit", "eval", "both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-kd", action="store_true", help="no-KD twin (ruling F5); starts from the no-KD leader")
    ap.add_argument("--backbone-lr", default=None, help="DenseNet backbone rate: 1e-4 or 3e-4 (ruling F8)")
    ap.add_argument("--init-run", default=None,
                    help="optional: must name the corrected-label re-run directory the job starts from (checked)")
    ap.add_argument("--gpu-budget-h", type=float, default=None, help="GPU-hour budget of this job (ledger; ≤ 2)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if set(a.seeds) - {0, 1, 2}:
        raise SystemExit("the registered seeds are 0, 1 and 2")
    stamp.require_stamp(f"v2 neural arm {a.arm} (stage {a.stage})")
    on_gpu = str(a.device).startswith("cuda")
    job = (lambda nm: LA.gpu_job(nm, gpu_block(a), gpu_budget(a))) if on_gpu else (lambda nm: contextlib.nullcontext())
    if a.arm == "embed":
        with job("litev2_embed_s" + "".join(map(str, a.seeds))):
            stage_embed(a)
        return 0
    cfg = arm_cfg(a)
    name = run_name(a)
    H.require_gate(name)                       # A4: refused before a run directory or a ledger line is written
    run_dir = paths.run_dir("lite", f"litev2_{name}")
    with job(f"litev2_{name}_{a.stage}_s" + "".join(map(str, a.seeds))):
        if a.stage in ("fit", "both"):
            stage_fit(a, cfg, name, run_dir)
        if a.stage in ("eval", "both"):
            stage_eval(a, cfg, name, run_dir)
    print("done", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
