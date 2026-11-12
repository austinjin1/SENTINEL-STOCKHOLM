#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-F BioSet-MA smoke at full model size on SYNTHETIC data only (protocol §9.4, "before the stamp").

Records time, VRAM, shapes and the guarded checkpoint round trip of the heaviest recipe (region context +
coherence loss, token caps 150 / 100, batch 256, AMP). No real row is read and no validation number of any arm is
produced: the inputs and the targets are random.

    SENTINEL_GPUS=$(python -c "from sentinel.models.bioassess import ftrack as F; \
        print(','.join(map(str, F.unoccupied_gpus())))") \
    python -m sentinel.core.gpu --name m5f_ma_smoke --min-free-mb 9000 -- \
        python -B scripts/train/bioassess_f_smoke.py --rows 14000 --epochs 3 [--chunk-epochs 1]

``--chunk-epochs N``: the fit runs in chunks of N epochs; every resume point goes to disk and comes back through
the guarded loader (protocol v1, lead ruling (o)). ``--recipe ma_zs`` is the F-Z recipe (no coordinates).
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core.registry import log_result
from sentinel.core.runlog import TrainingRun
from sentinel.models.bioassess import bioset_ma as MA
from sentinel.models.bioassess import fdata as FD
from sentinel.models.bioassess import ftrack as F
from sentinel.models.bioassess.bioset import LEVELS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _lineage(prefix: str, n: int) -> pd.DataFrame:
    k = np.arange(n)
    return pd.DataFrame({"phylum": [f"{prefix}p{i % 6}" for i in k], "class": [f"{prefix}c{i % 20}" for i in k],
                         "order": [f"{prefix}o{i % 60}" for i in k], "family": [f"{prefix}f{i % 200}" for i in k],
                         "genus": [f"{prefix}g{i // 2}" for i in k], "name": [f"{prefix}g{i // 2} sp{i % 2}" for i in k]})


def _tokens(keys, lineage: pd.DataFrame, lo: int, hi: int, rng) -> pd.DataFrame:
    n = rng.integers(lo, hi, len(keys))
    p = 1.0 / np.arange(1, len(lineage) + 1) ** 0.7                    # long-tailed taxon frequencies
    p /= p.sum()
    taxa = np.concatenate([rng.choice(len(lineage), int(k), replace=False, p=p) for k in n])
    t = lineage.iloc[taxa].reset_index(drop=True)
    t["key"] = np.repeat(np.asarray(keys, object), n)
    t["mixed_otu"] = t["name"]
    t["count"] = np.ceil(rng.lognormal(1.5, 1.5, len(t)))
    t["log_class"] = np.clip(np.floor(np.log10(t["count"])) + 1, 1, 5)
    return t


def synthetic_design(n_rows: int, seed: int = 0) -> FD.MADesign:
    """Random multi-assemblage design with the columns of the real one: UK-like rows (allowed), UK diatom-only rows
    and French-like rows of folds G1..G8 dated in the development training period."""
    rng = np.random.default_rng(seed)
    n_dia_only = n_rows // 7
    n_inv = n_rows - n_dia_only
    i = np.arange(n_inv)
    src = np.where(i % 2 == 0, "uk", "fr")
    folds = sorted(F.FOLDS)
    fold = np.where(src == "uk", "UK", np.array(folds, object)[(i // 2) % len(folds)])
    late = (src == "fr") & ((i // (2 * len(folds))) % 2 == 0)         # half the French rows: dev pseudo-test period
    date = np.where(late, pd.Timestamp("2019-06-01"), pd.Timestamp("2012-01-01")) + \
        pd.to_timedelta(rng.integers(0, 900, n_inv), unit="D")
    date = pd.DatetimeIndex(date)
    key = np.array([f"uk:{i}" if s == "uk" else f"fr:S{i}:{d.date().isoformat()}:13"
                    for i, (s, d) in enumerate(zip(src, date))], object)
    has_d = rng.random(n_inv) < 0.4
    dkey = np.array([f"{s}d:{i}" if h else None for i, (s, h) in enumerate(zip(src, has_d))], object)
    lat = np.where(src == "uk", rng.uniform(50.5, 55.0, n_inv), rng.uniform(43.0, 50.5, n_inv))
    lon = np.where(src == "uk", rng.uniform(-4.5, 1.0, n_inv), rng.uniform(-1.5, 7.5, n_inv))
    S = pd.DataFrame({"source": src, "station": [f"S{i}" for i in range(n_inv)], "date": date, "lat": lat, "lon": lon,
                      "altitude_m": rng.uniform(0, 800, n_inv), "region": [f"{s}R{i % 12}" for i, s in enumerate(src)],
                      "location_id": [f"L{i}" for i in range(n_inv)], "has_inv": True, "inv_key": key,
                      "diatom_key": dkey, "diatom_offset_days": np.where(has_d, rng.integers(-90, 91, n_inv), np.nan),
                      "protocol_invertebrate": [f"{s}:p{i % 3}" for i, s in enumerate(src)],
                      "protocol_diatom": [f"{s}:LM" if h else None for s, h in zip(src, has_d)],
                      "fold": fold,
                      "lock_role": None, "uk_allowed": src == "uk"}, index=pd.Index(key, name="key"))
    kd = np.array([f"ukd:x{j}" for j in range(n_dia_only)], object)
    Sd = pd.DataFrame({"source": "ukd", "station": [f"D{j}" for j in range(n_dia_only)],
                       "date": pd.Timestamp("2014-04-01"), "lat": rng.uniform(50.5, 55.0, n_dia_only),
                       "lon": rng.uniform(-4.5, 1.0, n_dia_only), "altitude_m": rng.uniform(0, 800, n_dia_only),
                       "region": [f"ukR{j % 12}" for j in range(n_dia_only)],
                       "location_id": [f"LD{j}" for j in range(n_dia_only)], "has_inv": False, "inv_key": None,
                       "diatom_key": kd, "diatom_offset_days": 0.0, "protocol_invertebrate": None,
                       "protocol_diatom": "uk:LM", "fold": "UK", "lock_role": None, "uk_allowed": True},
                      index=pd.Index(kd, name="key"))
    S = pd.concat([S, Sd])
    S["doy_sin"], S["doy_cos"] = FD.doy_features(S["date"])
    tok = {"invertebrate": _tokens(key, _lineage("I", 1500), 20, 200, rng),
           "diatom": _tokens(np.concatenate([dkey[has_d], kd]), _lineage("D", 1000), 10, 130, rng)}
    Y = pd.DataFrame(rng.normal(0.0, 1.0, (len(S), len(FD.Y_COLUMNS))), index=S.index, columns=list(FD.Y_COLUMNS))
    Y = Y.mask(rng.random(Y.shape) < 0.3)
    return FD.MADesign(S, tok, Y, {"synthetic": True})


def run(a) -> int:
    import torch

    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    cfg = MA.recipe(a.recipe, max_epochs=a.epochs, patience=a.epochs, coherence_weight=a.coherence)
    rc = {"recipe": a.recipe, "cfg": cfg, "rows": a.rows, "seed": 0, "device": device, "data": "synthetic"}
    with TrainingRun("bioassess", f"bioassess_f/smoke_synthetic/{a.recipe}", config=rc, data=[],
                     plan_ref="bioassess_f_protocol.md v1 §9.4: code/shape smoke before the stamp",
                     iteration="smoke") as tr_run:
        t0 = time.time()
        D = synthetic_design(a.rows)
        S = D.samples
        log(f"synthetic design: {len(S)} rows, tokens inv {len(D.tokens['invertebrate'])} dia {len(D.tokens['diatom'])}"
            f" ({time.time() - t0:.1f}s)")
        g = "G1"
        pool = MA.usable_ids(S, F.training_ids(S, "F-A", g, "dev"), cfg["modalities"])
        F.assert_fit_ids(pool, S, "F-A", g, "dev")
        fit_ids, val_ids = MA.inner_split(S, pool, seed=0)
        t0 = time.time()
        feat = MA.MAFeaturizer(cfg).fit(D, pool)
        tr, va = feat.transform(D, fit_ids), feat.transform(D, val_ids)
        t_feat = time.time() - t0
        shapes = {m: tuple(tr.mod[m].lev.shape) for m in tr.mod}
        log(f"featurizer: fit {len(fit_ids)} / inner-val {len(val_ids)} rows, token arrays {shapes} ({t_feat:.1f}s)")
        cols = list(FD.Y_COLUMNS)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fit_args = (cfg, feat, tr, D.Y.loc[fit_ids, cols].to_numpy(float), va, D.Y.loc[val_ids, cols].to_numpy(float))
        fit_kw = {"columns": cols, "primary": FD.PRIMARY["F-A"], "seed": 0, "device": device, "log": log}
        if a.chunk_epochs:
            fitted, state, n_resume = None, None, 0
            with tempfile.TemporaryDirectory() as tmp:              # every resume point: disk -> guarded loader
                while fitted is None:
                    fitted, state = MA.fit_ma_chunk(*fit_args, state=state, chunk_epochs=a.chunk_epochs, **fit_kw)
                    if fitted is None:
                        ck = Path(tmp) / "smoke.resume.pt"
                        MA.save_fit_state(ck, state, pool, track="F-A", g=g, design="dev")
                        state = MA.load_fit_state(ck, S, "F-A", g, "dev")
                        n_resume += 1
            log(f"resumable fit: {len(fitted.info['chunks'])} chunks, {n_resume} resume points through the guard")
        else:
            fitted = MA.fit_ma(*fit_args, **fit_kw)
        t_fit = time.time() - t0
        log(f"fit info: {fitted.info}")
        n_ep = len(fitted.history)
        vram_train = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0.0
        resv_train = torch.cuda.max_memory_reserved() / 2 ** 20 if device == "cuda" else 0.0
        ep_s = t_fit / max(n_ep, 1)                                 # includes the inner-validation pass
        log(f"fit: {n_ep} epochs, {ep_s:.1f} s/epoch ({len(fit_ids) / ep_s:.0f} rows/s), {fitted.n_params} params, "
            f"peak VRAM {vram_train:.0f} MB allocated / {resv_train:.0f} MB reserved")
        te = F.test_ids(S, g, "dev")
        q_ids = pd.Index(list(te) + list(val_ids)) if len(te) < 600 else pd.Index(te)
        q = feat.transform(D, q_ids)
        region = np.array(["one_big_region"] * len(q), object)      # > region_max rows: sampled-context path
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        p = fitted.predict(q, region=region, device=device)
        t_pred = time.time() - t0
        vram_pred = torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else 0.0
        assert p["loc"].shape == (len(q), len(cols)) and np.isfinite(p["loc"]).all() and (p["scale"] > 0).all()
        log(f"predict: {len(q)} rows in {t_pred:.1f}s ({len(q) / t_pred:.0f} rows/s), peak VRAM {vram_pred:.0f} MB")
        fitted.fit_embedding(tr, device=device)
        e = fitted.embed_low(q, device=device)
        assert e.shape == (len(q), cfg["embed_dim"]) and np.isfinite(e).all()
        with tempfile.TemporaryDirectory() as tmp:                  # guarded checkpoint round trip (GPU -> CPU)
            ck = Path(tmp) / "smoke.pt"
            MA.save_checkpoint(ck, fitted, feat, pool, track="F-A", g=g, design="dev")
            f2, feat2 = MA.load_fitted(ck, S, "F-A", g, "dev")
            small = feat2.transform(D, q_ids[:64])
            np.testing.assert_allclose(f2.predict(small)["loc"], fitted.predict(small, device=device)["loc"],
                                       atol=5e-2)                   # fp16 autocast on GPU vs fp32 on CPU
        log("checkpoint round trip through ftrack.load_checkpoint: ok")
        for k, v in (("n_params", fitted.n_params), ("fit_rows", len(fit_ids)), ("epoch_seconds", ep_s),
                     ("train_rows_per_s", len(fit_ids) / ep_s), ("featurize_seconds", t_feat),
                     ("peak_vram_train_mb", vram_train), ("peak_vram_reserved_mb", resv_train),
                     ("peak_vram_predict_mb", vram_pred), ("predict_rows_per_s", len(q) / t_pred)):
            tr_run.metric(k, v)
        tr_run.metric("n_chunks", len(fitted.info.get("chunks", [])))
        tr_run.note("synthetic inputs and random targets only; no validation number of any arm")
        log_result(workstream="bioassess", experiment="bioassess_f/smoke_synthetic", model=f"{a.recipe}_full_size",
                   metric="peak_vram_train_mb", value=float(vram_train), split="synthetic", n=len(fit_ids), config=rc,
                   notes="code/shape smoke on synthetic data before the M5-F stamp; no validation number",
                   extra={"epoch_seconds": ep_s, "peak_vram_predict_mb": vram_pred, "n_params": fitted.n_params,
                          "token_shapes": {m: list(s) for m, s in shapes.items()}, "device": device,
                          "chunk_epochs": a.chunk_epochs, "fit_info": fitted.info,
                          "context_cols": list(cfg["context_cols"])})
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="ma_full_rc", choices=sorted(MA.RECIPES))
    ap.add_argument("--rows", type=int, default=14000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--coherence", type=float, default=0.1)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--chunk-epochs", type=int, default=0, help="0 = one uninterrupted fit")
    ap.add_argument("--threads", type=int, default=2)
    raise SystemExit(run(ap.parse_args()))
