# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Self-supervised pretraining of the SatWQ-Net spectral encoder on unlabelled S2 water pixels
(orbit_protocol.md §11.6, lever 4; lead-approved 2026-11-09).

* **Pixels.** SCL-water pixels (all 12 bands valid) anywhere in the 64x64 chips of the existing chip stores whose
  point is a **train-role** unit (lake or non-lake) acquired **<= 2021** (train time block), not in a buffer zone and
  not in an inner (early-stopping) block. Roles are recomputed from each chip's lon/lat/date with the frozen v3 rule;
  every val/test-role unit is excluded by coordinates. No label is read. The included/excluded chip lists and counts
  are written to an exclusion manifest.
* **Objective.** Masked-band modelling: a random subset of each pixel's band tokens is dropped (padding mask) and
  their asinh-scaled reflectance is reconstructed from the CLS embedding plus the band's wavelength embedding (MSE).
  Only the spectral encoder (``val_embed``, ``wl_embed``, ``cls``, ``spec``) is trained; the SSL head is discarded.
* MDN_B and the GBMs cannot use unlabelled imagery: any gain is from an information source the comparators lack.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data import chips as C
from . import data as D
from . import features as F
from . import split as SP

SSL_VERSION = "v1"
P_PER_CHIP = 64


def ssl_dir() -> Path:
    return paths.ensure(paths.RUNS_ROOT / "orbit" / "ssl")


def eligible_chips(log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(included, excluded) chip tables over both stores; roles from the frozen v3 rule on (lon, lat, date)."""
    parts = []
    for store in D.STORES.values():
        idx = C.read_index_parts(paths.processed("s2_chips") / store).drop_duplicates("request_key", keep="last")
        idx = idx[idx.status == "ok"][["chip_id", "shard", "lon", "lat", "platform", "sensing_time_utc"]].copy()
        idx["chip_store"] = store
        parts.append(idx)
    idx = pd.concat(parts, ignore_index=True).drop_duplicates(["chip_store", "chip_id"])
    date = pd.to_datetime(idx.sensing_time_utc, utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    ok = date.notna().to_numpy() & np.isfinite(idx.lon.to_numpy(float)) & np.isfinite(idx.lat.to_numpy(float))
    idx, date = idx[ok].reset_index(drop=True), date[ok].reset_index(drop=True)
    sp = SP.assign(idx.lon.to_numpy(float), idx.lat.to_numpy(float), date.to_numpy(), log=log)
    idx = pd.concat([idx, sp[["role", "tblock", "block", "waterbody", "buffer_excluded", "protected", "anchor_lon",
                              "anchor_lat"]]], axis=1)
    inner = F.inner_block(idx["block"])
    keep = ((idx.role == "train") & (idx.tblock == "train") & ~idx.buffer_excluded.astype(bool)
            & ~idx.protected.astype(bool) & ~inner).to_numpy()
    reason = np.where(idx.role != "train", "role_" + idx.role.astype(str),
                      np.where(idx.tblock != "train", "year_" + idx.tblock.astype(str),
                               np.where(idx.buffer_excluded.astype(bool), "buffer", np.where(inner, "inner", "kept"))))
    idx["reason"] = reason
    return idx[keep].reset_index(drop=True), idx[~keep].reset_index(drop=True)


def _water_pixels(args):
    store, shard, chip_ids, n_max, seed = args
    rng = np.random.default_rng(seed)
    z = np.load(paths.processed("s2_chips") / store / "shards" / shard)
    L = {n: i for i, n in enumerate(C.S2_STORE_LAYERS)}
    from ...data.sources import stac as S

    out = []
    try:
        for cid in chip_ids:
            chip = z[cid]
            dn = np.stack([chip[L[b]] for b in D.BANDS])                  # [12, 64, 64]
            ok = (chip[L["SCL"]] == 6) & np.all(dn != S.NODATA_I16, axis=0)
            rr, cc = np.nonzero(ok)
            if rr.size == 0:
                continue
            pick = rng.choice(rr.size, min(n_max, rr.size), replace=False)
            out.append((cid, (dn[:, rr[pick], cc[pick]].T.astype(np.float32) * 1e-4).astype(np.float16)))
    finally:
        z.close()
    return out


def build_pixels(workers: int = 6, n_per_chip: int = P_PER_CHIP, seed: int = 0, variant: str = SSL_VERSION,
                 log=print) -> dict:
    """Extract the SSL pixel set and write ``ssl/pixels_<variant>.npz`` + the exclusion manifest
    ``manifests/orbit_ssl_pixels_<variant>.json`` (included/excluded chip lists with counts and sha256).
    ``variant`` ending in ``_trackU`` also drops chips of the held-out track-U regions + 25-km buffer (§11.2), so the
    track-U refits never see imagery of their test regions, even unlabelled."""
    from concurrent.futures import ProcessPoolExecutor

    inc, exc = eligible_chips(log=log)
    if variant.endswith("_trackU"):
        from . import track_u as U

        ux = U.assign(inc.anchor_lon, inc.anchor_lat).u_excluded_from_fit.to_numpy(bool)
        exc = pd.concat([exc, inc[ux].assign(reason="trackU_region")], ignore_index=True)
        inc = inc[~ux].reset_index(drop=True)
    log(f"[ssl] chips included {len(inc)}, excluded {len(exc)} ({exc.reason.value_counts().to_dict()})")
    jobs = [(st, sh, g.chip_id.tolist(), n_per_chip, seed + i)
            for i, ((st, sh), g) in enumerate(inc.groupby(["chip_store", "shard"]))]
    refl, plat, src = [], [], []
    pmap = dict(zip(zip(inc.chip_store, inc.chip_id), inc.platform))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (st, _, _, _, _), part in zip(jobs, ex.map(_water_pixels, jobs)):
            for cid, r in part:
                refl.append(r)
                plat += [pmap[(st, cid)]] * len(r)
                src.append((st, cid, len(r)))
    refl = np.concatenate(refl)
    out = ssl_dir() / f"pixels_{variant}.npz"
    np.savez(out, refl=refl, platform=np.array(plat))
    inc_ids = sorted(f"{s}/{c}" for s, c, _ in src)
    exc_ids = sorted(f"{s}/{c}" for s, c in zip(exc.chip_store, exc.chip_id))
    man = {"name": f"orbit_ssl_pixels_{variant}", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "description": "SSL water pixels: train-role units x train years (<=2021), no buffer, no inner blocks; "
                          "roles recomputed with the frozen split v3 rule; no labels read (orbit_protocol.md §11.6)",
           "split_sha256": SP.split_sha256(), "n_pixels": int(len(refl)), "n_chips_included": len(inc_ids),
           "n_chips_excluded": len(exc_ids), "excluded_by_reason": exc.reason.value_counts().to_dict(),
           "included_roles": sorted(inc.role.unique().tolist()), "included_tblocks": sorted(inc.tblock.unique().tolist()),
           "included_chip_ids_sha256": hashlib.sha256("\n".join(inc_ids).encode()).hexdigest(),
           "excluded_chip_ids_sha256": hashlib.sha256("\n".join(exc_ids).encode()).hexdigest(),
           "pixels_file": str(out), "pixels_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (ssl_dir() / f"chips_included_{variant}.txt").write_text("\n".join(inc_ids) + "\n")
    (ssl_dir() / f"chips_excluded_{variant}.txt").write_text("\n".join(exc_ids) + "\n")
    (paths.MANIFESTS / f"orbit_ssl_pixels_{variant}.json").write_text(json.dumps(man, indent=1, default=str))
    return man


def verify_pixels(version: str = SSL_VERSION) -> dict:
    man = json.loads((paths.MANIFESTS / f"orbit_ssl_pixels_{version}.json").read_text())
    if hashlib.sha256(Path(man["pixels_file"]).read_bytes()).hexdigest() != man["pixels_sha256"]:
        raise RuntimeError("SSL pixel file changed since its manifest")
    if man["split_sha256"] != SP.split_sha256() or man["included_roles"] != ["train"] or man["included_tblocks"] != ["train"]:
        raise RuntimeError("SSL pixel manifest is not train-role x train-years under the frozen split")
    return man


def pretrain_ssl(model, *, epochs: int, mask_frac: float, lr: float, batch: int, device, seed: int,
                 max_pixels: int | None = None, variant: str = SSL_VERSION, log=print) -> dict:
    """Masked-band modelling on the encoder of ``model`` (in place). Returns a history dict."""
    import torch
    import torch.nn as nn

    from .pipeline import lam_matrix
    from .satwq import REFL_SCALE, sinusoidal

    man = verify_pixels(variant)
    z = np.load(man["pixels_file"])
    refl, plat = z["refl"], z["platform"]
    rng = np.random.default_rng(seed)
    if max_pixels and len(refl) > max_pixels:
        sel = rng.choice(len(refl), max_pixels, replace=False)
        refl, plat = refl[sel], plat[sel]
    lam = lam_matrix(plat)
    d = model.d
    head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)).to(device)
    params = [p for n, p in model.named_parameters() if n.split(".")[0] in ("val_embed", "wl_embed", "cls", "spec")]
    opt = torch.optim.AdamW(params + list(head.parameters()), lr=lr, weight_decay=1e-4)
    n, B = len(refl), refl.shape[1]
    steps = max(1, n // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    hist = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).numpy()
        tot = 0.0
        for s in range(steps):
            ix = perm[s * batch:(s + 1) * batch]
            r = torch.as_tensor(refl[ix].astype(np.float32), device=device)
            lm = torch.as_tensor(lam[ix], device=device)
            masked = torch.rand(r.shape, generator=g).to(device) < mask_frac
            masked[masked.all(1)] = False                     # keep >= 1 visible band per pixel
            h = model.encode_pixels(r, lm, ~masked)           # masked bands are padding (not seen)
            tgt = torch.asinh(r / REFL_SCALE)
            wl = model.wl_embed(sinusoidal(lm))               # [N, B, d]
            pred = head(torch.cat([h[:, None, :].expand(-1, B, -1), wl], -1))[..., 0]
            loss = ((pred - tgt) ** 2)[masked].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        hist.append({"epoch": ep, "ssl_mse": tot / steps})
        log(f"[ssl] epoch {ep} mse {tot / steps:.4f}")
    return {"history": hist, "n_pixels": int(n), "manifest": man["name"], "pixels_sha256": man["pixels_sha256"]}
