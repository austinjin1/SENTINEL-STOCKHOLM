#!/usr/bin/env python
"""SENTINEL — cohesive end-to-end inference over the REAL, honest components.

Ties the two validated central-model pieces into one entry point:

  1. Sensor-stream early warning (Track A): the label-free, per-site-calibrated
     AquaSSM reconstruction-residual detector. Honest output = a sustained alarm
     at a known held-out false-positive rate, and (for a known event) the lead
     time — only trusted on sites that meet the FPR bar.

  2. Co-located multimodal fusion (Track B): real Sentinel-2 tile + date-aligned
     AquaSSM sensor window through the PerceiverIO fusion + head, giving a chl-a
     HAB-risk score. Honest output carries the caveat that, on real held-out
     data, fusion does NOT beat the best single encoder (AUROC ~0.65).

This is a demonstration of the integrated system on real inputs; every number it
emits traces to real data, and the simulation-only components (Stream GNN,
conformal, causal) are deliberately NOT wired in as if they were validated.

Usage:
    PYTHONNOUSERSITE=1 python scripts/sentinel_system.py --demo
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TA = ROOT / "scripts" / "experiments" / "trackA_leadtime"
TB = ROOT / "scripts" / "experiments" / "trackB_fusion"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SentinelSystem:
    """Cohesive inference over the real Track A detector + Track B fusion."""

    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.D = _load("trackA_detector", TA / "_detector.py")
        self.F = _load("trackB_fusion", TB / "_fusion.py")
        # Track A detector (NEON-fine-tuned if available, else base MPP backbone)
        _ft = ROOT / "checkpoints" / "sensor" / "aquassm_neon_ft.pt"
        _base = ROOT / "checkpoints" / "sensor" / "aquassm_v2_mpp_best.pt"
        self.detector = self.D.load_detector(_ft if _ft.exists() else _base, self.device)
        self.calib = json.load(open(ROOT / "results" / "trackA_leadtime" /
                                    "leadtime_results.json"))["sites"]
        # Track B encoders + fusion (loaded lazily)
        self._fusion = None
        self._sat = self._sen = None

    # ---- Track A: sensor-stream early warning ----
    def assess_sensor_stream(self, site: str):
        """Run the calibrated detector on a NEON site's extracted series."""
        npz = ROOT / "data" / "processed" / "trackA" / f"{site}.npz"
        if not npz.exists() or site not in self.calib:
            return {"site": site, "error": "no extracted series / calibration"}
        c = self.calib[site]
        if "tau" not in c:
            return {"site": site, "error": "site was not calibrated"}
        series = {k: v for k, v in np.load(npz, allow_pickle=True).items()}
        in_stats = {p: tuple(c["in_stats"][p]) for p in self.D.REAL_PARAMS}
        chans = {p: self.D.zscore_channel(series[p], series[f"mask_{p}"],
                                          in_stats[p][0], in_stats[p][1])
                 for p in self.D.REAL_PARAMS}
        ts = series["ts"]
        ends = np.arange(self.D.WIN, ts.size + 1, 8)
        X = np.zeros((len(ends), self.D.WIN, 6), "float32")
        for n, e in enumerate(ends):
            for j, p in zip(self.D.REAL_IDX, self.D.REAL_PARAMS):
                X[n, :, j] = chans[p][e - self.D.WIN:e]
        err = self.D.recon_errors(self.detector, X, self.device, batch=128)
        ec, es = np.array(c["err_center"]), np.array(c["err_scale"])
        s = ((err[:, self.D.REAL_IDX] - ec) / es).max(1)
        flags = s > c["tau"]
        # sustained (>=3 consecutive) alarm anywhere
        run = mx = 0
        for f in flags:
            run = run + 1 if f else 0
            mx = max(mx, run)
        return {"site": site, "role": c["role"], "tau": round(c["tau"], 3),
                "held_out_fpr": c.get("fpr_heldout"),
                "specificity_validated": c.get("fpr_heldout", 1) <= 0.05,
                "sustained_alarm": bool(mx >= 3),
                "lead_days_if_event": c.get("lead_days")}

    # ---- Track B: co-located multimodal fusion ----
    def _ensure_fusion(self):
        if self._fusion is not None:
            return
        ck = ROOT / "checkpoints" / "fusion" / "fusion_real_chl.pt"
        st = torch.load(ck, map_location="cpu", weights_only=False)
        self._fusion = self.F.FusionClassifier(num_latents=st["num_latents"]).to(self.device)
        # Load STRICTLY: the fusion checkpoint must match the current architecture
        # exactly.  strict=False here would silently run the ProjectionBank on
        # random-init weights (missing fc1/fc2/hidden_norm) and emit an unvalidated
        # chl_hab_risk, so we fail loudly if the checkpoint has drifted.
        missing, unexpected = self._fusion.load_state_dict(st["model_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"fusion checkpoint {ck.name} is stale relative to the current "
                f"FusionClassifier architecture: {len(missing)} missing key(s) "
                f"{list(missing)[:4]}... and {len(unexpected)} unexpected key(s) "
                f"{list(unexpected)[:4]}...  Regenerate it via "
                f"scripts/experiments/trackB_fusion/04_train_final.py.")
        self._fusion.eval()
        self._scal = st
        self._sat = self.F.load_satellite_encoder(self.device)
        self._sen = self.F.load_sensor_encoder(self.device)

    def assess_colocated(self, image: np.ndarray, site: str, date: str):
        """Sentinel-2 tile + date-aligned sensor window -> chl-a HAB risk score."""
        self._ensure_fusion()
        if self.F.SENSOR_STATS is None:
            z = np.load(ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz",
                        allow_pickle=True)
            self.F.set_sensor_stats_from_targets(z["targets"])
        w = self.F.build_sensor_window(site, date)
        if w is None:
            return {"site": site, "error": "insufficient sensor history"}
        sat = self.F.embed_satellite(self._sat, image[None], self.device)
        sen = self.F.embed_sensor(self._sen, w[0][None], w[1][None], self.device)
        sat = (sat - np.array(self._scal["sat_mean"])) / np.array(self._scal["sat_scale"])
        sen = (sen - np.array(self._scal["sen_mean"])) / np.array(self._scal["sen_scale"])
        with torch.no_grad():
            logit = self._fusion(sat=torch.tensor(sat, dtype=torch.float32, device=self.device),
                                 sensor=torch.tensor(sen, dtype=torch.float32, device=self.device))
            prob = torch.sigmoid(logit).item()
        return {"site": site, "date": date, "chl_hab_risk": round(prob, 3),
                "caveat": "fusion AUROC ~0.65 on held-out data; does NOT beat sensor-only (~0.64)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--site", default="KING")
    args = ap.parse_args()

    sys_ = SentinelSystem()
    print("=== SENTINEL cohesive assessment (real components) ===")
    print("\n[1] Sensor-stream early warning (Track A, calibrated detector):")
    print(json.dumps(sys_.assess_sensor_stream(args.site), indent=2, default=str))

    if args.demo:
        z = np.load(ROOT / "data" / "processed" / "satellite" / "drone_wq_pairs.npz",
                    allow_pickle=True)
        md = json.loads(str(z["metadata"]))
        i = 0
        print("\n[2] Co-located multimodal fusion (Track B, chl-a HAB risk):")
        print(json.dumps(sys_.assess_colocated(z["images"][i], md[i]["site_id"],
                                               md[i]["date"]), indent=2, default=str))


if __name__ == "__main__":
    main()
