#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit frozen test path (orbit_protocol.md §9): scores the HEADLINE cell (in-lake rows, test role x
2024-2025) ONCE with frozen artifacts. Requires --allow-test AND docs/preregistration/FREEZE_orbit.json
(guard.load_freeze verifies split, scoring code, data manifest). Every checkpoint / fitted-baseline file is
verified against FREEZE weights_sha256; comparators come only from FREEZE `comparators`.

Also reports: the >10-km isolation sensitivity, turbidity stratified by unit (W6), non-lake test rows and
train-role x test-year rows (secondary). Refuses to run twice (an existing results file stops it).

``--dry-run-dev`` runs the identical prediction/scoring code on the development selection cell (val role x val
years, in-lake) WITHOUT freeze or protected data, for checking against scripts/train/orbit_compare.py.

    python -B scripts/eval/orbit_test.py --dry-run-dev --satwq satwq_base_s0 ... --baselines baselines_dev_v3 \
        --comparators-json $SENTINEL_RUNS/orbit/compare_satwq_base5/comparators_val_selected.json
    python -B scripts/eval/orbit_test.py --allow-test            # after FREEZE_orbit.json exists
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.models.orbit import baselines as B  # noqa: E402
from sentinel.models.orbit import data as D  # noqa: E402
from sentinel.models.orbit import evaluate as E  # noqa: E402
from sentinel.models.orbit import features as F  # noqa: E402
from sentinel.models.orbit import guard  # noqa: E402
from sentinel.models.orbit import pipeline as PL  # noqa: E402
from sentinel.models.orbit.split import nearest_distance_km  # noqa: E402

MDN_B7 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")


def satwq_predict(ckpts, df, pix, device):
    import torch

    from sentinel.models.orbit.satwq import SatWQNet, mixture_quantiles, predict_mixture

    ws, mus, sgs = [], [], []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = SatWQNet(**ck["arch"]).to(device)
        m.load_state_dict(ck["state_dict"])
        nm = PL.Norm.from_dict(ck["norm"])
        ctx_off = bool(ck["config"]["finetune"]["aug"].get("ctx_off", False))
        w, mu, sg = predict_mixture(m, PL.make_set(df, pix, nm), device, nm.y_mean, nm.y_std, ctx_off=ctx_off)
        ws.append(w / len(ckpts))
        mus.append(mu)
        sgs.append(sg)
    w, mu, sg = np.concatenate(ws, -1), np.concatenate(mus, -1), np.concatenate(sgs, -1)
    med = {t: mixture_quantiles(w[:, j], mu[:, j], sg[:, j], (0.5,))[:, 0] for j, t in enumerate(D.TARGETS)}
    return med, (w, mu, sg)


def baseline_predict(name: str, t: str, fitted: dict, df, Ximg, Xctx):
    """Predictions of a (possibly variant) baseline for target t from the persisted fitted objects."""
    if name.startswith("gbm_imagery_context") or name.startswith("gbm_imagery"):
        base, _, v = name.partition("#v")
        X = pd.concat([Ximg, Xctx], axis=1) if base == "gbm_imagery_context" else Ximg
        return fitted[base](df, X, **({"variant": int(v)} if v else {}))[t]
    if name.startswith("location_knn"):
        base, _, k = name.partition("#k")
        return fitted[base](df, None, **({"k": int(k)} if k else {}))[t]
    if name == "gbm_season_location":
        return fitted[name](df, Xctx)[t]
    if name in ("global_median",):
        return fitted[name](df)[t]
    if name == "index_opt":
        return fitted[name](df, Ximg)[t]
    if re.match(r"^mdn_(b(7|12)(_\d+k_(lin|log))?|gloria)$", name):
        bands = D.BANDS if name.startswith("mdn_b12") else MDN_B7
        div = np.pi if name == "mdn_gloria" else 1.0
        Xl = np.stack([df[f"rw3_{b}"].to_numpy(float) / div for b in bands], 1)
        return fitted[(name, t)].predict(Xl)
    if name == "mdn_official":
        from sentinel.models.orbit.mdn import OfficialMDN

        off = OfficialMDN(paths.RAW / "mdn_stream_weights" / "MSI" / "x2025")
        po = off.predict_log10(np.stack([df[f"rw3_{b}"].to_numpy(float) / np.pi for b in OfficialMDN.MSI_BANDS], 1))
        return po.get({"chla": "chl", "tss": "tss"}.get(t, ""), np.full(len(df), np.nan))
    raise KeyError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--freeze-file", default=None)
    ap.add_argument("--dry-run-dev", action="store_true")
    ap.add_argument("--satwq", nargs="*", default=None, help="dry-run only (test: from FREEZE)")
    ap.add_argument("--baselines", default=None, help="dry-run only (test: from FREEZE)")
    ap.add_argument("--comparators-json", default=None, help="dry-run only (test: FREEZE comparators)")
    ap.add_argument("--label", default="sentinel_orbit")
    ap.add_argument("--n-boot", type=int, default=E.N_BOOT)
    ap.add_argument("--gpu", action="store_true", help="lock a GPU via the shared helper (cap 0,1; prefer 1) for inference; default CPU")
    ap.add_argument("--version", default=D.CACHE_VERSION, help="dry-run only (test: FREEZE data_version)")
    ap.add_argument("--blend-with", default=None, help="dry-run only (test: FREEZE blend_with): declared 50/50 "
                    "log10 hybrid with this baseline's inner-selected representative, as in orbit_compare.py")
    ap.add_argument("--track", default="headline", choices=["headline", "U"],
                    help="headline (+ track E strata) or track U (orbit_protocol.md §11.2; FREEZE track_u artifacts)")
    a = ap.parse_args()
    import cloudpickle

    if a.dry_run_dev:
        df, pix, info, masks = PL.load_dev(a.version)
        sel = masks[PL.PRIMARY_CELL]
        sel_b = None
        blend_with = a.blend_with
        dev_version = a.version
        ckpts = [paths.RUNS_ROOT / "orbit" / r / "model.pt" for r in a.satwq]
        fitted_path = paths.RUNS_ROOT / "orbit" / a.baselines / "fitted_baselines.pkl"
        comparators = json.loads(Path(a.comparators_json).read_text())
        out_dir = paths.run_dir("orbit", f"test_dryrun_{a.label}")
        exp_name = None
    else:
        freeze = guard.load_freeze(a.freeze_file)            # raises unless the freeze is valid
        if not a.allow_test:
            raise guard.ProtectedDataError("pass --allow-test to score the frozen headline cell")
        out_dir = paths.run_dir("orbit", f"TEST_{freeze['data_version']}" + ("" if a.track == "headline" else f"_track{a.track}"))
        if (out_dir / "test_metrics.csv").exists():
            raise RuntimeError(f"test already scored once ({out_dir}); RESEARCH_PLAN R6 forbids re-scoring")
        art = freeze if a.track == "headline" else freeze["track_u"]
        ckpts = art["satwq_checkpoints"]
        for c in ckpts:
            guard.verify_weights(freeze, Path(c).parent.name, c)
        fitted_path = art["fitted_baselines"]
        guard.verify_weights(freeze, "fitted_baselines" if a.track == "headline" else "fitted_baselines_track_u",
                             fitted_path)
        blend_with = freeze.get("blend_with")
        dev_version = freeze["data_version"]
        comparators = freeze["comparators"]
        df_all, info = D.build_table(allow_test=True, freeze_file=a.freeze_file)
        df = df_all[df_all.protected.to_numpy()].reset_index(drop=True)
        pix = D.extract_pixels(df)
        sel_b = None
        if a.track == "U":
            from sentinel.models.orbit import track_u as U

            U.verify_track_file()
            sel = U.test_mask(df, U.assign(df.anchor_lon, df.anchor_lat))
        else:
            sel = ((df.role == "test") & (df.tblock == "test") & df.in_lake & ~df.buffer_excluded).to_numpy()
            # Test-B (protocol v0.3 §9a, declared 2026-11-08 before any test look): test-role lakes x 2026-01-01..
            # 2026-09-30 (the WY2026 part not already in the 2024-2025 headline); reported separately and pooled
            sel_b = ((df.role == "test") & (df.tblock == "test_b") & df.in_lake & ~df.buffer_excluded
                     & (pd.to_datetime(df.date) <= pd.Timestamp("2026-09-30"))).to_numpy()
        exp_name = f"orbit_TEST_{freeze['data_version']}" + ("" if a.track == "headline" else f"_track{a.track}")
    if a.gpu:
        PL.setup_gpu(f"orbit_test_{a.label}")      # shared lock helper (GPUs 0/1 cap); never an unlocked device
    import torch

    torch.set_num_threads(2)
    device = "cuda" if a.gpu else "cpu"
    fitted = cloudpickle.loads(Path(fitted_path).read_bytes())["fitted"]
    cells = {"headline": sel}
    if sel_b is not None:
        cells.update({"testB_2026": sel_b, "pooled_2024_2026": sel | sel_b})
    rows, pairs = [], []
    for cell_name, sel in cells.items():
        if sel.sum() == 0:
            continue
        sub = df[sel].reset_index(drop=True)
        psub = {k: v[sel] for k, v in pix.items()}
        med, _ = satwq_predict(ckpts, sub, psub, device)
        Ximg, Xctx = F.image_features(sub, psub), B.location_season_matrix(sub)
        # isolation sensitivity: rows > 10 km from any train-role point of the development table
        dev_tr = D.load_dev(dev_version, training=False)[0] if not a.dry_run_dev else df[df.role == "train"]
        dev_tr = dev_tr[dev_tr.role == "train"]
        dist = nearest_distance_km(sub.lat, sub.lon, dev_tr.lat, dev_tr.lon)
        for t in D.TARGETS:
            y = np.log10(sub[t].to_numpy(float))
            if np.isfinite(y).sum() < 20:
                continue
            preds = {a.label: med[t]}
            if blend_with:
                preds[a.label] = 0.5 * med[t] + 0.5 * baseline_predict(blend_with, t, fitted, sub, Ximg, Xctx)
            for fam, c in comparators.get(t, {}).items():
                preds[f"{fam}[{c['model']}]"] = baseline_predict(c["model"], t, fitted, sub, Ximg, Xctx)
            if t in ("chla", "tss"):
                preds["mdn_official"] = baseline_predict("mdn_official", t, fitted, sub, Ximg, Xctx)
            strata = {"all": (np.ones(len(sub), bool), y), "isolated_gt10km": (dist > 10, y)}
            if t == "chla":
                # track E (orbit_protocol.md §11.1): a-priori bloom-extreme strata, thresholds from the frozen rule file
                trk = json.loads((paths.REPO_ROOT / "docs" / "preregistration" / "orbit_track_u.json").read_text())["track_E"]
                for thr in list(trk["thresholds_ugL_fixed"]) + [trk["threshold_fit_p95_ugL"]]:
                    strata[f"trackE_chla_ge{thr:g}"] = (y >= np.log10(thr), y)
            if t == "turbidity":
                # protocol v0.3 §2.1: primary turbidity = NTU family (+ AquaCorr unspecified); FNU and the pooled
                # all-unit value are separate, secondary strata (never mixed in the primary)
                cls = sub.turbidity_unit_class.to_numpy()
                for u in ("NTU", "unspecified"):
                    strata[f"unit={u}"] = (cls == u, y)
                for col, sname in (("turbidity_fnu", "unit=FNU"), ("turbidity_all_units", "pooled_all_units")):
                    if col in sub:
                        strata[sname] = (np.ones(len(sub), bool), np.log10(sub[col].to_numpy(float)))
            for sname, (sm, ys) in strata.items():
                yy = np.where(sm, ys, np.nan)
                if np.isfinite(yy).sum() < 20:
                    continue
                tab, ok = E.score_models(yy, preds, sub.block.to_numpy(), n_boot=a.n_boot)
                tab.insert(0, "stratum", sname)
                tab.insert(0, "cell", cell_name)
                tab.insert(0, "target", t)
                rows.append(tab)
                for comp in [k for k in preds if k != a.label]:
                    r = E.paired_mdsa(yy[ok], preds[a.label][ok], preds[comp][ok], sub.block.to_numpy()[ok], n_boot=a.n_boot)
                    pairs.append({"cell": cell_name, "target": t, "stratum": sname, "comparator": comp, **r})
    res, pr = pd.concat(rows, ignore_index=True), pd.DataFrame(pairs)
    res.to_csv(out_dir / ("test_metrics.csv" if exp_name else "dryrun_metrics.csv"), index=False)
    pr.to_csv(out_dir / ("test_paired.csv" if exp_name else "dryrun_paired.csv"), index=False)
    print(res.round(3).to_string(index=False))
    print(pr.round(3).to_string(index=False))
    if exp_name:
        from sentinel.core.registry import log_result

        for r in res.itertuples():
            log_result(workstream=PL.WORKSTREAM, experiment=exp_name, model=r.model, metric="mdsa", value=r.mdsa,
                       split=f"{r.cell}:{r.stratum}", n=r.n, ci_low=r.mdsa_ci_low, ci_high=r.mdsa_ci_high,
                       ci_method="role-block cluster bootstrap (B=2000)", target=r.target,
                       is_baseline=r.model != a.label, notes="TEST (scored once after freeze)")
        for p in pr.itertuples():
            log_result(workstream=PL.WORKSTREAM, experiment=exp_name, model=a.label, metric=f"mdsa_diff_vs_{p.comparator}",
                       value=p.diff, split=f"{p.cell}:{p.stratum}", n=p.n, ci_low=p.ci_low, ci_high=p.ci_high,
                       target=p.target, ci_method="paired role-block cluster bootstrap (B=2000)",
                       extra={"p_value": p.p_value}, notes="TEST (scored once after freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
