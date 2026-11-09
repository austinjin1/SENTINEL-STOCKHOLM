#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Score saved SENTINEL-HydroMet prediction files (one or more dirs) on IDENTICAL cells: per-site
metrics, medians with basin-group-bootstrap CIs (site bootstrap as sensitivity), pooled cross-site
skill, seed ensembles and — with --primary — the pre-registered comparison (protocol §6–§7). Test
blocks / test-role sites need --allow-test AND docs/preregistration/FREEZE_hydromet.json, and every
prediction file must carry the frozen artifact hash and the current freeze hash.
Logic: sentinel.models.hydromet.pipeline.

    python -B scripts/eval/hydromet_score.py --config configs/hydromet/common.yaml \\
        --predictions <run1>/predictions <run2>/predictions --block val --roles val --primary [--register]
"""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.models.hydromet import runner


def main() -> int:
    ap = runner.base_parser(__doc__)
    ap.add_argument("--predictions", nargs="+", required=True, help="prediction dirs (*__<block>.npz)")
    ap.add_argument("--block", default="val")
    ap.add_argument("--roles", default="val")
    ap.add_argument("--refs", default="", help="comma list of reference models for descriptive paired differences")
    ap.add_argument("--out", default=None)
    ap.add_argument("--primary", action="store_true", help="pre-registered primary comparison (protocol §6-§7)")
    ap.add_argument("--register", action="store_true", help="write to the registry (FINAL split, non-smoke only)")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--comparators", default=None, help="pre-declared comparators JSON (track R, protocol §10)")
    ap.add_argument("--models", default="", help="comma list of SENTINEL models in the primary family (default: all)")
    a = ap.parse_args()
    a.eval_blocks = a.block
    if a.allow_test or a.block in ("test", "test_b") or "test" in a.roles:
        raise SystemExit("protected data (test blocks / test-role sites) are scored ONLY through "
                         "scripts/eval/hydromet_test_look.py (single-pass guard; models, comparators and the Holm family "
                         "come from the freeze block)")
    runner.env_threads(8)
    cfg = runner.load_config(a.config, smoke=a.smoke)
    if a.track_r:   # protocol §10 (leave-region-out track)
        cfg["track_r"] = a.track_r
        cfg["secondary_eval_roles"] = []
    roles = tuple(r.strip() for r in a.roles.split(",") if r.strip())
    out = Path(a.out) if a.out else runner.run_directory(f"score_{a.block}", cfg, a.run_name)
    log = runner.make_logger(out / "log.txt")
    from sentinel.models.hydromet import pipeline

    data, _, _ = runner.prepare(a, cfg, out, training=False, log=log, roles=roles)
    si, ti = runner.eval_cells(data, a.block, roles)
    preds, sources = pipeline.load_prediction_set(a.predictions, data, si, ti, a.block, smoke=cfg["_smoke"],
                                                  access=data.access)
    log(f"[score] models: {sorted(preds)}")
    res = pipeline.score(data, preds, si, ti, a.block, roles, out, refs=tuple(r for r in a.refs.split(",") if r),
                         n_boot=int(cfg.get("n_boot", 2000)), min_days=int(cfg.get("min_days_per_site", 60)),
                         primary=a.primary, access=data.access, log=log,
                         fixed_comparators=json.loads(Path(a.comparators).read_text()) if a.comparators else None,
                         primary_models=[m for m in a.models.split(",") if m] or None)
    runner.write_json(out / "sources.json", sources)
    runner.print_summary(res["summary"], f"[score] {'SMOKE ' if cfg['_smoke'] else ''}{a.block} block, "
                                         f"{'/'.join(roles)}-role sites: median over sites [basin-group bootstrap 95% CI]",
                         log)
    if a.register:
        exp = a.experiment or f"M6/{a.block}"
        n = runner.register_summary(res["summary"], cfg, data.split_info, experiment=exp, block=a.block,
                                    artifacts=list(sources.values()), roles=roles)
        n += runner.register_pooled_and_primary(res["pooled"], res["primary"], cfg, data.split_info, experiment=exp,
                                                block=a.block, roles=roles, artifacts=list(sources.values()))
        n += runner.register_secondary(res.get("secondary"), cfg, data.split_info, experiment=exp, block=a.block,
                                       roles=roles, artifacts=list(sources.values()))
        log(f"[registry] {n} entries ({exp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
