#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write the M3 freeze CANDIDATE ``docs/preregistration/FREEZE_microbe_candidate.json`` (never a freeze:
``status: CANDIDATE`` + ``NOT_A_FREEZE``; ``sentinel.models.microbe.testlook.load_freeze`` refuses it).
The lead reviews it and writes ``FREEZE_microbe.json`` (status FROZEN, authorised_by) — only that unlocks
the test. Everything the scorer uses is pinned here: model + track, per-target H-M3 references (tier-0
choice from development LODO, stronger of tuned/untuned RF/XGB, +context parity baselines for +context
models), event σ/Platt maps, artifacts (sha256), data content pins, scoring-code sha256.

    python -B scripts/eval/microbe_freeze_candidate.py --model xgb_v4clr --track lodo --events-json <dev events json>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.models.microbe import testlook as TL


def strongest(scheme_dir: Path, cands, t, S) -> str:
    import pandas as pd

    from sentinel.models.microbe import evaluate as E

    best, bv = None, -1e9
    for c in cands:
        f = scheme_dir / f"{c}.parquet"
        if not f.exists():
            continue
        o = pd.read_parquet(f)
        v = E.summary(S.loc[o.index, t], o[t], S.loc[o.index, "domain_id"])["r2"]
        if v > bv:
            best, bv = c, v
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--track", default="lodo", choices=["lodo", "lodo_x001"])
    ap.add_argument("--events-json", required=True, help="output of scripts/eval/events_microbe.py for --model")
    ap.add_argument("--configs-tried", type=int, required=True)
    a = ap.parse_args(argv)
    from sentinel.models.microbe import baselines as B
    from sentinel.models.microbe.design import TARGET_NAMES, load_design

    S = load_design().samples
    lodo = paths.RUNS_ROOT / "microbe" / "cv" / "lodo"
    tier0 = json.loads((lodo / "tier0_best_choice.json").read_text())
    pass_refs, comps = {}, set()
    for t in TARGET_NAMES:
        refs = [tier0[t]["model"], strongest(lodo, ["rf_clr", "rf_clr_tuned"], t, S),
                strongest(lodo, ["xgb_clr", "xgb_clr_tuned"], t, S)]
        if a.model in B.CONTEXT_TABLES:
            refs.append(strongest(lodo, ["rf_clr_context", "xgb_clr_context"], t, S))
        pass_refs[t] = refs
        comps |= set(refs)
    ev = json.loads(Path(a.events_json).read_text())
    maps_path = TL.FREEZE_DIR / "microbe_event_maps.json"
    maps_path.write_text(json.dumps(ev["frozen_maps"], indent=1, default=float))
    art_files = ["docs/preregistration/microbe_split.json", "docs/preregistration/microbe_split_lodo_x001.json",
                 "docs/preregistration/microbe_event_maps.json", str(lodo / "tier0_best_choice.json")]
    artifacts = {f: TL.sha_file(f if Path(f).is_absolute() else paths.REPO_ROOT / f) for f in art_files}
    pins = {"microbe_wq": {"rel_dir": "processed/microbe_wq", "patterns": ["*.parquet"]},
            "microbe_wq_v1": {"rel_dir": "processed/microbe_wq_v1", "patterns": ["*.parquet"]}}
    enc = {B.SEQEMB.get(m) for m in [a.model] + sorted(comps)} - {None}
    for e in sorted(enc):
        pins[f"seqemb_{e}"] = {"rel_dir": "interim/microbe_seqemb", "patterns": [f"{e}.npy", f"{e}_ids.parquet"]}
    for p in pins.values():
        p["content_sha256"] = TL.content_pin(p["rel_dir"], p["patterns"])
    rec = {"NOT_A_FREEZE": "candidate for lead review; copy to FREEZE_microbe.json with status FROZEN + authorised_by",
           "status": "CANDIDATE", "authorised_by": "", "created": dt.datetime.now().isoformat(timespec="seconds"),
           "split_sha256": TL.sha_file(TL.FREEZE_DIR / "microbe_split.json"),
           "scoring_code_sha256": TL.scoring_code_sha256(), "scoring_code_files": TL.frozen_files(),
           "model": {"name": a.model, "track": a.track, "comparators": sorted(comps), "pass_refs": pass_refs,
                     "n_hypotheses": 2 * len(TARGET_NAMES)},
           "looks": {"M3/test": {"routes": ["timeout", "reserve"], "role": "primary"},
                     "M3/test_b": {"role": "secondary", "status": "data not yet available (RELEASE-2027)"}},
           "event_maps": "docs/preregistration/microbe_event_maps.json",
           "artifacts": artifacts, "data_snapshots": pins, "configs_tried": a.configs_tried}
    out = TL.FREEZE_DIR / TL.CANDIDATE_NAME
    out.write_text(json.dumps(rec, indent=1))
    print(out, rec["scoring_code_sha256"])


if __name__ == "__main__":
    main()
