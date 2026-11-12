#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Write the M5-X freeze-CANDIDATE record ``docs/preregistration/bioassess_x_freeze_candidate.json`` (protocol §9).

Built from the development selection (``x_track/dev/selection_x_v1.json``). It fixes, for the single pass:
candidates per block (C-X + ≤ 2 secondaries) and per-target comparators as executable specs (model, variant per
target, ensemble members); EVERY listed §6 comparator of the block with its spec and development scores, and the
methods "not compared, with reason" (GJAM with its fidelity record, TRPI) — amendments v9, v10; the two rule
comparators of v11 item 1 (``cxr_nobioset``, ``cxr_swap``), checked with the scorer's own ``check_listed``;
seeds, B (2,000 marginal / 10,000 confirmatory), α and the 12-test Holm family; the
decision table; configuration counts and the selection note; sha256 of every code/config/protocol/split/pool/fold/
frozen-split file (every file of the scoring import closure singly), of the data files, of the TabPFN weights, of
the cached training design and frozen-key cache; Python and R package versions; the reproducibility tolerances
(v11 item 6). It copies the pinned files to a read-only code snapshot under ``sentinel_runs/bioassess_x/`` and
records it (v11 item 3). It never creates the FREEZE file (the lead does, after the independent pre-test review).
No model is fitted here. Run it from the live repository only.

    python -B scripts/eval/bioassess_x_freeze_candidate.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from sentinel.core import paths
from sentinel.core.registry import file_sha256
from sentinel.models.bioassess import xdev as XD
from sentinel.models.bioassess import xtrack as X

REPO = paths.REPO_ROOT
OUT = REPO / "docs/preregistration/bioassess_x_freeze_candidate.json"
SEL = XD.DEV_DIR / "selection_x_v1.json"
CODE = ["docs/preregistration/bioassess_x_protocol.md", "configs/bioassess/bioset.yaml", "configs/bioassess/baselines.yaml",
        "sentinel/models/bioassess/xtrack.py", "sentinel/models/bioassess/xdata.py", "sentinel/models/bioassess/xdev.py",
        "sentinel/models/bioassess/xbioset.py", "sentinel/models/bioassess/xtestlook.py",
        "sentinel/models/bioassess/data.py", "sentinel/models/bioassess/uk.py",
        "sentinel/models/bioassess/features.py", "sentinel/models/bioassess/baselines.py",
        "sentinel/models/bioassess/bioset.py", "sentinel/models/bioassess/cv.py", "sentinel/models/bioassess/splits.py",
        "sentinel/models/bioassess/pretrain.py", "sentinel/models/bioassess/evaluate.py",
        "sentinel/models/bioassess/__init__.py", "sentinel/models/orbit/split.py",
        "sentinel/eval/splits.py", "sentinel/eval/bootstrap.py", "sentinel/eval/metrics.py", "sentinel/core/seed.py",
        "sentinel/core/registry.py", "sentinel/core/paths.py", "sentinel/core/gpu.py", "sentinel/core/runlog.py",
        "sentinel/models/chemtox/tabpfn_model.py", "scripts/eval/bioassess_x_final.py", "scripts/eval/bioassess_x_dev.py",
        "scripts/eval/bioassess_x_bioinfer.R", "scripts/eval/bioassess_x_gjam.R", "scripts/train/bioassess_x_bioset.py",
        "scripts/eval/bioassess_x_select.py", "scripts/eval/bioassess_x_freeze_candidate.py",
        "tests/test_bioassess_models.py", "tests/test_bioassess_x_dev.py", "sentinel/events/bioassess_x.py",
        "scripts/eval/bioassess_x_events.py", *X.FROZEN_SPLIT_FILES.values()]
RUN_FILES = [XD.XDIR / "splits_x_v1.parquet", XD.XDIR / "splits_x_v1.json", XD.XDIR / "pools_x_v1.json",
             XD.XDIR / "power_x_v1.json", XD.DEV_DIR / "folds_dev_v1.json", SEL]
DATA = ["processed/nars_streams/visits.parquet", "processed/nars_streams/benthic_counts.parquet",
        "processed/nars_streams/chemistry_wide.parquet", "processed/nars_streams/chemistry_long.parquet",
        "processed/nars_lakes/visits.parquet", "processed/nars_lakes/benthic_counts.parquet",
        "processed/nars_lakes/chemistry_wide.parquet", "processed/nars_lakes/chemistry_long.parquet",
        "processed/taxonomy_harmonised/taxon_map.parquet", "processed/uk_ea_ecology/sites.parquet",
        "processed/uk_ea_ecology/samples_invertebrate.parquet", "processed/uk_ea_ecology/counts_invertebrate.parquet",
        "processed/uk_ea_ecology/chemistry_matched_wide_pm30d_invertebrate.parquet",
        "processed/fuse_tier_c/rows.parquet", "processed/s2_wqp_matchups_hist/samples.parquet"]
PACKAGES = ["tabpfn", "torch", "scikit-learn", "lightgbm", "numpy", "pandas", "scipy", "joblib", "pyarrow", "PyYAML"]
TABPFN_CKPT = Path.home() / ".cache/tabpfn/tabpfn-v2-regressor.ckpt"
BASE_ENS = {"bens_rfg_tabpfn": ("rf_community_geography", "tabpfn_community"),
            "bens_rfg_gbm_tabpfn": ("rf_community_geography", "gbm_community", "tabpfn_community"),
            "bens_rfctx_tabpfn": ("rf_community_context", "tabpfn_community")}


# Every §6 comparator fitted and compared at the look (protocol amendment v9 item 1), by SOTA-campaign class.
RULE = ("cxr_nobioset", "cxr_swap")        # protocol v11 item 1: derived by rule from the selected C-X
LISTED_EXTRA = ("bioinfer", *BASE_ENS, *RULE)
REHEARSAL_FILE = paths.RUNS_ROOT / "bioassess_x" / "rehearsal_v11.json"     # written after the X-B4 rehearsal
CLASS = {"published community-to-chemistry method": ("wapls2", "knn_community", "bioinfer"),
         "simple or practitioner baseline": ("region_mean", "rf_geography", "nearest_reference", "idx_linear",
                                             "idx_linear_geo", "idx_rf", "idx_rf_geo"),
         "machine-learning baseline": ("rf_community", "gbm_community", "tabpfn_community", "rf_community_geography",
                                       "rf_community_context"),
         "baseline ensemble (protocol §6 item 7)": tuple(BASE_ENS),
         "rule comparator (protocol v11 item 1): C-X without BioSet / with BioSet replaced": RULE}
GJAM_FILES = {"info": "r_out/gjam/{b}/fold00_info.txt", "log": "logs/gjam_{b}_fold00.log"}
GJAM_V4CHECK = "r_out/gjam/X-L/fold00_v4check_info.txt"


def listed_names(block: str) -> list[str]:
    return [*XD.models_for(block), *LISTED_EXTRA]


def comparator_class(name: str) -> str:
    (cls,) = [c for c, names in CLASS.items() if name in names]
    return cls


def parse_r_print(text: str) -> dict:
    """Parse R ``print(list)`` output: ``$name`` blocks holding ``[1] value`` or alternating name / value lines."""
    blocks, key = {}, None
    for ln in text.splitlines():
        if ln.startswith("$"):
            key = ln[1:].strip()
            blocks[key] = []
        elif not ln.strip():
            key = None
        elif key is not None:
            blocks[key].append(ln)
    out = {}
    for k, lines in blocks.items():
        if not lines:
            continue
        if lines[0].lstrip().startswith("[1]"):
            v = lines[0].split()[1]
            out[k] = {"TRUE": True, "FALSE": False}.get(v, None) if v in ("TRUE", "FALSE") else float(v)
        else:
            if len(lines) % 2:
                raise ValueError(f"unpaired name/value lines in ${k}")
            names = [x for i in range(0, len(lines), 2) for x in lines[i].split()]
            vals = [float(x) for i in range(1, len(lines), 2) for x in lines[i].split()]
            if len(names) != len(vals):
                raise ValueError(f"names and values differ in ${k}")
            out[k] = dict(zip(names, vals))
    return out


def gjam_fidelity(block: str) -> dict:
    """The full-size fidelity record of fold 0 (amendment v6), per target, from the run log (the ``_info.txt`` file is
    R ``str()`` output and shortens the vectors); both files are listed with their sha256."""
    files = {k: XD.DEV_DIR / v.format(b=block) for k, v in GJAM_FILES.items()}
    info = parse_r_print(files["log"].read_text())
    T = list(XD.TARGETS[block])
    r_in, r_out = info["fidelity_r_insample_mode"], info["fidelity_r_outofsample_mode"]
    n_fin = list(info["xpredMu_n_finite"].values())
    if list(r_in) != T or list(r_out) != T or len(n_fin) != len(T):
        raise ValueError(f"GJAM fidelity log of {block} does not list the block's targets in order")
    ok = all(v >= 0.90 for v in r_in.values())
    if info["fidelity_pass"] is not False or ok:
        raise ValueError(f"GJAM fidelity of {block} is not a recorded failure: GJAM would have to be a comparator")
    return {"fidelity_pass": False, "criterion": "Pearson r >= 0.90 for every target (our routine in gjam's "
            "in-sample scheme vs gjam's own xpredMu, fold 0, fit rows)", "fold": 0,
            "n_fit_rows": int(info["n_fit"]), "n_taxa": int(info["n_taxa"]), "fit_minutes": info["fit_minutes"],
            "per_target": {t: {"r_insample_scheme_vs_xpredMu": r_in[t], "r_outofsample_mode_vs_xpredMu": r_out[t],
                               "n_rows_xpredMu_finite": int(n_fin[i]), "passes_0.90": bool(r_in[t] >= 0.90)}
                           for i, t in enumerate(T)},
            "files": {str(f): file_sha256(f) for f in files.values()}}


def not_compared() -> dict:
    """Methods "not compared, with reason" (amendment v9 items 5-6; lead ruling of 2026-11-10 on GJAM)."""
    fid = {b: gjam_fidelity(b) for b in XD.BLOCKS}
    v4 = XD.DEV_DIR / GJAM_V4CHECK
    mins = {b: fid[b]["fit_minutes"] for b in XD.BLOCKS}
    return {
        "gjam": {
            "method": "GJAM-style joint-species inverse prediction (protocol §6 item 4; R package gjam)",
            "status": "not compared, with reason: failed the pre-declared fidelity check (protocol v4.8 item 8 as "
                      "corrected in v6)",
            "ground_used": "The fidelity check failed in both blocks: the in-sample-scheme r is below 0.90 for at "
                           "least one target in each block (per_target below). The ground was fixed before any GJAM "
                           "development prediction was scored (v4.8 item 8; v6).",
            "time_ground_not_used": "The 2-CPU-hour ground of §6 item 4 is NOT the ground used. Measured gjam fit "
                                    f"times: {mins['X-L']:.1f} min (X-L fold 0, {fid['X-L']['n_fit_rows']} rows) and "
                                    f"{mins['X-U']:.1f} min (X-U fold 0, {fid['X-U']['n_fit_rows']} rows).",
            "fidelity": fid,
            "first_check_v4_8": {"note": "The check as first written in v4.8 item 8 was run first on X-L fold 0 and "
                                         "failed; v6 corrected the check after that failure was seen (disclosed in "
                                         "DISCLOSURE_GJAM_FIDELITY_AMENDMENT_V6).",
                                 "file": str(v4), "sha256": file_sha256(v4), "text": v4.read_text()},
            "run_once_no_tuning": "The corrected check was run once, on fold 0 of each block. It is not repeated and "
                                  "gjam is not tuned further to pass it.",
            "test_rows": "No GJAM prediction is fitted or scored on test rows.",
            "limitation": "One published joint-species method (GJAM) could not be reproduced faithfully and is not "
                          "in the comparison. This is stated in every report.",
            "claim_wording": "No claim may say that C-X beats all joint-species models; GJAM is named as not "
                             "compared."},
        "trpi": {
            "method": "TRPI (UK pressure index; protocol §6 item 5)",
            "status": "not compared, with reason: TRPI is not in the EA dataset and its published taxon scores were "
                      "not obtained under licence before the freeze (protocol §6 item 5; v9 item 6)"}}


def comparator_spec(name: str, choice: dict, rule: dict | None = None) -> dict:
    if name in RULE:
        if not rule or name not in rule:
            raise ValueError(f"{name}: the selection holds no rule comparators (run bioassess_x_select.py first)")
        return {"kind": "mean", "members": [comparator_spec(m, choice) for m in rule[name]["members"]]}
    if name in BASE_ENS:
        return {"kind": "mean", "members": [comparator_spec(m, choice) for m in BASE_ENS[name]]}
    if name in ("bioinfer", "gjam"):
        return {"kind": "r", "method": name}
    return {"kind": "baseline", "name": name, "variant": dict(choice.get(name, {}))}


def sentinel_spec(name: str, blk: dict) -> dict:
    choice = blk["variant_choice"]
    if name.startswith("bsx_"):
        return {"kind": "bioset", "config": name}
    best = {"kind": "bioset", "config": blk["best_bioset"]}
    if name == "ens_c_form":
        return {"kind": "mean", "members": [best, comparator_spec("rf_community_geography", choice),
                                            comparator_spec("tabpfn_community", choice)]}
    if name == "ens_top2":
        return {"kind": "mean", "members": [best] + [comparator_spec(m, choice) for m in blk["ens_top2_members"]]}
    raise ValueError(name)


def _scorer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("x_final", REPO / "scripts/eval/bioassess_x_final.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event_track(sel: dict) -> dict:
    """Secondary event track (amendment v8): specs, dev power, test-block capacity, frozen calibration data."""
    from sentinel.events import bioassess_x as XE

    evm = REPO / "sentinel/eval/events.py"
    devf, calf = XD.DEV_DIR / "events" / "events_dev_v1.json", XD.DEV_DIR / "events" / "events_calibration_v1.parquet"
    et = {"amendment": "v8", "role": "SECONDARY (never in the 12-test confirmatory family); scored in the same single "
                                    "M5-X look; original M5 test blocks are spent and unused",
          "events": {f"{b}|{c}": {"target": t, "shared_event_code": XE.EVENT_CODE[c],
                                  "direct_head": (b, c) in XE.POWERED,
                                  "status": "descriptive only" if (b, c) in XE.DESCRIPTIVE_ONLY else "secondary"}
                     for (b, c), (t, _, _) in XE.TRACK.items()},
          "descriptive_only_note": "Power rule (>= 30 positives and >= 10 clusters on the development pseudo-test) "
                                   "governs (amendment v8a: the lead corrected an earlier instruction that conflicted "
                                   "with it, before any event output existed). Powered: X-L E5b, X-L E6, X-U E6 (35/25). "
                                   "Descriptive only: X-U E2a (19/7), X-L E2a (11/11), X-L E3 (11/11).",
          "threshold_baselines": {b: {m: comparator_spec(m, sel["blocks"][b]["variant_choice"]) for m in XE.COMPARATORS}
                                  for b in XD.BLOCKS}}
    if not evm.is_file() or not devf.is_file() or not calf.is_file():
        et["status"] = ("BLOCKED: " + ("shared sentinel/eval/events.py not built yet; " if not evm.is_file() else "")
                        + ("development event run not done (waits for the lead's events.py v1 release notice)"
                           if not devf.is_file() else ""))
        return et
    dev = json.loads(devf.read_text())
    et.update({"status": "active", "events_module_sha256": file_sha256(evm), "calibration_file": str(calf),
               "calibration_sha256": file_sha256(calf), "dev_results_file": str(devf),
               "dev_results_sha256": file_sha256(devf)})
    cap = []
    for k, v in dev["events"].items():
        et["events"][k].update({"power_dev": v["power_dev"], "projection": v["projection"]})
        pr = v["projection"]["projected_test_positives"]
        cap.append(f"{k}: dev {v['power_dev']['n_pos']} positives / {v['power_dev']['n_clusters_pos']} clusters "
                   f"({v['power_dev']['flag']}); projected test positives ≈ {pr:.0f} of {v['projection']['n_test_split']} "
                   f"rows -> {'can carry it (secondary)' if (pr >= 30 and et['events'][k]['status'] == 'secondary') else 'descriptive only'}")
    et["test_block_capacity"] = ("Test labels unread; projection = development prevalence x development label "
                                 "availability x test rows. The plan's power rule is re-applied on the test rows inside "
                                 "the look. " + " | ".join(cap))
    return et


def build() -> dict:
    from sentinel.models.bioassess import xtestlook as XL

    if (REPO / XL.SNAPSHOT_MARKER).is_file():
        raise SystemExit("the candidate record is generated from the live repository, never from a code snapshot")
    F = _scorer()
    S, frozen, _ = XD.load_context()
    for b, fl in XD.read_folds(XD.DEV_DIR / "folds_dev_v1.json", S, frozen).items():
        for fo in fl:                                   # re-verify the development folds the selection used
            X.assert_fit_ids(XD.keys_of(fo.train), S, frozen)
    sel = json.loads(SEL.read_text())
    power = json.loads((XD.XDIR / "power_x_v1.json").read_text())
    gj = {}
    for b in XD.BLOCKS:
        f = XD.DEV_DIR / "r_out" / "gjam" / b / "fold00_info.txt"
        gj[b] = f.read_text() if f.exists() else "missing"
    rec = {"status": "candidate", "prepared": dt.datetime.now().isoformat(timespec="seconds"),
           "DISCLOSURE_GJAM_FIDELITY_AMENDMENT_V6": (
               "PROMINENT DISCLOSURE: the GJAM fidelity check pre-declared in amendment v4.8 was run first and FAILED "
               "(X-L fold 0: r = 0.43 TP, 0.52 TN, 0.54 chloride; NA for conductivity, turbidity, pH; "
               "x_track/dev/r_out/gjam/X-L/fold00_v4check_info.txt). Amendment v6, which replaces the comparison by "
               "our routine run in gjam's in-sample scheme, was written AFTER that failure was seen (but before any "
               "GJAM development prediction was scored). It can only add a comparator. Full-size v6 outcome per "
               "block is attached below; GJAM is a comparator only if v6 passed in both blocks. OUTCOME: v6 failed "
               "in both blocks, so GJAM is 'not compared, with reason' (see not_compared.gjam; amendment v9 item 5)."),
           "gjam_v6_fidelity_info": gj,
           "protocol": "docs/preregistration/bioassess_x_protocol.md", "seeds": [0, 1, 2], "n_boot": 2000,
           "n_boot_confirmatory": 10000, "alpha": 0.05,
           "family": "C-X vs per-target comparator on r2_anomaly, 2 blocks x 6 targets = 12 tests, Holm (alpha 0.05); "
                     "p = two-sided paired location-cluster bootstrap p if delta > 0 else 1. v11 item 1: the "
                     "per-target comparator is re-chosen among the enlarged listed set (X-L 16, X-U 19) by the "
                     "unchanged §6 rule; any Holm family formed over listed comparators (events: derived vs each "
                     "comparator) grows with the list; no denominator is held at its earlier size.",
           "decision_table": ["Protocol §8 as amended by v9, v10 and v11; first matching row. 'Listed comparator' = "
                              "every comparator in comparator_set of the block (X-L 16, X-U 19, the two rule "
                              "comparators of v11 item 1 included); each is compared with C-X by the paired "
                              "location-cluster bootstrap of r2_anomaly (B = 10,000, seed-mean predictions).",
                              "3* (v10, checked first) paired CI upper < 0 against ANY listed comparator -> Below",
                              "1 Holm-rejected & per-seed delta > 0 (seeds 0,1,2) & region-unit paired CI > 0 & paired "
                              "CI lower > 0 against EVERY listed comparator (no finite estimate = not beaten) -> SOTA "
                              "(M5-X, block, within-region)",
                              "2 Holm-rejected (row 1 not met) -> Above, not robust; the reason names each failed "
                              "condition and each listed comparator not beaten",
                              "3 CI upper < 0 against the per-target comparator -> Below",
                              "4 CI includes 0 & realised Holm-worst MDE (3.71 * CI width / 3.92) > 0.02 -> Inconclusive "
                              "(underpowered)",
                              "5 otherwise -> Competitive",
                              "A family delta with no finite estimate -> Not estimable (reported, never a claim)",
                              "v11 item 6d: a confirmatory row in which a label-deciding quantity lies within the "
                              "reproducibility tolerance of its threshold is flagged; the label stands. 6e: a "
                              "bootstrap p of 0 is reported as '< 2/B'."],
           "power_x": power,
           "DISCLOSURE_POWER_TABLE_OPTIMISTIC_V11": (
               "Protocol v11 item 5: the registered power table (power_x; a proxy from NRSA Test A) is optimistic. "
               "The pre-test reviewer scaled the development paired CIs of C-X vs comparator to the test cluster "
               "counts: projected test SE 0.008-0.020 on X-L (registered 0.006-0.010) and 0.006-0.026 on X-U "
               "(registered 0.004-0.0065); projected Holm-worst detectable difference above 0.02 for all 12 targets "
               "(X-L 0.031-0.073, X-U 0.022-0.098). Most rows whose CI spans 0 are therefore expected to read "
               "'Inconclusive (underpowered)'. The decision table and the 0.02 threshold are unchanged; row 4 uses "
               "the realised CI."),
           "candidates": {}, "comparators": {}, "comparator_set": {}, "rule_comparators": {}, "dev_evidence": {}}
    n_conf = {}
    for b in XD.BLOCKS:
        blk = sel["blocks"][b]
        sel_name, sec = blk["selected"][0], blk["secondary"]
        rec["candidates"][b] = {"confirmatory": {"name": sel_name, "spec": sentinel_spec(sel_name, blk)},
                                "secondary": [{"name": m, "spec": sentinel_spec(m, blk)} for m in sec]}
        rule = blk.get("rule_comparators") or {}
        rec["comparators"][b] = {t: {"name": v["model"],
                                     "spec": comparator_spec(v["model"], blk["variant_choice"], rule),
                                     "dev_r2_anomaly": v["dev_r2_anomaly"]}
                                 for t, v in blk["per_target_comparator"].items()}
        choice = blk["variant_choice"]
        if any(v != "same" for v in choice.get("bioinfer", {}).values()):
            raise ValueError("bio.infer is run in the same-continent variant only")
        dev = {t: dict(map(tuple, v["ranking"])) for t, v in blk["per_target_comparator"].items()}
        names = listed_names(b)
        short = [m for m in names if any(m not in dev[t] for t in dev)]
        if short:
            raise ValueError(f"{b}: listed comparators without a development score for every target: {short}")
        rec["comparator_set"][b] = {m: {"spec": comparator_spec(m, choice, rule), "class": comparator_class(m),
                                        "dev_r2_anomaly": {t: dev[t][m] for t in dev},
                                        "dev_mean_r2_anomaly": blk["comparator_mean_r2_anomaly"][m]} for m in names}
        rec["rule_comparators"][b] = {
            "rule": "protocol v11 item 1 (lead ruling X-B3): cxr_nobioset = equal-weight mean of C-X's baseline "
                    "members; cxr_swap = those plus the listed single baseline (not an ensemble, not a member of "
                    "C-X, bio.infer and GJAM excluded) with the highest mean development r2_anomaly over the "
                    "block's six targets on seed 0. No new fit; both compete for the per-target comparator.",
            "cx": sel_name, "cxr_nobioset_members": rule["cxr_nobioset"]["members"],
            "cxr_swap_members": rule["cxr_swap"]["members"], "swap_member": rule["cxr_swap"]["swap"],
            "swap_member_dev_mean_r2_anomaly": rule["cxr_swap"]["swap_dev_mean_r2_anomaly"],
            "singles_ranked": rule["cxr_swap"]["singles_ranked"],
            "same_members_as_listed_ensemble": {r: rule[r].get("same_members_as", []) for r in RULE},
            "same_members_note": "A rule comparator with the same members as a listed baseline ensemble is the same "
                                 "model. It is still listed under its rule name (v11 item 1); in the development "
                                 "ranking it takes the ensemble's predictions bit for bit, so the tie for the "
                                 "per-target comparator is exact and stays with the ensemble's own name.",
            "per_target_comparator_is_rule_comparator": {t: v["model"] for t, v in blk["per_target_comparator"].items()
                                                         if v["model"] in RULE},
            "disclosure": "The pre-test reviewer found that no listed X-U comparator was 'C-X without BioSet' and "
                          "checked three unlisted alternatives on development (seed 0): mean(RF[community+context], "
                          "RF[index+geography]) 0.3160, the same plus TabPFN 0.3314, the same plus LightGBM 0.3375, "
                          "against C-X 0.3469 and the best listed comparator 0.3247. The rule was fixed by the lead "
                          "after that check. No choice among alternatives is made from development deltas: the rule "
                          "names one swap member by its single-model development score, and that member need not be "
                          "the alternative that looked strongest in the reviewer's check."}
        rec["dev_evidence"][b] = {k: blk[k] for k in ("sentinel_mean_r2_anomaly", "comparator_mean_r2_anomaly",
                                                       "paired_dev_vs_comparator", "n_pseudo_test")}
        rec.setdefault("sota_expectation", {"definition": (
            "SOTA campaign (docs/plans/sota_campaign.md): expected outcome of the confirmatory candidate C-X per target "
            "on DEVELOPMENT (seed 0, paired location-cluster bootstrap B=2,000 of r2_anomaly): 'SOTA-expected' if the "
            "paired CI excludes 0 above vs BOTH the strongest published method (WA-PLS2, kNN-MAT, bio.infer, GJAM "
            "where its fidelity check passed) AND the strongest simple/practitioner baseline (region mean, "
            "RF[geography], nearest reference, UK-index regressions) AND the per-target comparator; 'negative' if "
            "below either; else 'competitive'. Development only — the label itself comes only from the single test "
            "look (§8).")})
        rec["sota_expectation"][b] = blk.get("sota_expectation_cx", {})
        n_conf[b] = blk["n_sentinel_configurations"]
    rec["not_compared"] = not_compared()
    rec["reproducibility_tolerance"] = dict(F.TOLERANCE)
    rec["reproducibility_rule"] = (
        "Protocol v11 item 6: runs are not bit-reproducible (two development dry runs of one record differed by up to "
        "4.4e-5 in delta, 9.7e-5 in CI ends, 1.5e-4 in per-seed delta and 0.0016 in p). (a) The sealed predictions of "
        "the one look are the record and are never refitted. (b) A rescore from the same sealed predictions must "
        "reproduce every stored number within rescore_abs and every label exactly; otherwise both result sets are "
        "kept and no label is used until the lead has ruled. (c) A later refit from the snapshot (reproduction, never "
        "a second look) reproduces the result when delta and CI ends agree within delta / ci, per-seed delta within "
        "per_seed_delta and p within p. (d) Confirmatory rows within these tolerances of a label threshold are "
        "flagged. (e) A bootstrap p of 0 is reported as '< 2/B'.")
    faults = F.check_listed(rec)
    if faults:
        raise ValueError("the record does not follow the protocol's comparator rules: " + "; ".join(faults))
    n_listed = ", ".join(f"{b}: {len(rec['comparator_set'][b])}" for b in XD.BLOCKS)
    rec["listed_comparator_rule"] = (
        f"Amendments v9, v10 and v11 item 1 (written before any test data were seen): every comparator in "
        f"comparator_set ({n_listed}) "
        "is fitted at the look in its development-chosen variant (seeds 0-2 where stochastic) and compared with C-X; "
        "results/listed_comparators.json holds every paired estimate. The scorer refuses a record whose list differs "
        "from the protocol's or whose rule comparators do not follow from its own C-X. The confirmatory family stays "
        "12 tests (C-X vs the per-target comparator, re-chosen among the enlarged list by the unchanged rule).")
    rec["event_track"] = event_track(sel)
    rec["n_configurations_tried"] = {**n_conf, "total": sum(n_conf.values()), "cap": 12,
                                     "rule": "counted per (configuration, block); amendment v4.9"}
    rec["selection_note"] = ("Selected by the pre-registered rule (highest mean development r2_anomaly over the block's 6 "
                             "targets, seed 0) among the SENTINEL configurations of amendment v4.9; development = "
                             "NLA 2017 LOEO (X-L) and UK 2016-20 leave-one-area-out (X-U) pseudo-tests on training-pool "
                             "rows only. Comparators: strongest of all §6 comparators per (block, target) by development "
                             "r2_anomaly, each in its chosen variant. Amendments v9/v10 change the scorer and the "
                             "label rule only: no model was fitted or selected on development after them. Amendment "
                             "v11 adds the two rule comparators (means of existing development predictions) and "
                             "restructures the look; C-X, the secondaries, every model, feature set, seed, split and "
                             "fold are unchanged and no model was fitted on development for it.")
    pinned = sorted(set(CODE) | set(XL.scoring_code_files()))
    files = {rel: file_sha256(REPO / rel) for rel in pinned}
    files.update({str(f): file_sha256(f) for f in RUN_FILES})
    rec["sha256"] = files
    rec["data_sha256"] = {str(paths.DATA_ROOT / f): file_sha256(paths.DATA_ROOT / f) for f in DATA}
    rec["tabpfn_checkpoint"] = {"path": str(TABPFN_CKPT), "sha256": file_sha256(TABPFN_CKPT)}
    rec["sha256"][str(TABPFN_CKPT)] = rec["tabpfn_checkpoint"]["sha256"]
    rec["package_versions"], rec["r_package_versions"] = F.package_versions(PACKAGES), F.r_package_versions()
    rec["scoring_code_sha256"] = XL.scoring_code_sha256()
    rec["scoring_code_files"] = XL.scoring_code_files()     # static import closure (incl. events.py -> hazards, ...)
    rec["code_snapshot"] = XL.make_snapshot(pinned)        # v11 item 3: the look runs from this read-only copy
    arts = [*RUN_FILES, TABPFN_CKPT, XD.DEV_DIR / "design_v1.pkl", XD.DEV_DIR / "frozen_keys.json",
            Path(rec["not_compared"]["gjam"]["first_check_v4_8"]["file"]),
            *(Path(f) for b in XD.BLOCKS for f in rec["not_compared"]["gjam"]["fidelity"][b]["files"])]
    if rec["event_track"].get("status") == "active":
        arts += [Path(rec["event_track"]["calibration_file"]), Path(rec["event_track"]["dev_results_file"])]
    rec["artifacts"] = {str(f): file_sha256(f) for f in arts}   # all verified before the claim
    rec["data_snapshots"] = XL.make_data_pins()          # content-hash pins (manifest sha + exact loader files)
    rec["freeze_procedure"] = ("After the independent pre-test review, the LEAD writes "
                               "docs/preregistration/bioassess_x_freeze.json = this record with status 'frozen', "
                               "frozen_at and candidate_sha256 = sha256 of this file. The scorer accepts only that "
                               "file (xtestlook.load_freeze) and never this candidate. The look is started from the "
                               "code snapshot in code_snapshot.dir (python -B <dir>/scripts/eval/bioassess_x_final.py); "
                               "every command verifies the snapshot's exact file set and hashes and refuses in real "
                               "mode from any other tree. The freeze file, this candidate and the registry stay in "
                               "the live repository (v11 item 3).")
    rec["resume_rule"] = ("Protocol v11 item 2 (replaces the single-resume rule of §9, v5 and v7b). Four commands: "
                          "claim, chunk, seal, score. Prediction phase: label-free; one write-once unit per (block, "
                          "leaf model, fold, seed) and per event fit, each with its own file and sha256; chunks may "
                          "run in any order, in parallel, and be started again any number of times before sealing; "
                          "each chunk re-verifies the freeze record, the code snapshot, the data pins and the "
                          "artifacts; a failed or killed chunk leaves finished units in place and does not end the "
                          "look. The seal is written once, when every planned unit exists and verifies. Scoring is "
                          "single-shot and is not resumed: if it stops before the one locked registry write, "
                          "--rescore-from-saved recomputes everything from the sealed predictions (recorded; "
                          "agreement within rescore_abs required where the stopped run left results); once the "
                          "scoring rows exist, no scoring run can write again.")
    ld = XL.look_dir()
    rec["one_time_guards"] = {"look_dir": str(ld),
                              "markers": {"claim": str(ld / "TEST_LOOK.json"),
                                          "sealed": str(ld / "PREDICTIONS_SAVED.json"),
                                          "scoring_started": str(ld / "SCORING_STARTED.json"),
                                          "scored": str(ld / "SCORED.json")},
                              "registry_experiment": XL.EXPERIMENT,
                              "registry_metrics": {"claim": "look_claimed", "sealed": "predictions_sealed"},
                              "result_experiment": XL.EXPERIMENT,
                              "outputs": "sealed mode 0444 + SHA256SUMS"}
    rec["test_rows_note"] = ("test rows are loaded with the design rules of xdata (NLA: first visit with benthic taxa; "
                             "UK: >= 3 harmonised taxa); the numbers loaded vs in the split table are recorded in "
                             "pass_info.json. Correction (v11 item 4a): 459 of the 974 X-L test rows lie at Fuse/D2n "
                             "test sites (439 Fuse tier-C, 20 more D2n-only), not 549 as §2 said; counts from sample "
                             "keys only.")
    if REHEARSAL_FILE.is_file():            # v11 item 7: summary of the development rehearsal (numbers not usable)
        rec["rehearsal"] = {"file": str(REHEARSAL_FILE), "sha256": file_sha256(REHEARSAL_FILE),
                            "summary": json.loads(REHEARSAL_FILE.read_text())}
    else:
        rec["rehearsal"] = {"status": "not run yet (protocol v11 item 7): the record is regenerated after it"}
    return rec


if __name__ == "__main__":
    r = build()
    OUT.write_text(json.dumps(r, indent=1, default=float))
    print(f"wrote {OUT}: {len(r['sha256'])} files, {len(r['data_sha256'])} data files; "
          f"configurations {r['n_configurations_tried']}")
