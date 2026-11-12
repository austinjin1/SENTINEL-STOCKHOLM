# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Pre-declared constants and configuration lists of the SENTINEL-Sonde architecture round v2 (amendment §2a/§2h;
lead rulings B, E, H, L). ONE place: the drivers read nothing else. :func:`declaration` is the record whose sha256
is logged to the registry before the first run of any arm (``sonde_v2_cv.py --phase declare``); a driver refuses a
configuration that is not in the list, a list whose hash is not the declared one, and a GBM arm while
``GBM_SETTINGS["ruled"]`` is False. The GBM settings are those of the lead's GBM ruling
(``docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md``, G-2, G-3, G-5, G-6, G-7)."""

from __future__ import annotations

import hashlib
import json

SEED = 20261110
HORIZONS = (1, 2, 3, 5, 7, 10, 14)
L = 14                                  # network leads 1..14
N_FOLDS = 5
ES_FRACTION = 0.15                      # inner early-stopping basin groups (of the in-fold groups; seed SEED + fold)
PRODUCT = "gefs_c00"                    # the one declared NWP product (lead ruling G(b))
TRAIN_END = "2021-09-30"

SCORED = ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "ph_median", "turb_fnu_mean",
          "turb_fnu_median", "no3no2_mgl_mean"]
PRIMARY = ["temp_c_mean", "do_mgl_mean", "do_mgl_min", "spcond_uscm_mean", "turb_fnu_mean"]
# §2d auxiliary report: SONDE-HF against persistence and climatology only (no claim against the references)
AUX_REPORTED = ["temp_c_max", "temp_c_min", "temp_c_7dadm", "do_mgl_max", "do_amp_mgl", "do_pctsat",
                "spcond_uscm_max", "turb_fnu_max", "discharge_cfs_mean", "gage_ft_mean"]
TRANSFORMED = {"do_pctsat": "deterministic transform of the model's DO-mean quantiles at its T-mean median"}

# what each arm writes (lead ruling H)
WRITES = {"sonde_hf": "SCORED + AUX_REPORTED (config C3: PRIMARY only)", "rnh": "SCORED, one model per target",
          "gbm_nwp": "SCORED", "pf_gbm": "SCORED", "chronos_cov": "SCORED"}

CONFIGS = {
    "sonde_hf": {"C1": {}, "C2": {"hourly": False}, "C3": {"multitask": False}, "C4": {"head": "cmal"},
                 "C5": {"hidden_d": 384, "hidden_f": 192}},
    "rnh": {"R1": {"head": "cmal", "hidden_f": 128}, "R2": {"head": "cmal", "hidden_f": 256},
            "R3": {"head": "quantile", "hidden_f": 128}, "R4": {"head": "quantile", "hidden_f": 256},
            "R5": {"head": "cmal", "hidden_f": 128, "lr": 3e-4}},
    "gbm_nwp": {"Q19": {"quantile_mode": "quantile", "quantile_levels": 19},
                "Q": {"quantile_mode": "quantile", "quantile_levels": 7},
                "LS": {"quantile_mode": "location_scale"}},
    "pf_gbm": {"Q19": {"quantile_mode": "quantile", "quantile_levels": 19, "physics": True},
               "Q": {"quantile_mode": "quantile", "quantile_levels": 7, "physics": True},
               "LS": {"quantile_mode": "location_scale", "physics": True}},
    "chronos_cov": {"ZS": {"mode": "multi", "context": 512, "future_clim": True}},   # zero-shot, no fitting
}
MANDATORY = {"gbm_nwp": "Q19", "pf_gbm": "Q19"}          # lead ruling B
SEEDS = {"cv": [0], "step1": [0]}

# stage 2 of both networks (identical for SONDE-HF and R-NH; lead ruling J): the stage-1 configuration with
STAGE2 = {"lr_divisor": 3, "max_epochs": 10, "patience": 3, "init": "stage-1 best weights, fresh optimiser",
          "product": "archived bias-corrected GEFS c00 of the train block (NaN where none exists)"}

STEP1 = {"variables": ["temp_c_mean", "do_mgl_mean"], "horizons": [3, 5, 7], "eval_from": "2015-10-01",
         "held_fold": 0, "min_rows_per_site": 30, "threshold": 0.01, "candidate": ["sonde_hf", "C1"],
         "comparator": ["gbm_nwp", "Q19"], "also_run": []}

# GBM settings shared by GBM-NWP and PF-GBM (lead rulings D, E; lead GBM ruling of 2026-11-10, cited as G-n):
# full rows for every configuration of both arms in step 1 and CV (G-2; ``max_fit_rows`` None, no driver applies a
# cap to fit rows), the recipe of the frozen ``GBMConfig`` written out (G-5), LightGBM on one GPU per process with
# the device settings of the benchmark record (G-6), booster-level resume (G-3; ``resume``), and the CV GPU-hour
# limit above which no CV GBM process starts without a new lead ruling (G-7d). ``cap_ladder`` is the ladder of the
# timing rule; it is read by the timing benchmark script only and by no driver (G-9).
GBM_SETTINGS = {
    "ruled": True,
    "ruling": {"file": "docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md",
               "sha256": "3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b"},
    "learning_rate": 0.05, "max_rounds": 2000, "early_stopping_rounds": 100, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "num_threads": 6,
    "device": "gpu", "device_settings": {"device_type": "gpu", "gpu_use_dp": False, "max_bin": 255},
    "es_max_rows": 400_000, "seed": 0,
    "max_fit_rows": {"Q19": None, "Q": None, "LS": None},
    "cap_ladder": [None, 1_500_000, 1_000_000, 500_000, 250_000],
    "min_fit_rows": 1000, "min_es_rows": 200,
    "resume": {"process_limit_min": 115, "first_booster_allowance_min": 40, "factor": 1.5, "booster_limit_h": 2,
               "abort_min": 119, "predict_latest_min": 105},
    "cv_gpu_hours_limit": 300,
}

ROW_RULE = ("train rows: issue days of the TRAIN block with target day (issue + h) <= 2021-09-30 by the frozen "
            "issue_dates rule (networks: issue + 14 for every lead), no fit_end mask, no extra embargo inside the "
            "train block; reference-window issue rule for every model; early-stopping rows: the same rule on the "
            "inner 15 % of in-fold basin groups")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def arm_list(arm: str) -> dict:
    """The declared configuration list of one arm with every setting that defines a run of it."""
    out = {"arm": arm, "configs": CONFIGS[arm], "writes": WRITES[arm], "seeds": SEEDS, "row_rule": ROW_RULE,
           "product": PRODUCT, "horizons": list(HORIZONS), "es_fraction": ES_FRACTION, "seed": SEED}
    if arm in ("gbm_nwp", "pf_gbm"):
        out["gbm_settings"] = GBM_SETTINGS
        out["mandatory"] = MANDATORY[arm]
    if arm in ("sonde_hf", "rnh"):
        from dataclasses import asdict

        from .train_v2 import ORDER, TrainConfig
        out["train_config_defaults"] = asdict(TrainConfig(kind=arm))
        out["stage2"] = STAGE2
        out["order_constraints"] = [list(o) for o in ORDER] if arm == "sonde_hf" else []
    return out


def declaration() -> dict:
    arms = {a: arm_list(a) for a in CONFIGS}
    return {"round": "sonde_arch_v2", "seed": SEED, "scored": SCORED, "primary": PRIMARY,
            "aux_reported": AUX_REPORTED, "transformed": TRANSFORMED, "step1": STEP1, "n_folds": N_FOLDS,
            "arms": arms, "arm_sha256": {a: _sha(v) for a, v in arms.items()}}


def declaration_sha256() -> str:
    return _sha(declaration())


def train_overrides(arm: str, cname: str) -> dict:
    if cname not in CONFIGS.get(arm, {}):
        raise KeyError(f"configuration {cname!r} is not declared for arm {arm!r}")
    return dict(CONFIGS[arm][cname])


def gbm_config(arm: str, cname: str, *, seed: int | None = None, horizons_note: str = ""):
    """The ``GBMNWPConfig`` of a declared GBM configuration: the ruled recipe and device, full fit rows (G-2).
    Refuses while the settings are not ruled and refuses any declared cap on fit rows."""
    from .gbm_nwp import GBMNWPConfig

    if arm not in ("gbm_nwp", "pf_gbm"):
        raise KeyError(arm)
    g = GBM_SETTINGS
    if not g["ruled"]:
        raise RuntimeError("GBM settings are not ruled yet (timing benchmark -> lead ruling -> declared.py); "
                           "no GBM arm may run")
    over = train_overrides(arm, cname)
    if g["max_fit_rows"][cname] is not None:
        raise RuntimeError(f"GBM ruling G-2: every GBM configuration is fitted on full rows; a cap on the fit rows "
                           f"of {cname} ({g['max_fit_rows'][cname]}) needs a new lead ruling")
    if g["device"] == "gpu":
        if g["device_settings"] != {"device_type": "gpu", "gpu_use_dp": False, "max_bin": g["max_bin"]}:
            raise RuntimeError("GBM ruling G-6: a GPU fit uses device_type gpu, gpu_use_dp False and the declared "
                               f"max_bin; the declared device settings are {g['device_settings']}")
    elif g["device"] != "cpu":                 # "cpu" only in unit tests; every declared run is "gpu" (G-6)
        raise RuntimeError(f"GBM ruling G-6: device {g['device']!r} is not ruled")
    return GBMNWPConfig(**over, learning_rate=g["learning_rate"], max_rounds=g["max_rounds"],
                        early_stopping_rounds=g["early_stopping_rounds"], num_leaves=g["num_leaves"],
                        min_data_in_leaf=g["min_data_in_leaf"], feature_fraction=g["feature_fraction"],
                        bagging_fraction=g["bagging_fraction"], bagging_freq=g["bagging_freq"],
                        lambda_l2=g["lambda_l2"], num_threads=g["num_threads"], max_bin=g["max_bin"],
                        device=g["device"], es_max_rows=g["es_max_rows"], max_fit_rows=None,
                        seed=g["seed"] if seed is None else seed)
