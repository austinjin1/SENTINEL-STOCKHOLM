#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Trivial baselines + replicate noise floor on ADORE, logged to the results registry.

Reproduces docs/research/04 §A.2: global-mean, per-species-mean and species+duration-mean
predictors of log10 LC50/EC50 (mol/L), fit on the official training rows of
``split_occurrence`` / ``split_scaffold-murcko`` (scaffold columns exist only in the t-* files),
scored on the official test rows ("<challenge>/<split>/test") and by 5-fold CV over the official
folds 0-4 ("<challenge>/<split>/cv": mean of the five fold scores; per-fold and pooled
out-of-fold scores in ``extra``). The replicate repeatability floor is logged per challenge as
``replicate_pooled_sd`` ("<challenge>/all"). All entries are ``is_baseline=True`` — they are
reference points, not SENTINEL results.

Re-running is idempotent: an entry identical to an existing one is skipped; a changed value is
logged with ``supersedes`` pointing at the entry it replaces.

    python -B scripts/data/build_adore_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths  # noqa: E402
from sentinel.core.registry import log_result, registry_frame  # noqa: E402
from sentinel.data import chemtox_baselines as B  # noqa: E402
from sentinel.data.sources import adore as A  # noqa: E402

log = logging.getLogger("adore_baselines")
EXPERIMENT = "adore_trivial_baselines"
WORKSTREAM = "chemtox"
TARGET_NAME = "log10_LC50_EC50_mol_per_L"
CHALLENGES = ("t-F2F", "t-C2C", "t-A2A", "a-CA2F-same", "a-CA2F-diff")
SPLITS = (A.PRIMARY_SPLIT, A.SECONDARY_SPLIT)
MANIFESTS = ["adore_raw", "adore"]


class Logger:
    def __init__(self):
        df = registry_frame()
        self.existing = df[df["experiment"] == EXPERIMENT] if len(df) else df
        self.ids: list[dict] = []
        self.skipped = 0

    def log(self, **kw):
        prev = None
        if len(self.existing):
            m = (
                (self.existing["model"] == kw["model"])
                & (self.existing["metric"] == kw["metric"])
                & (self.existing["split"] == kw["split"])
                & (self.existing["target"] == kw["target"])
            )
            hits = self.existing[m].sort_values("timestamp")
            if len(hits):
                prev = hits.iloc[-1]
                if abs(float(prev["value"]) - float(kw["value"])) <= 1e-12 and int(prev["n"] or -1) == int(
                    kw["n"]
                ):
                    self.skipped += 1
                    self.ids.append(
                        {k: kw[k] for k in ("model", "metric", "split")}
                        | {"id": prev["id"], "value": float(prev["value"]), "n": kw["n"], "new": False}
                    )
                    return
        e = log_result(
            workstream=WORKSTREAM,
            experiment=EXPERIMENT,
            is_baseline=True,
            data_manifests=MANIFESTS,
            supersedes=None if prev is None else prev["id"],
            **kw,
        )
        self.ids.append(
            {k: kw[k] for k in ("model", "metric", "split")}
            | {"id": e["id"], "value": e["value"], "n": kw["n"], "new": True}
        )


def main() -> int:
    out = paths.processed("adore")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "baselines.log"), logging.StreamHandler()],
    )
    L = Logger()
    full = {}
    art = out / "trivial_baselines.json"
    for ch in CHALLENGES:
        df = A.load_challenge(ch)
        full[ch] = {}
        for sp in SPLITS:
            if sp not in df.columns:
                continue
            res = B.evaluate_split(df, sp, A.TARGET)
            full[ch][sp] = res
            short = sp.replace("split_", "")
            cfg = {
                "challenge": ch,
                "split_column": sp,
                "target_column": A.TARGET,
                "species_column": B.SPECIES_COL,
                "duration_column": B.DURATION_COL,
                "fallback": "species+duration -> species -> global mean (training rows only)",
            }
            for model in B.BASELINES:
                for metric in ("rmse", "mae", "r2"):
                    L.log(
                        model=model,
                        metric=metric,
                        value=res["test"][model][metric],
                        split=f"{ch}/{short}/test",
                        n=res["n_test"],
                        target=TARGET_NAME,
                        config=cfg,
                        artifacts=[art],
                        notes="trivial baseline (research report 04 §A.2), official test rows",
                        extra={
                            "n_train": res["n_train"],
                            "test_sd": res["test_sd"],
                            "fallbacks_test": res["fallbacks_test"],
                        },
                    )
                    cv = res["cv"][model]
                    L.log(
                        model=model,
                        metric=metric,
                        value=cv[metric],
                        split=f"{ch}/{short}/cv",
                        n=res["cv_n"],
                        target=TARGET_NAME,
                        config=cfg,
                        artifacts=[art],
                        notes="mean of 5 official CV folds (fit on 4 folds, scored on the 5th)",
                        extra={
                            "per_fold": cv[metric + "_per_fold"],
                            "fold_sd": cv[metric + "_fold_sd"],
                            "pooled_oof": cv["pooled_" + metric],
                            "folds": res["cv_folds"],
                            "n_val_per_fold": res["cv_n_val_per_fold"],
                        },
                    )
            t = res["test"]
            log.info(
                "%-12s %-22s ntr=%5d nte=%5d SD=%.3f | global %.3f (%.3f) | species %.3f (%.3f) | "
                "species+dur %.3f (%.3f) | CV global %.3f",
                ch,
                sp,
                res["n_train"],
                res["n_test"],
                res["test_sd"],
                t["global_mean"]["rmse"],
                t["global_mean"]["r2"],
                t["species_mean"]["rmse"],
                t["species_mean"]["r2"],
                t["species_duration_mean"]["rmse"],
                t["species_duration_mean"]["r2"],
                res["cv"]["global_mean"]["rmse"],
            )
        nf = B.replicate_noise_floor(df, A.TARGET)
        wc = B.within_chemical_sd(df, A.TARGET)
        full[ch]["noise_floor"] = nf
        full[ch]["within_chemical_sd"] = wc
        L.log(
            model="replicate_noise_floor",
            metric="replicate_pooled_sd",
            value=nf["pooled_sd"],
            split=f"{ch}/all",
            n=nf["dof"],
            target=TARGET_NAME,
            config={"challenge": ch, "replicate_keys": nf["keys"], "pooling": "ddof=1 within key"},
            artifacts=[art],
            notes="label repeatability floor; n = pooled degrees of freedom",
            extra={k: v for k, v in nf.items() if k != "keys"} | {"within_chemical_sd": wc},
        )
        log.info(
            "%-12s noise floor pooled SD %.3f (dof %d, keys %d, with replicates %d, mean/median per-key "
            "SD %.3f/%.3f); within-chemical SD median %.3f mean %.3f (n=%d)",
            ch,
            nf["pooled_sd"],
            nf["dof"],
            nf["n_keys"],
            nf["n_keys_with_replicates"],
            nf["mean_per_key_sd"],
            nf["median_per_key_sd"],
            wc["median_sd"],
            wc["mean_sd"],
            wc["n_chemicals"],
        )
    art.write_text(json.dumps(full, indent=2, default=float))
    (out / "trivial_baselines_registry_ids.json").write_text(json.dumps(L.ids, indent=2))
    log.info(
        "registry: %d new entries, %d identical entries skipped", sum(i["new"] for i in L.ids), L.skipped
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
