#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the frozen unseen-taxa test blocks (ChemTox follow-on track, RESEARCH_PLAN §15.5 5-7).

Fits nothing and prints no target value: it writes the three frozen test blocks (insects,
molluscs, amphibians — whole taxonomic classes ADORE never contains), the training pool those
classes are removed from, the pre-registered power statistics and a manifest. The protocol is
``docs/preregistration/chemtox_unseen_taxa_plan.md``.

    python -B scripts/data/chemtox_build_unseen_taxa.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.core.registry import file_sha256
from sentinel.models.chemtox import prospective as PRO
from sentinel.models.chemtox import unseen_taxa as U

log = logging.getLogger("chemtox_unseen_taxa")
COLUMNS = ["result_id", "test_id", "reference_number", "publication_year", "test_cas",
           "chem_inchikey", "chemical_name", "chem_smiles", "chem_structure_class",
           "effect_norm", "endpoint_norm", "duration_h", "media_type_norm", "conc1_type_norm",
           "censor_kind", "log10_conc_mol_L", "taxon_group", "class", "tax_order", "family",
           "genus", "species", "latin_name", "species_number", "exposure_type_norm",
           "test_location_norm", "organism_lifestage_norm"]
OUT = "chemtox_unseen_taxa"


def add_chemistry(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ADORE's RDKit representations (canonical SMILES, MACCS, Morgan, MW, logP)."""
    chem = PRO.rdkit_chemistry(df["chem_smiles"])
    chem = chem[chem["parse_ok"].fillna(False)] if "parse_ok" in chem.columns else chem
    keep = ["_smiles_in", "chem_rdkit_can_smiles", "chem_MACCS_fp", "chem_Morgan_fp", "chem_mw",
            "chem_rdkit_clogp", "n_carbon_rdkit", "n_fragments"]
    out = df.merge(chem[keep], left_on="chem_smiles", right_on="_smiles_in", how="inner")
    return out.drop(columns=["_smiles_in"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    src = paths.DATA_ROOT / "processed" / "ecotox" / "ecotox_aquatic.parquet"
    eco = pd.read_parquet(src, columns=COLUMNS)
    log.info("ECOTOX aquatic rows %d", len(eco))
    curated = U.curate(eco)
    log.info("after ADORE's acute-lethality rules: %d rows", len(curated))

    pool = U.training_pool(curated)
    log.info("training pool (ADORE taxa, held-out classes removed): %d rows, %d chemicals",
             len(pool), pool["test_cas"].nunique())
    # the structural key needs ADORE's canonical SMILES convention
    pool = add_chemistry(pool)
    log.info("training pool with RDKit chemistry: %d rows", len(pool))

    blocks, report = [], {"source": str(src), "source_sha256": file_sha256(src),
                          "curated_rows": int(len(curated)),
                          "training_pool": {"rows": int(len(pool)),
                                            "chemicals": int(pool["test_cas"].nunique()),
                                            "species": int(pool["latin_name"].nunique()),
                                            "groups": sorted(pool["taxon_group"].unique())},
                          "held_out_classes": list(U.HELD_OUT_CLASSES), "blocks": {}}
    curated_chem = add_chemistry(curated)
    for b in U.BLOCKS:
        sub = U.test_block(curated_chem, b, pool)
        U.assert_no_test_rows(pool, sub, what="training pool")
        report["blocks"][b] = U.power_summary(sub)
        report["blocks"][b]["classes"] = sorted(sub["class"].astype(str).unique())
        blocks.append(sub)
        log.info("%s: %d rows (%d primary, %d chemicals, %d species)", b, len(sub),
                 report["blocks"][b]["rows_primary"], report["blocks"][b]["chemicals_primary"],
                 report["blocks"][b]["species_primary"])
    test = pd.concat(blocks, ignore_index=True)
    U.assert_no_test_classes(pool, what="training pool (final)")
    U.assert_no_test_rows(pool, test, what="training pool (final)")

    out_dir = paths.ensure(paths.DATA_ROOT / "processed" / OUT)
    tp, pp = out_dir / "test_blocks.parquet", out_dir / "training_pool.parquet"
    test.to_parquet(tp, index=False)
    pool.to_parquet(pp, index=False)
    report["test_blocks_sha256"] = file_sha256(tp)
    report["training_pool_sha256"] = file_sha256(pp)
    report["test_rows_total"] = int(len(test))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    write_manifest(OUT, files=[tp, pp], sources=[str(src)],
                   description=("ChemTox unseen-taxa test blocks (insects, molluscs, amphibians: "
                                "whole taxonomic classes absent from ADORE) curated from ECOTOX "
                                "with ADORE's acute-lethality rules, plus the training pool those "
                                "classes are removed from"),
                   license="US Government work (EPA ECOTOX)", synthetic=False,
                   stats={"test_rows": int(len(test)), "pool_rows": int(len(pool))})
    print(json.dumps({"test_blocks_sha256": report["test_blocks_sha256"],
                      "training_pool_sha256": report["training_pool_sha256"],
                      "test_rows": int(len(test)), "pool_rows": int(len(pool))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
