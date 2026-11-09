#!/usr/bin/env python3
"""Build a VALID aquatic-toxicogenomics dataset from real GEO microarray studies.

This replaces the earlier (invalid) ToxiGene pipeline that used human/cattle
disease GEO series with hard-coded labels and a random "Reactome" hierarchy.

Design (all real, no fabrication):
  * Studies: zebrafish (Danio rerio) toxicology microarray series, all on the
    SAME platform GPL14664 (Agilent zebrafish microarray) so probe IDs and gene
    symbols are directly comparable across studies.
  * Expression: parsed from each GEO series_matrix.txt.gz (probe x sample table).
  * Probe -> gene symbol: mapped via the real GPL14664 platform annotation
    (GENE_SYMBOL column); control probes dropped; multiple probes per gene
    collapsed by mean within a study.
  * Gene intersection: column j = the SAME gene symbol across all studies.
  * Labels: derived from REAL GEO sample metadata. A sample is class 0
    (control) if its title/source_name/characteristics indicate a control /
    solvent / vehicle / untreated condition, else class 1 (toxicant-exposed).
    The task is binary: "was this animal exposed to an aquatic toxicant?"
  * Evaluation: leave-one-study-out cross validation (see evaluate script).

Output: data/processed/molecular/realgeo/
  X.npy            (N_samples, N_genes) float32, per-study standardized log2
  y.npy            (N_samples,) int  (0=control, 1=exposed)
  study.npy        (N_samples,) str  GSE id per sample
  genes.json       list of N_genes gene symbols (the shared feature axis)
  samples.json     per-sample provenance + raw metadata + assigned label + why

MIT License -- Bryan Cheng, 2026
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GEO_DIR = PROJECT_ROOT / "data" / "raw" / "molecular" / "geo_aquatic"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "molecular" / "realgeo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROBE2GENE_PATH = GEO_DIR / "GPL14664_probe2gene.json"

# GPL14664 zebrafish toxicology studies with control-vs-exposure designs.
STUDIES = [
    "GSE89653",   # Ag nanoparticles / ionic silver, liver
    "GSE93227",   # dexamethasone / prednisolone (glucocorticoids), larvae
    "GSE90457",   # dietary silver from brine shrimp, intestine/liver
    "GSE110543",  # PCB-126 vs vehicle, liver, 3 strains
    "GSE116641",  # dydrogesterone (DDG) chronic, testes/brain
    "GSE121338",  # thyroid disruptors PTU/TBBPA, eye
    "GSE89756",   # cadmium, liver
    "GSE84394",   # imidazole/chronic env contaminant, liver
    "GSE129812",  # neuroactive compounds, larval brain
    "GSE93532",   # wastewater effluent, larvae
    "GSE112016",  # metal oxide nanoparticles (Cu/Ni/Zn ONP), gill
    "GSE110340",  # microcystin-LR (MCLR), testes
]

# Tokens (lowercased) that mark a sample as a CONTROL (class 0).
CONTROL_TOKENS = [
    "control", "solvent control", "vehicle", "untreated", "0.05%dmso",
    "dmso", "0% maize", "h2o", "blank", " ck", "ck_", "ck ", "naive",
]
# Tokens that should NOT be treated as control even if "control" substring
# appears spuriously (none needed currently, kept for clarity).
NON_CONTROL_OVERRIDE: list[str] = []


def parse_series_matrix(path: Path):
    """Parse a GEO series_matrix.txt.gz.

    Returns:
        probe_ids: list[str]
        sample_ids: list[str] (GSM ids)
        expr: (n_probes, n_samples) float64 with NaN for missing
        meta: dict gsm -> {'title','source','chars':[...]}
    """
    sample_ids: list[str] = []
    titles: list[str] = []
    sources: list[str] = []
    chars: list[list[str]] = []  # one list per characteristics line, len=n_samples

    probe_ids: list[str] = []
    rows: list[list[float]] = []

    in_table = False
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = _split_quoted(line)
            elif line.startswith("!Sample_title"):
                titles = _split_quoted(line)
            elif line.startswith("!Sample_source_name_ch1"):
                sources = _split_quoted(line)
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append(_split_quoted(line))
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                parts = line.rstrip("\n").split("\t")
                if parts[0].strip('"') == "ID_REF":
                    # header row inside table -> column order = sample_ids
                    table_cols = [p.strip('"') for p in parts[1:]]
                    continue
                pid = parts[0].strip('"')
                if not pid:
                    continue
                vals = []
                for v in parts[1:]:
                    v = v.strip().strip('"')
                    if v in ("", "null", "NA", "NaN", "None"):
                        vals.append(np.nan)
                    else:
                        try:
                            vals.append(float(v))
                        except ValueError:
                            vals.append(np.nan)
                probe_ids.append(pid)
                rows.append(vals)

    expr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 0))

    # Build per-sample metadata. table_cols should equal sample_ids.
    meta = {}
    n = len(sample_ids)
    for i in range(n):
        gsm = sample_ids[i]
        meta[gsm] = {
            "title": titles[i] if i < len(titles) else "",
            "source": sources[i] if i < len(sources) else "",
            "chars": [c[i] for c in chars if i < len(c)],
        }
    return probe_ids, sample_ids, expr, meta


def _split_quoted(line: str) -> list[str]:
    parts = line.rstrip("\n").split("\t")[1:]
    return [p.strip().strip('"') for p in parts]


def assign_label(meta_entry: dict) -> tuple[int, str]:
    """Assign binary control(0)/exposed(1) from real metadata.

    Returns (label, reason_string).
    """
    text = " | ".join(
        [meta_entry.get("title", ""), meta_entry.get("source", "")]
        + meta_entry.get("chars", [])
    ).lower()

    # Look at exposure/treatment-type characteristic fields first if present.
    for token in CONTROL_TOKENS:
        if token in text:
            return 0, f"control-token '{token.strip()}' in metadata"
    return 1, "no control token -> treated/exposed"


def collapse_to_genes(probe_ids, expr, probe2gene):
    """Map probes -> gene symbols, mean-collapse multi-probe genes.

    Returns dict gene_symbol -> (n_samples,) vector and the sample axis kept
    as-is (expr is probes x samples).
    """
    gene_rows: dict[str, list[np.ndarray]] = {}
    for i, pid in enumerate(probe_ids):
        g = probe2gene.get(pid)
        if g is None:
            continue
        gene_rows.setdefault(g, []).append(expr[i])
    gene_vec = {}
    for g, lst in gene_rows.items():
        arr = np.vstack(lst)
        gene_vec[g] = np.nanmean(arr, axis=0)
    return gene_vec


def main():
    probe2gene = json.loads(PROBE2GENE_PATH.read_text())
    print(f"Loaded probe->gene map: {len(probe2gene)} probes, "
          f"{len(set(probe2gene.values()))} genes")

    per_study_gene_vecs = {}   # gse -> {gene: (n_samp,)}
    per_study_samples = {}     # gse -> list[gsm]
    per_study_labels = {}      # gse -> list[int]
    per_study_reasons = {}     # gse -> list[str]
    per_study_meta = {}        # gse -> {gsm: meta}

    for gse in STUDIES:
        path = GEO_DIR / f"{gse}_series_matrix.txt.gz"
        if not path.exists():
            print(f"  SKIP {gse}: file missing")
            continue
        probe_ids, sample_ids, expr, meta = parse_series_matrix(path)
        if expr.size == 0 or len(sample_ids) == 0:
            print(f"  SKIP {gse}: no expression table parsed "
                  f"(probes={len(probe_ids)}, samples={len(sample_ids)})")
            continue

        # log2-transform if data look like linear intensities (max large, min>=0)
        finite = expr[np.isfinite(expr)]
        if finite.size and np.nanmax(finite) > 100 and np.nanmin(finite) >= 0:
            expr = np.log2(np.clip(expr, 0, None) + 1.0)

        gene_vec = collapse_to_genes(probe_ids, expr, probe2gene)
        labels, reasons = [], []
        for gsm in sample_ids:
            lab, why = assign_label(meta[gsm])
            labels.append(lab)
            reasons.append(why)

        per_study_gene_vecs[gse] = gene_vec
        per_study_samples[gse] = sample_ids
        per_study_labels[gse] = labels
        per_study_reasons[gse] = reasons
        per_study_meta[gse] = meta

        n_ctrl = labels.count(0)
        n_exp = labels.count(1)
        print(f"  {gse}: {len(sample_ids)} samples ({n_ctrl} ctrl / {n_exp} exp), "
              f"{len(gene_vec)} mapped genes")

    if not per_study_gene_vecs:
        raise SystemExit("No studies parsed.")

    # Gene intersection across studies (shared feature axis).
    common = None
    for gse, gv in per_study_gene_vecs.items():
        gset = set(gv.keys())
        common = gset if common is None else (common & gset)
    common_genes = sorted(common)
    print(f"\nShared genes across {len(per_study_gene_vecs)} studies: "
          f"{len(common_genes)}")

    # Build matrices. Per-study standardization (z-score across that study's
    # samples) so cross-study batch effects don't leak study identity as scale.
    X_parts, y_parts, study_parts, sample_records = [], [], [], []
    for gse in per_study_gene_vecs:
        gv = per_study_gene_vecs[gse]
        sample_ids = per_study_samples[gse]
        labels = per_study_labels[gse]
        reasons = per_study_reasons[gse]
        n = len(sample_ids)
        mat = np.zeros((n, len(common_genes)), dtype=np.float64)
        for j, g in enumerate(common_genes):
            mat[:, j] = gv[g]
        # impute remaining NaN with per-gene (within-study) mean
        col_mean = np.nanmean(mat, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        nanmask = np.isnan(mat)
        mat[nanmask] = np.take(col_mean, np.where(nanmask)[1])
        # per-study z-score per gene
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0)
        sd[sd < 1e-8] = 1.0
        mat = (mat - mu) / sd

        X_parts.append(mat.astype(np.float32))
        y_parts.extend(labels)
        study_parts.extend([gse] * n)
        for i, gsm in enumerate(sample_ids):
            sample_records.append({
                "gse": gse,
                "gsm": gsm,
                "label": int(labels[i]),
                "label_reason": reasons[i],
                "title": per_study_meta[gse][gsm]["title"],
                "source": per_study_meta[gse][gsm]["source"],
                "characteristics": per_study_meta[gse][gsm]["chars"],
            })

    X = np.concatenate(X_parts, axis=0)
    y = np.array(y_parts, dtype=np.int64)
    study = np.array(study_parts, dtype=object)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "study.npy", study)
    json.dump(common_genes, open(OUT_DIR / "genes.json", "w"))
    json.dump(sample_records, open(OUT_DIR / "samples.json", "w"), indent=2)

    print(f"\nFinal dataset: X={X.shape}, "
          f"{int((y==0).sum())} control / {int((y==1).sum())} exposed, "
          f"{len(set(study_parts))} studies")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
