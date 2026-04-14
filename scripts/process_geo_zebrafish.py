#!/usr/bin/env python3
"""Process real zebrafish toxicology GEO datasets for ToxiGene v2.

Reads downloaded SOFT.gz files from data/raw/molecular/geo_zebrafish/,
extracts per-sample expression values, maps probe IDs to gene identifiers,
and builds a unified expression matrix with consistent gene labels.

Each dataset is a microarray study with log-ratio or log-intensity values.
We:
  1. Parse platform probe → gene-name mapping from each SOFT file
  2. Extract per-sample VALUES from sample tables
  3. Aggregate multiple probes per gene (median)
  4. Build union gene set across all datasets
  5. Assemble final matrix: (n_samples, n_genes), missing → 0
  6. Apply log1p transform (shift negative values first) + per-sample z-score
  7. Assign toxicological outcome labels from sample metadata

Output:
  data/processed/molecular/geo_zebrafish_expression.npy   — (N, G) float32
  data/processed/molecular/geo_zebrafish_gene_names.json  — list of gene names
  data/processed/molecular/geo_zebrafish_outcomes.npy     — (N, 7) binary float32
  data/processed/molecular/geo_zebrafish_metadata.json    — per-sample metadata
  data/processed/molecular/geo_zebrafish_summary.json     — summary stats

MIT License — Bryan Cheng, 2026
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GEO_DIR = Path("/home/bcheng/SENTINEL/data/raw/molecular/geo_zebrafish")
OUT_DIR = Path("/home/bcheng/SENTINEL/data/processed/molecular")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Known outcome labels for each dataset based on chemical class
# These map known toxicants to SENTINEL's 7 adverse outcome categories:
# 0=reproductive_impairment, 1=growth_inhibition, 2=immunosuppression,
# 3=neurotoxicity, 4=hepatotoxicity, 5=oxidative_damage, 6=endocrine_disruption

CHEMICAL_OUTCOMES = {
    # diuron / dichlorophenyl compounds — herbicide, endocrine disruption + oxidative
    'diuron': [0, 1, 0, 0, 0, 1, 1],
    'dcf': [0, 1, 0, 0, 1, 1, 0],
    'diclofenac': [0, 1, 0, 0, 1, 1, 0],
    'naproxen': [0, 0, 0, 0, 1, 1, 0],
    'dca': [0, 1, 0, 0, 0, 1, 1],      # 3,4-dichloroaniline
    'dichloroaniline': [0, 1, 0, 0, 0, 1, 1],
    'chloroaniline': [0, 1, 0, 0, 0, 1, 1],
    'dichlorophenol': [0, 1, 0, 0, 1, 1, 1],
    'pentachlorophenol': [0, 1, 1, 0, 1, 1, 0],
    'pcp': [0, 1, 1, 0, 1, 1, 0],
    'nitrophenol': [0, 1, 0, 1, 1, 1, 0],
    # benzo-a-pyrene — PAH, carcinogen, hepatotoxic + oxidative
    'benzo': [0, 1, 0, 0, 1, 1, 0],
    'benzopyrene': [0, 1, 0, 0, 1, 1, 0],
    'pyrene': [0, 0, 0, 0, 1, 1, 0],
    'bap': [0, 1, 0, 0, 1, 1, 0],
    # estradiol / hormone — endocrine disruption + reproductive
    'estradiol': [1, 0, 0, 0, 0, 0, 1],
    'estrogen': [1, 0, 0, 0, 0, 0, 1],
    'e2': [1, 0, 0, 0, 0, 0, 1],
    # metals
    'cadmium': [1, 1, 1, 1, 1, 1, 0],
    'cd': [1, 1, 1, 1, 1, 1, 0],
    'arsenic': [0, 1, 1, 0, 1, 1, 0],
    'as': [0, 1, 0, 0, 1, 1, 0],
    'zinc': [0, 1, 0, 1, 1, 1, 0],
    'copper': [0, 1, 0, 1, 1, 1, 0],
    'mercury': [0, 1, 0, 1, 1, 1, 0],
    'lead': [0, 1, 0, 1, 1, 0, 0],
    # dichlorvos — organophosphate neurotoxin
    'dichlorvos': [0, 1, 0, 1, 1, 1, 0],
    'ddvp': [0, 1, 0, 1, 1, 1, 0],
    'organophosphate': [0, 1, 0, 1, 1, 1, 0],
    # air particles / PM2.5 — oxidative, immune
    'air': [0, 1, 1, 0, 1, 1, 0],
    'particle': [0, 1, 1, 0, 1, 1, 0],
    'pm': [0, 1, 1, 0, 1, 1, 0],
    # general fallback
    'control': [0, 0, 0, 0, 0, 0, 0],
    'vehicle': [0, 0, 0, 0, 0, 0, 0],
    'dmso': [0, 0, 0, 0, 0, 0, 0],
    'untreated': [0, 0, 0, 0, 0, 0, 0],
}

OUTCOME_NAMES = [
    'reproductive_impairment', 'growth_inhibition', 'immunosuppression',
    'neurotoxicity', 'hepatotoxicity', 'oxidative_damage', 'endocrine_disruption'
]


def infer_outcome(text: str) -> list[int]:
    """Infer 7-class outcome label from sample/chemical text description."""
    text_lower = text.lower()
    for keyword, outcome in CHEMICAL_OUTCOMES.items():
        if keyword in text_lower:
            return outcome
    # Default: moderate stress response (growth inhibition + oxidative damage)
    return [0, 1, 0, 0, 0, 1, 0]


def parse_soft_gz(filepath: Path) -> tuple[dict, list[dict]]:
    """Parse a GEO SOFT.gz file.

    Returns:
        platform_map: probe_id -> gene_name dict
        samples: list of {id, title, characteristics, values: {probe_id: float}}
    """
    platform_map: dict[str, str] = {}
    samples: list[dict] = []

    current_sample: dict | None = None
    in_platform_table = False
    in_sample_table = False
    platform_header: list[str] = []
    platform_gene_col = -1
    sample_table_header: list[str] = []
    value_col = 1  # default

    gse_title = ''
    gse_organism = ''

    try:
        with gzip.open(filepath, 'rt', errors='replace') as fh:
            for line in fh:
                line = line.rstrip('\n')

                # Series metadata
                if line.startswith('!Series_title'):
                    gse_title = line.split('=', 1)[-1].strip()
                elif line.startswith('!Series_organism'):
                    gse_organism = line.split('=', 1)[-1].strip()

                # Platform table
                elif '!platform_table_begin' in line.lower():
                    in_platform_table = True
                    platform_header = []
                    continue
                elif '!platform_table_end' in line.lower():
                    in_platform_table = False
                    continue
                elif in_platform_table:
                    if not platform_header:
                        platform_header = line.split('\t')
                        # Find best gene column: GENE_SYMBOL, ORF, Gene_Name, GENE_ID
                        for priority in ['GENE_SYMBOL', 'gene_symbol', 'ORF', 'Gene_Name',
                                         'gene_name', 'GENE_ID', 'gene_id', 'GenBank_Accession',
                                         'GB_ACC']:
                            if priority in platform_header:
                                platform_gene_col = platform_header.index(priority)
                                break
                        # Also check for Ensembl columns
                        for ens_col in ['ACCESSION_STRING', 'Sequence_ID', 'GB_LIST']:
                            if ens_col in platform_header:
                                # We'll check this col for ENSDARG/ENSDART
                                pass
                        continue
                    parts = line.split('\t')
                    if len(parts) < 2:
                        continue
                    probe_id = parts[0].strip()
                    gene_name = ''
                    # Try gene symbol column
                    if platform_gene_col >= 0 and platform_gene_col < len(parts):
                        gene_name = parts[platform_gene_col].strip()
                    # Fallback: look for ENSDARG/ENSDART in any column
                    if not gene_name or gene_name in ('---', '', 'NONE'):
                        for p in parts[1:]:
                            # Prefer ENSDARG (gene) over ENSDART (transcript)
                            m = re.search(r'ENSDARG\d+', p)
                            if m:
                                gene_name = m.group(0)
                                break
                        if not gene_name or gene_name in ('---', ''):
                            for p in parts[1:]:
                                m = re.search(r'ENSDART\d+', p)
                                if m:
                                    gene_name = m.group(0)
                                    break
                    if probe_id and gene_name and gene_name not in ('---', '', 'NONE'):
                        # Clean gene name: strip trailing .1 etc for ensembl
                        platform_map[probe_id] = gene_name

                # Sample headers
                elif line.startswith('^SAMPLE'):
                    if current_sample is not None and current_sample.get('values'):
                        samples.append(current_sample)
                    current_sample = {
                        'id': line.split('=', 1)[-1].strip(),
                        'title': '',
                        'characteristics': [],
                        'values': {},
                    }
                    in_sample_table = False
                elif current_sample is not None:
                    if line.startswith('!Sample_title'):
                        current_sample['title'] = line.split('=', 1)[-1].strip()
                    elif line.startswith('!Sample_characteristics_ch1') or line.startswith('!Sample_characteristics_ch2'):
                        current_sample['characteristics'].append(line.split('=', 1)[-1].strip())
                    elif line.startswith('!Sample_source_name'):
                        current_sample['characteristics'].append('source: ' + line.split('=', 1)[-1].strip())
                    elif '!sample_table_begin' in line.lower():
                        in_sample_table = True
                        sample_table_header = []
                        value_col = 1
                        continue
                    elif '!sample_table_end' in line.lower():
                        in_sample_table = False
                        continue
                    elif in_sample_table:
                        if not sample_table_header:
                            sample_table_header = line.split('\t')
                            # Find VALUE column
                            for col_name in ['VALUE', 'value', 'VALUE_CH1', 'CH1_MEAN',
                                             'Signal', 'Intensity', 'LOG_RATIO', 'RATIO']:
                                if col_name in sample_table_header:
                                    value_col = sample_table_header.index(col_name)
                                    break
                            continue
                        parts = line.split('\t')
                        if len(parts) > value_col:
                            probe_id = parts[0].strip()
                            try:
                                val = float(parts[value_col].strip())
                                if not np.isnan(val) and not np.isinf(val):
                                    current_sample['values'][probe_id] = val
                            except (ValueError, IndexError):
                                pass

        # Don't forget the last sample
        if current_sample is not None and current_sample.get('values'):
            samples.append(current_sample)

    except Exception as e:
        print(f"  ERROR parsing {filepath.name}: {e}")

    return platform_map, samples, gse_title, gse_organism


def build_sample_expression(sample: dict, platform_map: dict) -> dict[str, float]:
    """Map probe values to gene names, aggregating duplicates by median."""
    gene_vals: dict[str, list[float]] = defaultdict(list)
    for probe_id, val in sample['values'].items():
        gene = platform_map.get(probe_id, '')
        if gene:
            gene_vals[gene].append(val)
    return {g: float(np.median(vals)) for g, vals in gene_vals.items()}


def main():
    t0 = time.time()
    print("=" * 70)
    print("Processing Real Zebrafish GEO Datasets for ToxiGene v2")
    print("=" * 70)

    soft_files = sorted(GEO_DIR.glob("*_family.soft.gz"))
    print(f"\nFound {len(soft_files)} SOFT.gz files:")
    for f in soft_files:
        print(f"  {f.name} ({f.stat().st_size // 1024 // 1024:.1f} MB)")

    all_samples_meta = []
    all_gene_expr: list[dict[str, float]] = []
    gene_universe: set[str] = set()
    dataset_summaries = []

    for soft_path in soft_files:
        gse_id = soft_path.name.replace('_family.soft.gz', '')
        print(f"\n--- Parsing {gse_id} ---", flush=True)

        platform_map, samples, title, organism = parse_soft_gz(soft_path)
        print(f"  Platform probes mapped: {len(platform_map)}")
        print(f"  Samples extracted: {len(samples)}")
        print(f"  Title: {title[:80]}")
        print(f"  Organism: {organism}")

        if not samples:
            print(f"  SKIP: no samples extracted")
            continue

        dataset_sample_count = 0
        for samp in samples:
            gene_expr = build_sample_expression(samp, platform_map)
            if len(gene_expr) < 100:
                continue  # too few genes, skip
            gene_universe.update(gene_expr.keys())
            all_gene_expr.append(gene_expr)

            # Build metadata
            char_text = ' '.join(samp['characteristics'] + [samp['title']])
            outcome = infer_outcome(char_text)
            all_samples_meta.append({
                'gse': gse_id,
                'sample_id': samp['id'],
                'title': samp['title'],
                'characteristics': samp['characteristics'],
                'n_genes': len(gene_expr),
                'outcome': outcome,
            })
            dataset_sample_count += 1

        dataset_summaries.append({
            'gse': gse_id,
            'title': title,
            'organism': organism,
            'n_platform_probes': len(platform_map),
            'n_samples_extracted': dataset_sample_count,
        })
        print(f"  -> {dataset_sample_count} usable samples, {len(gene_universe)} genes in universe so far")

    print(f"\n{'='*70}")
    print(f"Total samples: {len(all_gene_expr)}")
    print(f"Total gene universe: {len(gene_universe)}")

    if len(all_gene_expr) == 0:
        print("ERROR: No samples extracted. Exiting.")
        sys.exit(1)

    # Sort gene names for reproducibility
    gene_names = sorted(gene_universe)
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_samples = len(all_gene_expr)
    n_genes = len(gene_names)

    print(f"\nBuilding expression matrix: ({n_samples}, {n_genes})...")
    expr_matrix = np.zeros((n_samples, n_genes), dtype=np.float32)

    for s_idx, gene_expr in enumerate(all_gene_expr):
        for gene, val in gene_expr.items():
            g_idx = gene_idx.get(gene, -1)
            if g_idx >= 0:
                expr_matrix[s_idx, g_idx] = val

    print(f"Matrix built. Non-zero entries: {np.count_nonzero(expr_matrix):,} "
          f"({100*np.count_nonzero(expr_matrix)/(n_samples*n_genes):.1f}% dense)")

    # Normalize: shift so min >= 0 per sample, then log1p, then z-score across samples
    print("Normalizing: shift→log1p→z-score...")
    # Per-sample shift to make minimum 0
    row_mins = expr_matrix.min(axis=1, keepdims=True)
    expr_matrix = expr_matrix - row_mins  # shift so min=0
    # Apply log1p
    expr_matrix = np.log1p(expr_matrix)
    # Per-gene z-score across samples
    gene_means = expr_matrix.mean(axis=0)
    gene_stds = expr_matrix.std(axis=0)
    gene_stds[gene_stds < 1e-6] = 1.0
    expr_matrix = (expr_matrix - gene_means) / gene_stds

    # Build outcome labels
    print("Building outcome labels...")
    outcomes = np.array([m['outcome'] for m in all_samples_meta], dtype=np.float32)

    # Summary statistics
    outcome_counts = outcomes.sum(axis=0)
    print(f"Outcome class balance:")
    for name, cnt in zip(OUTCOME_NAMES, outcome_counts):
        print(f"  {name}: {int(cnt)} positive ({100*cnt/n_samples:.1f}%)")

    # Save
    print(f"\nSaving outputs...")

    expr_out = OUT_DIR / 'geo_zebrafish_expression.npy'
    np.save(expr_out, expr_matrix)
    print(f"  expression: {expr_out} — {expr_matrix.shape}, {expr_out.stat().st_size//1024//1024:.1f} MB")

    gene_out = OUT_DIR / 'geo_zebrafish_gene_names.json'
    with open(gene_out, 'w') as f:
        json.dump(gene_names, f)
    print(f"  gene names: {gene_out} — {n_genes} genes")

    outcomes_out = OUT_DIR / 'geo_zebrafish_outcomes.npy'
    np.save(outcomes_out, outcomes)
    print(f"  outcomes: {outcomes_out} — {outcomes.shape}")

    meta_out = OUT_DIR / 'geo_zebrafish_metadata.json'
    with open(meta_out, 'w') as f:
        json.dump(all_samples_meta, f, indent=2)
    print(f"  metadata: {meta_out}")

    summary = {
        'n_samples': n_samples,
        'n_genes': n_genes,
        'datasets': dataset_summaries,
        'outcome_names': OUTCOME_NAMES,
        'outcome_class_counts': {name: int(cnt) for name, cnt in zip(OUTCOME_NAMES, outcome_counts)},
        'normalization': 'per-sample shift to min=0, log1p, per-gene z-score',
        'elapsed_s': time.time() - t0,
    }
    summary_out = OUT_DIR / 'geo_zebrafish_summary.json'
    with open(summary_out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  summary: {summary_out}")

    print(f"\nDone in {(time.time()-t0)/60:.1f}m")
    return n_samples, n_genes


if __name__ == '__main__':
    main()
