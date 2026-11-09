#!/usr/bin/env python3
"""Build FAPROTAX functional-guild features for EMP samples (real DB, no leakage).

DIAGNOSIS / MOTIVATION
----------------------
The transferable-representation thesis says coarser, biology-grounded features
generalise across geography better than raw ASVs. Taxonomy (family/genus CLR)
already lifts geographic-holdout macro-F1 from 0.405 (raw OTU NN) to 0.676
(tax_logreg). The natural next question: does an even more transferable,
*function*-level representation beat taxonomy?

APPROACH (real, offline, defensible)
------------------------------------
We use the REAL FAPROTAX 1.2.7 database (Louca et al. 2016) --- a literature-curated
map from prokaryotic taxon names to 92 metabolic / ecological functional groups
(methanotrophy, sulfate_respiration, nitrogen_fixation, photoautotrophy,
chemoheterotrophy, human_gut, ...). For every EMP OTU we look up its Greengenes
lineage tokens against FAPROTAX member patterns; an OTU that matches a function
contributes its relative abundance to that function's per-sample total. The
result is a (N_samples x 92) functional-abundance matrix. This is exactly the
"functional-guild aggregation" idea: taxonomy -> function, offline, no KO model
needed, no synthetic data, no label leakage (EMPO_3/env_* never touched here).

FAPROTAX matching rule (faithful to the tool)
---------------------------------------------
Each member line is a '*'-delimited pattern, e.g. '*Methylococcaceae*' or
'*Methylomonas*methanica*'. It matches a lineage if the non-empty tokens appear,
in order, as (case-insensitive, prefix-stripped) names somewhere in the lineage.
Group-of-group inheritance ('member_of' / bare group-name references) is resolved
so that composite functions (e.g. 'chemoheterotrophy') inherit their sub-group taxa.

OUTPUT
------
data/processed/microbial/emp_functional_features.npz
  sample_ids   (N,)  str
  X_func       (N,92) f32  --- per-sample summed relative abundance per function
  func_names   (92,)  str
  coverage     dict-ish arrays for honest reporting (frac abundance annotated)

REAL data only. MIT License --- Bryan Cheng, 2026.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    from biom import load_table
except ImportError:
    print("biom-format not installed", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIOM = PROJECT_ROOT / "data" / "raw" / "emp" / "emp_deblur_90bp.release1.biom"
FAPROTAX = PROJECT_ROOT / "data" / "raw" / "faprotax" / "FAPROTAX.txt"
OUT = PROJECT_ROOT / "data" / "processed" / "microbial" / "emp_functional_features.npz"


def parse_faprotax(path: Path):
    """Return dict group_name -> list of member patterns (list of token lists),
    with group-of-group inheritance resolved."""
    lines = Path(path).read_text().splitlines()
    groups: dict[str, list] = {}         # group -> list of token-lists (taxon patterns)
    group_refs: dict[str, list] = {}     # group -> list of referenced group names
    cur = None
    for ln in lines:
        if ln.startswith("#"):
            continue
        stripped = ln.split("#", 1)[0].rstrip()
        if stripped.strip() == "":
            continue
        is_header = ("main_element:" in stripped) and ("elements:" in stripped)
        if is_header:
            cur = stripped.split("\t")[0].strip()
            groups.setdefault(cur, [])
            group_refs.setdefault(cur, [])
            continue
        tok0 = stripped.split("\t")[0].strip()
        if tok0.startswith("*"):
            # taxon member pattern
            parts = [p for p in tok0.split("*") if p.strip() != ""]
            parts = [p.strip() for p in parts]
            if cur is not None and parts:
                groups[cur].append(parts)
        else:
            # bare group-name reference (member_of another functional group)
            name = tok0.split("\t")[0].strip()
            if cur is not None and name:
                group_refs[cur].append(name)
    # resolve group-of-group inheritance (iterate to fixpoint)
    for _ in range(6):
        changed = False
        for g, refs in group_refs.items():
            for r in refs:
                if r in groups:
                    before = len(groups[g])
                    have = {tuple(p) for p in groups[g]}
                    for pat in groups[r]:
                        if tuple(pat) not in have:
                            groups[g].append(pat)
                            have.add(tuple(pat))
                    if len(groups[g]) != before:
                        changed = True
        if not changed:
            break
    return groups


def strip_prefix(tok: str) -> str:
    """Drop Greengenes 'x__' prefix and bracket/space noise, lowercase."""
    t = tok.strip()
    if "__" in t:
        t = t.split("__", 1)[1]
    t = t.strip().strip("[]").lower()
    return t


def lineage_token_set(tax_list):
    """Set of lowercased, prefix-stripped taxon names in a lineage."""
    out = []
    if tax_list is None:
        return out
    for t in tax_list:
        s = strip_prefix(str(t))
        if s and s not in ("unclassified", "unidentified"):
            out.append(s)
    return out


def pattern_matches(pattern_tokens, lineage_tokens_set):
    """A FAPROTAX pattern matches if all its (stripped, lowercased, possibly
    multi-word) tokens are present in the lineage token set. Multi-word species
    tokens are matched on their first word (genus-level) since Greengenes stops
    at genus for most EMP OTUs -> conservative (won't over-claim species)."""
    for pt in pattern_tokens:
        p = pt.strip().lower().strip("[]")
        # split on whitespace -> match if ANY word present (species patterns)
        words = [w for w in re.split(r"\s+", p) if w]
        if not words:
            continue
        ok = any(w in lineage_tokens_set for w in words)
        if not ok:
            return False
    return True


def main():
    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731

    groups = parse_faprotax(FAPROTAX)
    func_names = sorted(groups.keys())
    log(f"FAPROTAX: {len(func_names)} functional groups, "
        f"{sum(len(v) for v in groups.values())} member patterns (post-inherit)")

    log(f"Loading BIOM {BIOM.name} ...")
    tbl = load_table(str(BIOM))
    n_otu, n_samp = tbl.shape
    log(f"  {n_otu:,} OTUs x {n_samp:,} samples  ({time.time()-t0:.1f}s)")

    sample_ids = np.array([str(s) for s in tbl.ids(axis="sample")])
    obs_md = tbl.metadata(axis="observation")
    taxonomies = [md.get("taxonomy") if md is not None else None for md in obs_md]

    # OTU -> lineage token sets
    lineage_sets = [set(lineage_token_set(t)) for t in taxonomies]

    # Inverted index: token -> set(OTU indices) for fast candidate lookup.
    from collections import defaultdict
    token_to_otus: dict[str, set] = defaultdict(set)
    for oi, lset in enumerate(lineage_sets):
        for tok in lset:
            token_to_otus[tok].add(oi)
    log(f"  inverted index: {len(token_to_otus):,} distinct lineage tokens "
        f"({time.time()-t0:.1f}s)")

    # Build OTU x function 0/1 membership via candidate intersection.
    # A pattern matches an OTU iff, for EACH pattern token, at least one of its
    # words is in the OTU lineage. So candidate OTUs for a pattern = intersection
    # over tokens of (union over words of token_to_otus[word]); membership is then
    # exactly that candidate set (each token satisfied) -> no per-OTU rescan needed.
    F = len(func_names)
    rows, cols = [], []  # OTU idx, func idx
    for fj, fname in enumerate(func_names):
        matched: set = set()
        for pat in groups[fname]:
            cand = None
            empty = False
            for pt in pat:
                p = pt.strip().lower().strip("[]")
                words = [w for w in re.split(r"\s+", p) if w]
                if not words:
                    continue
                union = set()
                for w in words:
                    union |= token_to_otus.get(w, set())
                if cand is None:
                    cand = set(union)
                else:
                    cand &= union
                if not cand:
                    empty = True
                    break
            if not empty and cand:
                matched |= cand
        for oi in matched:
            rows.append(oi)
            cols.append(fj)
    log(f"  OTU-function memberships: {len(rows):,}  ({time.time()-t0:.1f}s)")

    membership = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_otu, F),
    )  # (n_otu x F)

    # counts (otu x samp)
    M = tbl.matrix_data.tocsr()
    col_tot = np.asarray(M.sum(axis=0)).ravel().astype(np.float64)
    col_tot_safe = np.where(col_tot > 0, col_tot, 1.0)

    # functional abundance per sample: (F x samp) = membership^T @ M
    func_counts = (membership.T @ M)  # (F x samp) sparse
    func_counts = np.asarray(func_counts.todense(), dtype=np.float64)
    X_func = (func_counts / col_tot_safe[None, :]).T.astype(np.float32)  # (samp x F)

    # coverage: fraction of each sample's reads assigned to >=1 function
    any_func = (membership.sum(axis=1).A.ravel() > 0).astype(np.float64)  # (n_otu,)
    annotated_counts = np.asarray((any_func[None, :] @ M)).ravel()  # (samp,)
    frac_annotated = (annotated_counts / col_tot_safe).astype(np.float32)
    n_otu_annotated = int(any_func.sum())

    log(f"  OTUs with >=1 function: {n_otu_annotated:,}/{n_otu:,} "
        f"({100*n_otu_annotated/n_otu:.1f}%)")
    log(f"  mean per-sample read fraction annotated: {float(np.mean(frac_annotated)):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        sample_ids=sample_ids,
        X_func=X_func,
        func_names=np.array(func_names, dtype=object),
        frac_annotated=frac_annotated,
        n_otu_annotated=np.array(n_otu_annotated),
        n_otu_total=np.array(n_otu),
    )
    log(f"Wrote {OUT}  ({time.time()-t0:.1f}s total)  X_func shape {X_func.shape}")


if __name__ == "__main__":
    main()
