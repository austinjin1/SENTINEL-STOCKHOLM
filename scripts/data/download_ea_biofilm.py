#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Stream ENA PRJEB90117 (EA river-biofilm 16S V4, Warren et al. 2026) R1 reads one run at a time and keep only
dereplicated, quality-filtered harmonised V4 windows (docs/preregistration/microbe_ea_protocol.md §2.3).

For each read:
- locate the 515F 3' core (GC[AC]GCCGCGGT[AG]A) ending within bases 15..25;
- take the next 230 bases;
- drop the read if the window contains N or its expected errors exceed 1.0.

Per sample, exact windows with >= 2 reads are kept (no DADA2; disclosed). Each FASTQ is deleted right after
processing; disk usage is checked every 50 runs. No chemistry/labels are read here.
Output: ``processed/ea_biofilm/{samples,tokens,sequences}.parquet`` + manifest + datasheet.

    python -B scripts/data/download_ea_biofilm.py
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.manifest import write_manifest

UA = "SENTINEL-v2 research (public-data download)"
PROJECT = "PRJEB90117"
WIN, EE_MAX, MIN_COUNT, CAP_GB = 230, 1.0, 2, 5.0
CORE = ["G", "C", "AC", "G", "C", "C", "G", "C", "G", "G", "T", "AG", "A"]    # 3' 13 bp of 515F-Y
ERR = 10 ** (-(np.arange(256) - 33).clip(0, 60) / 10.0)


def ena_runs() -> pd.DataFrame:
    url = (f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={PROJECT}&result=read_run&fields="
           "run_accession,sample_accession,sample_title,collection_date,instrument_model,read_count,fastq_ftp&format=tsv")
    out = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "-A", UA, url], check=True, capture_output=True)
    return pd.read_csv(io.BytesIO(out.stdout), sep="\t")


def process_fastq(path: Path, chunk: int = 200_000) -> tuple[pd.Series, dict]:
    """Return window -> count (all windows passing the filters, before the >= 2 filter) and read stats."""
    seqs, quals, counts, stats = [], [], {}, {"reads": 0, "core": 0, "pass": 0}

    def flush():
        if not seqs:
            return
        L = WIN + 25
        n = len(seqs)
        S = np.full((n, L), ord("N"), np.uint8)
        Q = np.full((n, L), 33, np.uint8)
        for i, (s, q) in enumerate(zip(seqs, quals)):
            m = min(len(s), L)
            S[i, :m] = np.frombuffer(s[:m], np.uint8)
            Q[i, :m] = np.frombuffer(q[:m], np.uint8)
        end = np.full(n, -1)
        for e in range(15, 26):                              # core end position
            ok = np.ones(n, bool)
            for j, allowed in enumerate(CORE):
                col = S[:, e - 13 + j]
                ok &= np.isin(col, np.frombuffer(allowed.encode(), np.uint8))
            end = np.where((end < 0) & ok, e, end)
        has = (end >= 0) & (np.array([len(s) for s in seqs]) >= end + WIN)
        stats["core"] += int((end >= 0).sum())
        idx = np.flatnonzero(has)
        if idx.size:
            cols = end[idx, None] + np.arange(WIN)[None, :]
            W = S[idx[:, None], cols]
            ee = ERR[Q[idx[:, None], cols]].sum(1)
            good = (ee <= EE_MAX) & ~(W == ord("N")).any(1)
            W = W[good]
            stats["pass"] += int(good.sum())
            if len(W):
                v = W.view(f"S{WIN}").ravel()
                u, c = np.unique(v, return_counts=True)
                for w, k in zip(u, c):
                    counts[w] = counts.get(w, 0) + int(k)
        seqs.clear()
        quals.clear()

    with gzip.open(path, "rb") as fh:
        for i, line in enumerate(fh):
            r = i % 4
            if r == 1:
                s = line.rstrip()
            elif r == 3:
                stats["reads"] += 1
                if len(s) >= 15 + WIN:
                    seqs.append(s)
                    quals.append(line.rstrip())
                    if len(seqs) >= chunk:
                        flush()
    flush()
    return pd.Series(counts, dtype="int64"), stats


def main(argv=None):
    runs = ena_runs()
    raw = paths.raw("ea_biofilm")
    tmp = raw / "_stream"
    tmp.mkdir(parents=True, exist_ok=True)
    out = paths.processed("ea_biofilm")
    part_dir = out / "_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in part_dir.glob("*.parquet")}
    rows = []
    for k, r in enumerate(runs.itertuples(index=False)):
        run = r.run_accession
        r1 = str(r.fastq_ftp).split(";")[0]
        if run in done or not r1 or r1 == "nan":
            continue
        f = tmp / f"{run}_1.fastq.gz"
        subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "20", "-A", UA, "-o", str(f),
                        f"ftp://{r1}"], check=True)
        try:
            cnt, st = process_fastq(f)
        finally:
            f.unlink(missing_ok=True)                        # FASTQ deleted immediately
        keep = cnt[cnt >= MIN_COUNT]
        pd.DataFrame({"run": run, "window": [w.decode() for w in keep.index], "count": keep.to_numpy()}).to_parquet(
            part_dir / f"{run}.parquet", compression="zstd", index=False)
        rows.append({"run": run, **st, "retained_reads": int(keep.sum()), "n_windows": int(len(keep))})
        if k % 50 == 0:
            used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
            free = shutil.disk_usage(out).free / 1e9
            print(f"{k}/{len(runs)} stored {used:.2f} GB, disk free {free:.0f} GB", flush=True)
            if used > CAP_GB or free < 50:
                raise SystemExit(f"stopping: stored {used:.2f} GB / free {free:.0f} GB (cap {CAP_GB} GB)")
    tmp.rmdir()
    parts = pd.concat([pd.read_parquet(p) for p in sorted(part_dir.glob("*.parquet"))], ignore_index=True)
    parts["seq_id"] = [hashlib.sha1(w.encode()).hexdigest() for w in parts["window"]]
    seqs = parts[["seq_id", "window"]].drop_duplicates("seq_id").rename(columns={"window": "sequence"})
    tok = parts[["run", "seq_id", "count"]].rename(columns={"run": "sample_id"})
    st = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["run"])
    smp = runs.drop(columns=["fastq_ftp"]).rename(columns={"run_accession": "sample_id"})
    smp["group"] = smp["sample_title"].astype(str).str[0]
    smp["n_reads"] = smp["sample_id"].map(tok.groupby("sample_id")["count"].sum()).fillna(0).astype(int)
    smp["primer_set"] = "V4"                                  # 515F/806R; harmonised window starts after 515F
    for name, df in (("samples", smp), ("tokens", tok), ("sequences", seqs)):
        df.to_parquet(out / f"{name}.parquet", compression="zstd", index=False)
    if len(st):
        st.to_parquet(out / "run_stats_this_pass.parquet", index=False)
    stats = {"retrieved": dt.date.today().isoformat(), "project": PROJECT, "n_runs": int(len(runs)),
             "n_samples_ge_1000": int((smp["n_reads"] >= 1000).sum()), "n_sequences": int(len(seqs)),
             "n_tokens": int(len(tok)), "window_bp": WIN, "ee_max": EE_MAX, "min_count": MIN_COUNT}
    write_manifest("ea_biofilm", files=[out / f"{n}.parquet" for n in ("samples", "tokens", "sequences")],
                   sources=[f"https://www.ebi.ac.uk/ena/browser/view/{PROJECT}"],
                   description="EA river-biofilm 16S V4 (ENA PRJEB90117) R1 reads -> harmonised V4 230-bp windows, "
                               "EE<=1, exact dereplication, >=2 reads per sample (no DADA2). M3 evaluation only.",
                   license="INSDC (ENA) public data; cite Warren et al. 2026 Water Research 300:125959",
                   synthetic=False, stats=stats)
    (paths.REPO_ROOT / "docs" / "datasheets" / "ea_biofilm.md").write_text(f"""# Datasheet — ea_biofilm (EA river-biofilm 16S, England)

- **Source:** ENA {PROJECT} (Warren et al. 2026, *Water Research* 300:125959; EA River Surveillance Network,
  biofilm samples from 2021–2023). Retrieved {stats['retrieved']}. R1 reads only, streamed one run at a time;
  no FASTQ is kept.
- **Licence:** INSDC public data. Cite the paper and the accession.
- **Processing** (docs/preregistration/microbe_ea_protocol.md §2.3):
  - harmonised V4 window of 230 bp after the 515F 3′ core (bases 15–25);
  - expected errors ≤ {EE_MAX}; no N;
  - exact dereplication per run; windows with ≥ {MIN_COUNT} reads kept;
  - **no DADA2 denoising** (disclosed).
- **Coverage:** {stats['n_runs']} runs, {stats['n_samples_ge_1000']} with ≥ 1,000 retained reads,
  {stats['n_sequences']} unique windows.
- **Known issues:**
  - matrix is benthic biofilm, whereas NEON is water-column filtrate;
  - ENA lists one institutional coordinate for every run;
  - site codes are anonymised per year;
  - one target only (3-month-mean TON / nitrate-N from the authors' WIMS-derived table, not same-visit).
- **Use restriction:** **M3 evaluation data only.** No EA row is used for training or tuning. EA labels and
  derived EA tables are never passed to other workstreams (M5-X uses WIMS independently).
- **Rebuild:** `python -B scripts/data/download_ea_biofilm.py` (resumable; parts in `processed/ea_biofilm/_parts`).
""")
    print(stats)


if __name__ == "__main__":
    main()
