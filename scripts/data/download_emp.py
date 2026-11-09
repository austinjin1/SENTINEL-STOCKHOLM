#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Download EMP 16S Release 1 (mapping file + Deblur 150-bp table, ~0.19 GB) and build the freshwater
subset ``processed/emp_freshwater`` (samples / tokens / sequences Parquet) for SENTINEL-Microbe unlabeled
pretraining. EMP has no water-quality labels; it never enters a WQ claim (RESEARCH_PLAN §5.3).

    python -B scripts/data/download_emp.py            # download (skips existing files) + build
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data.sources import emp as EMP

UA = "SENTINEL-v2 research (public-data download)"
DOCS = paths.REPO_ROOT / "docs" / "datasheets"


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "30", "-A", UA, "-o", str(tmp), url],
                   check=True)
    tmp.rename(dest)
    return dest


def main(argv=None):
    raw = paths.raw("emp")
    raw.mkdir(parents=True, exist_ok=True)
    files = {k: fetch(u, raw / Path(u).name) for k, u in EMP.FILES.items()}
    m = EMP.parse_mapping(files["mapping"])
    fw = m[m["empo_3"].isin(EMP.FRESHWATER_EMPO3)].copy()
    long = EMP.read_biom_hdf5(files["deblur150"], fw["sample_id"])
    long["seq_id"] = [hashlib.sha1(s.encode()).hexdigest() for s in long["sequence"]]
    seqs = long[["seq_id", "sequence"]].drop_duplicates("seq_id").assign(length=lambda d: d["sequence"].str.len())
    tok = long[["sample_id", "seq_id", "count"]]
    reads = tok.groupby("sample_id")["count"].sum()
    keep_cols = [c for c in ("sample_id", "study_id", "empo_1", "empo_2", "empo_3", "env_biome", "env_feature",
                             "env_material", "latitude_deg", "longitude_deg", "collection_timestamp", "country",
                             "pcr_primers", "target_gene", "target_subfragment") if c in fw]
    smp = fw[keep_cols].set_index("sample_id").assign(n_reads=reads).dropna(subset=["n_reads"]).reset_index()
    out = paths.processed("emp_freshwater")
    fs = []
    for name, df in (("samples", smp), ("tokens", tok), ("sequences", seqs)):
        p = out / f"{name}.parquet"
        df.to_parquet(p, compression="zstd", index=False)
        fs.append(p)
    stats = {"retrieved": dt.date.today().isoformat(), "empo_3": list(EMP.FRESHWATER_EMPO3),
             "n_mapping_rows": int(len(m)), "n_freshwater_mapping": int(len(fw)), "n_samples_with_reads": int(len(smp)),
             "n_tokens": int(len(tok)), "n_sequences": int(len(seqs)), "n_studies": int(smp["study_id"].nunique()),
             "median_reads": float(smp["n_reads"].median()),
             "raw_bytes": {k: int(v.stat().st_size) for k, v in files.items()}}
    write_manifest("emp_freshwater", files=fs + list(files.values()), sources=list(EMP.FILES.values()),
                   description="EMP 16S Release 1, Deblur 150-bp (515F-forward exact sequences), empo_3 = Water "
                               "(non-saline) samples. Unlabeled pretraining data for SENTINEL-Microbe only.",
                   license="Public EMP Release 1 data (no explicit licence on the FTP site; cite Thompson et al. "
                           "2017 Nature 551:457, as requested by the EMP)", synthetic=False, stats=stats)
    (DOCS / "emp_freshwater.md").write_text(f"""# Datasheet — emp_freshwater (EMP 16S Release 1, freshwater subset)

- **Source:** Earth Microbiome Project Release 1, {EMP.BASE} (mapping `emp_qiime_mapping_release1.tsv`,
  table `emp_deblur_150bp.release1.biom`). Retrieved {stats['retrieved']}.
- **Licence / terms:** public research data. The FTP site states no explicit licence; the EMP asks users to cite
  Thompson et al. 2017, *Nature* 551:457.
- **Use:** unlabeled community data for SENTINEL-Microbe pretraining **only**. There are no water-quality labels,
  and no EMP sample enters a WQ claim, a CV fold or a test set.
- **Content:** samples with `empo_3` = "Water (non-saline)". Deblur exact sequences of 150 bp, forward reads from
  515F (primer removed); `seq_id` = sha1(sequence). These match the first 150 bp of the SENTINEL V4 window.
- **Coverage:** {stats['n_samples_with_reads']} samples, {stats['n_studies']} studies, {stats['n_sequences']} sequences,
  {stats['n_tokens']} (sample, sequence) rows; median reads per sample {stats['median_reads']:.0f}.
- **QC:** as released (EMP Deblur QC). No further filtering at ingest.
- **Known issues:** the studies differ in protocol details, sequencing depth and geography. Mostly read length 150
  (Deblur trims to exactly 150 bp).
- **Rebuild:** `python -B scripts/data/download_emp.py`
""")
    print(stats)


if __name__ == "__main__":
    main()
