# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Earth Microbiome Project (EMP) 16S Release 1 — parsing helpers (Thompson et al. 2017, Nature 551:457).

Used by SENTINEL-Microbe ONLY as unlabeled community data for pretraining (protocol §10.3/§10.5 notes;
RESEARCH_PLAN §5.3): EMP carries no co-located water-quality labels and never enters a WQ claim.

* ``parse_mapping`` — the release-1 QIIME mapping TSV -> DataFrame (``#SampleID`` -> ``sample_id``).
* ``read_biom_hdf5`` — a BIOM 2.1 (HDF5) table -> long DataFrame ``sample_id, sequence, count`` for the
  requested samples. In the Deblur tables the observation id IS the exact sequence (515F-forward reads,
  primer removed, trimmed to 90/100/150 bp), so it lines up with the start of the SENTINEL V4 window
  (``sentinel.models.microbe.dna.v4_window``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE = "ftp://ftp.microbio.me/emp/release1"
FILES = {
    "mapping": f"{BASE}/mapping_files/emp_qiime_mapping_release1.tsv",
    "deblur150": f"{BASE}/otu_tables/deblur/emp_deblur_150bp.release1.biom",
}
FRESHWATER_EMPO3 = ("Water (non-saline)",)


def parse_mapping(path) -> pd.DataFrame:
    m = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    m = m.rename(columns={"#SampleID": "sample_id"})
    return m


def read_biom_hdf5(path, sample_ids=None) -> pd.DataFrame:
    """Long table of a BIOM 2.1 HDF5 file, restricted to ``sample_ids`` (column-compressed sample axis)."""
    import h5py

    with h5py.File(path, "r") as f:
        obs = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["observation/ids"][:]])
        smp = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["sample/ids"][:]])
        indptr = f["sample/matrix/indptr"][:]
        indices = f["sample/matrix/indices"][:]
        data = f["sample/matrix/data"][:]
    want = np.arange(len(smp)) if sample_ids is None else np.flatnonzero(np.isin(smp, list(sample_ids)))
    rows = []
    for j in want:
        a, b = indptr[j], indptr[j + 1]
        if b > a:
            rows.append(pd.DataFrame({"sample_id": smp[j], "sequence": obs[indices[a:b]],
                                      "count": data[a:b].astype(np.int64)}))
    if not rows:
        return pd.DataFrame(columns=["sample_id", "sequence", "count"])
    return pd.concat(rows, ignore_index=True)
