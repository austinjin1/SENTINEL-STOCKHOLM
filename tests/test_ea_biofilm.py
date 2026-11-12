# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EA biofilm read processing on a tiny SYNTHETIC FASTQ (code-path test only)."""

import gzip
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "dl_ea", Path(__file__).resolve().parents[1] / "scripts" / "data" / "download_ea_biofilm.py")
EA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EA)


def test_window_extraction_quality_and_counts(tmp_path):
    rs = np.random.default_rng(0)
    v4 = "".join(rs.choice(list("ACGT"), 240))
    good = "GTGCCAGCAGCCGCGGTAA" + v4                       # core ends at 19
    other = "GTGCCAGCAGCCGCGGTAA" + "".join(rs.choice(list("ACGT"), 240))
    recs = [(good, "I" * len(good))] * 3 + [(other, "I" * len(other))] + \
           [(good, "#" * len(good))] + [("ACGT" * 9, "I" * 36)]          # low quality, short junk
    p = tmp_path / "r1.fastq.gz"
    with gzip.open(p, "wt") as fh:
        for i, (s, q) in enumerate(recs):
            fh.write(f"@r{i}\n{s}\n+\n{q}\n")
    cnt, st = EA.process_fastq(p)
    assert st["reads"] == 6 and st["pass"] == 4
    assert cnt[v4[:EA.WIN].encode()] == 3 and len(cnt) == 2
