#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EA-dev / EA-test partition of the EA river-biofilm samples (ENA PRJEB90117) by anonymised letter group.

Label-free: the only input is the ENA sample report (run accession, sample title, collection date). No
chemistry file is read. Rule (docs/preregistration/microbe_arch_v2_amendment.md §10.2):

- a sample belongs to a letter group if its title matches ``^[A-Z]\\d+_\\d{4}$``; the group is the first letter;
  other titles (blanks, malformed) are in neither partition;
- groups are sorted; ``numpy.random.default_rng(seed).permutation(n_groups)``; the first ``n_test`` are EA-test;
- the draw is accepted if EA-test holds 25-40 % of the grouped runs, otherwise seed + 1 (every draw recorded).

    python -I scripts/data/microbe_ea_partition.py <ena_samples.tsv> <out.json> [--seed 20261110] [--n-test 7]
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re

import numpy as np

STD = re.compile(r"^[A-Z]\d+_\d{4}$")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--n-test", type=int, default=7)
    a = ap.parse_args(argv)
    raw = open(a.tsv, "rb").read()
    rows = list(csv.DictReader(raw.decode().splitlines(), delimiter="\t"))
    titles = [(r.get("sample_title") or "") for r in rows]
    grouped = [t for t in titles if STD.match(t)]
    other = sorted(t for t in titles if not STD.match(t))
    n_by_group = collections.Counter(t[0] for t in grouped)
    n_by_group_year = collections.Counter((t[0], t.split("_")[1]) for t in grouped)
    groups = sorted(n_by_group)
    total, draws, seed = len(grouped), [], a.seed
    while True:
        perm = np.random.default_rng(seed).permutation(len(groups))
        test = sorted(groups[i] for i in perm[: a.n_test])
        n = sum(n_by_group[g] for g in test)
        ok = 0.25 <= n / total <= 0.40
        draws.append({"seed": seed, "ea_test_groups": test, "n_test_runs": n, "fraction": round(n / total, 4),
                      "accepted": ok})
        if ok:
            break
        seed += 1
    dev = [g for g in groups if g not in test]
    years = sorted({y for _, y in n_by_group_year})
    out = {
        "rule": "sorted letter groups; numpy default_rng(seed).permutation; first n_test = EA-test; accept if "
                "EA-test holds 25-40 % of grouped runs, else seed + 1",
        "input_sha256": hashlib.sha256(raw).hexdigest(), "numpy": np.__version__,
        "n_runs": len(rows), "n_unique_titles": len(set(titles)), "n_grouped_runs": total,
        "titles_in_neither_partition": other,
        "groups": {g: {"n_runs": n_by_group[g], **{y: n_by_group_year[(g, y)] for y in years}} for g in groups},
        "draws": draws,
        "ea_test_groups": test, "ea_dev_groups": dev,
        "n_test_runs": sum(n_by_group[g] for g in test), "n_dev_runs": sum(n_by_group[g] for g in dev),
        "runs_by_title_year": {p: {y: sum(n_by_group_year[(g, y)] for g in gs) for y in years}
                               for p, gs in (("ea_test", test), ("ea_dev", dev))},
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("n_runs", "n_grouped_runs", "draws", "ea_dev_groups", "n_dev_runs",
                                           "n_test_runs", "runs_by_title_year", "numpy")}, indent=1))


if __name__ == "__main__":
    main()
