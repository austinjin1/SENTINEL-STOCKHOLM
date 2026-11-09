# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Causal-discovery claim ladder for SENTINEL (RESEARCH_PLAN §7.2; docs/preregistration/sim_protocol.md §7).

* :mod:`.prep` — per-site daily anomaly matrices (DO deficits, not raw DO) and hourly diel matrices.
* :mod:`.synthetic` — step 1 simulation-recovery generator (declared synthetic, manifest synthetic=True).
* :mod:`.discovery` — tigramite PCMCI+ / Bagged-PCMCI+ / LPCMCI / J-PCMCI+ wrappers with explicit,
  bounded parallelism (tigramite's own bootstrap uses ``n_jobs=-1``, which is not allowed on the shared
  host, so bootstrap replicates are run in our own sequential loop) and a lagged cross-correlation baseline.
* :mod:`.controls` — positive/negative controls, scoring, replication and evidence cards.

Rungs (docs/research/06 §b.3): only links passing steps 1–4 may be called "causal evidence".
"""
