# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Microbe (M3): NEON surface-water 16S community -> co-located water quality.

Modules: ``design`` (load the joined NEON microbe x WQ design, development/holdout guard),
``splits`` (leave-one-domain-out, leave-one-site-out, cross-primer, temporal; frozen-split hash),
``features`` (training-fold genus CLR / presence-absence / context features), ``dna`` (k-mer and
frozen DNABERT-S sequence encoders), ``seqset`` (set transformer over DNA-encoded ASVs),
``baselines``, ``evaluate`` (pooled / per-domain / anomaly skill, CIs, permutation null,
site-identity probe), ``cv`` (fold runner). Protocol: docs/preregistration/microbe_protocol.md.
"""
