# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-BioAssess (M5): benthic community -> water chemistry.

Modules: ``data`` (NRSA design, holdout guard), ``splits`` (location-cluster CV schemes),
``features`` (training-fold tabular features), ``baselines``, ``bioset`` (taxon-embedding set
model), ``uk`` (UK EA pretraining data), ``evaluate`` (pooled / within-region metrics + CIs),
``cv`` (fold runner). Protocol: docs/preregistration/bioassess_protocol.md.
"""
