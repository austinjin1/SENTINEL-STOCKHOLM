# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Downstream applications (RESEARCH_PLAN.md §8, Goal 4): D1 SafeSwim, D2 ToxinWatch, D3 KillAlert,
D5 BioIntegrity, D6 MusselGuard, D8 RiverPulse.

Shared plumbing lives in :mod:`.common` (frozen splits, test guard, registry wrapper, bootstrap
evaluation, CPU slots), :mod:`.weather` (observed gridMET antecedent weather at arbitrary points) and
:mod:`.crossfit` (the hook for upstream M1/M2/M6 features under the cross-fitting rule; not used in
phase 1).
"""
