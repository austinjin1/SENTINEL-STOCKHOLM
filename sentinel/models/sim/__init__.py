# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Sim: multi-site LSTM + fixed oxygen physics scenario engine (RESEARCH_PLAN.md §7.1).

Modules: :mod:`.physics` (Benson–Krause solubility, Schmidt scaling, light, hydraulic geometry,
exact hourly DO integrator), :mod:`.data` (split-guarded site × day arrays), :mod:`.model`
(LSTM → GPP/ER/K600 + temperature/SC/turbidity heads), :mod:`.baselines` (persistence,
climatology, damped anomaly, physics-only), :mod:`.hindcast` (issue-date skill benchmark),
:mod:`.scenario` (validated perturbation API), :mod:`.checks` (metamorphic physics checks).
Protocol: docs/preregistration/sim_protocol.md.
"""
