# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Orbit architecture round v2 (docs/preregistration/orbit_arch_v2_amendment.md).

Everything here lives outside the v1 freeze globs (``sentinel/models/orbit/*.py`` etc.), so that v2 work can never
change the v1 scoring-code hash. v1 primitives are imported, never copied.

Modules:
* ``stamp``     — the lead-stamp gate: no validation number of a new arm or a Track N target before the stamp;
* ``targets``   — Track N / auxiliary targets: label rules (§2.4), co-located join, counts-only for protected rows;
* ``chipfeat``  — per-chip image features: shore distance d_p, land ring, glint geometry (§3.4);
* ``context``   — lake-context set C(w, t) (§3.4 item 3) and the development context cache with a refusing loader;
* ``lakeset``   — SatWQ-LakeSet-MT (arm A1);
* ``train``     — EMA seed-unit trainer (§3.2), interval likelihood, checkpoint/resume;
* ``devm``      — dev_v5m cache (write/load).
"""
