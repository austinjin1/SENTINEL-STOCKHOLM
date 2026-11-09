# legacy/ — SENTINEL v1 (archived, results retracted)

This directory preserves the SENTINEL v1 code, results, demo and paper exactly as they
stood on `main` at commit `e18e312` (2026-08-17), for provenance only.

**Do not cite, reuse or present any number from this directory.** A 2026-10-21
integrity review (kept by the authors outside the repository; the reusable-asset
inventory is in `../docs/research/10_local_inventory.md`) found that most v1 headline
results were invalid: figures showing retracted or hard-coded values, an early-warning
detector that failed a placebo test, results attributed to the wrong models, synthetic or
circular data presented as real, and incorrect references.

| Path | What it was |
|---|---|
| `sentinel/` | v1 Python package (importable only with `PYTHONPATH=legacy`) |
| `scripts/`, `configs/`, `tests/` | v1 experiment, training and test code |
| `results/` | v1 result files (retracted) |
| `demo/` | v1 React demo (contained randomly generated data labelled as observed) |
| `paper_v1/` | "Stockholm Paper Final" (.docx/.pdf, Aug 2026) |
| `README_v1.md`, `Dockerfile`, `docker-compose.yml`, `environment.yml`, `requirements.txt`, `pyproject.toml` | v1 top-level files |

SENTINEL v2 (repository root) is a rebuild. Where v1 code was sound (clean downloaders and
evaluation protocols identified in `docs/research/10_local_inventory.md`), it is ported
into the v2 package with tests rather than imported from here.
