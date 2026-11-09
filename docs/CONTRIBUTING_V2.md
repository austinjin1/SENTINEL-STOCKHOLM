# SENTINEL v2 — conventions for code and data (read before writing a module)

## Environment
- Python env: `/home/bcheng/miniconda3/envs/sentinel2/bin/python` (3.11, torch 2.7.1+cu118). The package is installed editable (`pip install -e .`), so `import sentinel` works from anywhere.
- Always run Python with `-B` (no `__pycache__` in the repo). Never `git commit`/`push`; never edit files owned by another workstream.
- New pip packages: install into `sentinel2` only, then add the package to `requirements-v2.txt` (pinned `==` version) and to the right optional group in `pyproject.toml`.
- GPU use only through `sentinel.core.gpu` (`python -m sentinel.core.gpu --name <job> -- <cmd>` or `acquire_gpu()`); the budget is GPUs 0–4. Data-only work uses no GPU.

## Files and layout
- Data source clients: `sentinel/data/sources/<source>.py` (pure functions: build request → parse → tidy DataFrame). Command-line downloaders: `scripts/data/download_<source>.py`. Dataset builders (joins/matchups/features): `sentinel/data/<topic>.py` + `scripts/data/build_<dataset>.py`.
- Every `.py` file starts with:
  ```python
  # MIT License — Bryan Cheng, 2026
  # Part of SENTINEL v2
  ```
- Data locations come only from `sentinel.core.paths` (`paths.raw("<source>")`, `paths.interim(...)`, `paths.processed(...)`). Never hard-code `/home/...` paths. Raw text/zip is parsed to Parquet (zstd) and not kept unless small (< 100 MB).
- Tests: `tests/test_<module>.py` — parsing tests may use tiny recorded samples (store under `tests/fixtures/<source>/`, < 1 MB each) — never network access inside tests.

## Data rules
- **Missing codes:** convert sentinel values (−999999, −9999, 999999 where documented, non-numeric qualifier strings such as `Ice`, `Eqp`, `***`, `Dis`, `Ssn`, `ND`) to NaN at parse time; keep the original code in a `qualifier`/`flag` column. Add a test.
- **Units are part of identity:** store a `unit` column; never mix RFU and µg/L, FNU and NTU, mg/L as N vs as NO3; convert only with a documented, tested factor.
- **Time:** store UTC timestamps (`time_utc`) for instantaneous data, local `date` for daily values; keep source time-zone codes.
- **IDs as strings:** site numbers, HUC codes, COMIDs, CAS numbers are strings (leading zeros!).
- **Detection limits / censoring:** keep `<DL` information (`censored` boolean + `detection_limit` value), don't substitute DL/2 at ingest.
- **Provenance:** after a download/build, call `sentinel.core.manifest.write_manifest(name, files=..., sources=[urls], description=..., license=..., synthetic=False, stats={...})`. Record request parameters and retrieval date in `stats`.
- **Datasheet:** write `docs/datasheets/<name>.md`: source, URL, license, retrieval date, coverage (sites, time, variables), row counts, known issues, QC applied, how to rebuild (exact command).
- **Politeness:** sequential or ≤ 4 concurrent requests per host, retries with backoff, honour HTTP 429/`Retry-After`; identify via a generic User-Agent `SENTINEL-v2 research (public-data download)`. Never send personal emails/credentials in requests.
- **Credentials:** read optional keys only from environment variables (`API_USGS_PAT`, `NEON_TOKEN`) or `~/.config/sentinel/keys.env` via existing loader code; never print, log or write them, and never open/cat/grep the keys file with tools.
- **No synthetic data** in any dataset. Synthetic arrays are allowed only inside unit tests.

## Results rules (modelling workstreams)
- Every reported metric goes through `sentinel.core.registry.log_result()`; every training run is wrapped in `sentinel.core.runlog.TrainingRun`.
- Splits come from `sentinel.eval.splits` and are saved (JSON) next to results with `split_report()`.
- Metrics come from `sentinel.eval.metrics`; CIs from `sentinel.eval.bootstrap` (state the resampling unit).
- Test data are scored once, after the configuration is frozen (`docs/preregistration/`).

## CPU budget (shared 48-core host; added 2026-10-22 after load reached 86)
- Each job uses at most **8 CPU threads** (`n_jobs`/`num_threads`/`OMP_NUM_THREADS`/`torch.set_num_threads`/DataLoader workers combined), except M1 GBM training which may use 12.
- Run at most **2 CPU-heavy jobs per workstream** concurrently; queue the rest.
- Downloaders keep their existing low concurrency.

## Shared-host allocation (revised 2026-10-24 22:30 after load reached 96 with 9 workstreams)
- **CPU:** each workstream runs at most **one** CPU-heavy job at a time, ≤ **6 threads** (M1 Sonde may run two). GPU jobs set `torch.set_num_threads(2)` and ≤ 2 DataLoader workers; pytest runs count as light. Queue everything else.
- **GPU home assignment** (take the lock only on your home GPU; GPU 4 is the shared queue):
  GPU 0 = M1 Sonde · GPU 1 = M2 Orbit · GPU 2 = M7 Lite · GPU 3 = M6 HydroMet · GPU 4 = shared (M3 Microbe, M5 BioAssess, M4 ChemTox, Downstream, Fusion/Sim), first come first served, one job at a time.
- A job already holding a non-home GPU finishes its current run, then releases; do not start a new job on someone else's home GPU.
- **Other users share these GPUs (2026-11-08).** The lock helper now also requires physically free memory (`--min-free-mb`, default 3000 MiB; `--status` shows free MiB). Pass the memory your job really needs, e.g. `python -m sentinel.core.gpu --name job --prefer 4 --min-free-mb 8000 -- ...`; never touch processes of other users.
- **Overflow rule (2026-11-08, GPU 4 blocked by another user):** shared-queue workstreams (M3, M4, M5, Downstream, Fuse, Sim) may take any of GPUs 0–3 whose SENTINEL lock is free and that has the free memory they need (`--prefer 4` already falls through), but each such overflow job must finish within **4 hours** (split longer work into chunks), so home workstreams are never starved.
- **Overflow amendment (2026-11-08 15:00, after a Sim queue starved M2 on GPU 1):** overflow onto another workstream's home GPU is for **single one-off jobs only** (e.g. a dry run, a retry, one seed) — never from a queue/chain that re-acquires repeatedly. Queued or chained small jobs stay on GPU 4 and should request only the memory they really need (`--min-free-mb`).
- **GPU budget reduced to GPUs 0 and 1 (user instruction, 2026-11-08 16:00).** Enforced by the hard cap file `$SENTINEL_RUNS/.gpu_locks/ALLOWED` (= `0,1`); `SENTINEL_GPUS` can only narrow it. Never set `CUDA_VISIBLE_DEVICES` yourself or bypass the lock helper. Both GPUs are now shared by all workstreams: one job per GPU (lock), each job ≤ 2 h (chunk longer work), and pending test/freeze-critical jobs (M4 prospective scoring pass, M5 dry run, M1 EW freeze runs) go first — others don't queue new GPU jobs while those are waiting. Prefer CPU where a job fits the CPU rule.

## Event layer (added 2026-11-09; plan: `docs/plans/event_layer_plan.md`)
Every model (A) predicts water quality and (B) says which harmful event is happening. (B) is added as a
pre-registered **event track**. Steps for a workstream adding one:
1. **Use the shared module, never a copy.**
   - Labels, the derived route and metrics come from `sentinel.eval.events` (E-codes, plan §1). Import it
     directly; never export it from `sentinel/eval/__init__.py`.
   - `events.py` v1 is immutable: fixes go into `events_v2.py`. Do not edit `hazards.py`, `chloride.py` or
     any file hashed by a freeze.
   - Workstream code goes in `sentinel/events/<ws>.py`, `scripts/eval/events_<ws>.py` and
     `configs/events/*.yaml`. Never use a path matched by another workstream's freeze globs
     (`sentinel/models/{orbit,lite,hydromet}/*.py`, `scripts/*/{orbit,lite,hydromet}_*.py`, …).
2. **Write a protocol section first** (`docs/preregistration/events_<ws>_protocol.md` or a dated section of
   your protocol), before any event-track model selection. It states:
   - the E-codes claimed;
   - the cells and cluster unit;
   - the label counts, registered with `scripts/eval/events_feasibility_*.py --register` or your own
     train/val-only counter;
   - the metrics (plan §1.2): AUPRC + Brier/BSS for state, and hit rate, lead time and false alarms per
     site-year for onset.
3. **Pre-register both routes and compare them.**
   - (i) **Derived** (primary where the model has a predictive distribution):
     `events.exceedance_probability`, then one `events.recalibrate` fit on validation only, with
     reliability/ECE reported.
   - (ii) **Direct** classifier head (challenger): same inputs, training cell and tuning budget.
   - The comparison is `events.compare_routes`: paired ΔBrier/ΔAUPRC, cluster bootstrap, Holm across the
     workstream's events.
4. **Required comparators:**
   - threshold-on-baseline (the event threshold applied to the strongest simple baseline);
   - persistence and site × DOY climatology;
   - the strongest published event detector for the modality (e.g. CyAN/CI-cyano for blooms; plan §2.3).
5. **Underpowered rule.** Any (model, event) pair with < 30 validation positives or < 10 validation clusters
   carrying positives (`events.power_flag`) is reported as descriptive only. It gets no direct head and no
   claim.
6. **Test block and freeze.**
   - The event track has its own freeze (`FREEZE_events_<ws>.json`, which pins the parent model's freeze by
     sha) and a single look.
   - If your primary test block is still unscored, freeze the event track **before** your first test look
     and reuse that block.
   - If it is spent (M1 Test-A, all M5 blocks, M4 ADORE / prospective, and the ToxinWatch, MusselGuard,
     D1n, SafeSwim and BioIntegrity looks), use the new block named in plan §4.2: M1 Test-B / live Test-C,
     prospective NRC, M5-X or prospective UK EA.
   - Rows of any frozen or spent test block (e.g. KillAlert fish kills 2018–2026, ToxinWatch test lakes or
     ≥ 2022) never enter an event-track training set (RESEARCH_PLAN §15.5 item 7).
