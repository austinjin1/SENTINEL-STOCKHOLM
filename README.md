# SENTINEL v2

**Honest, multimodal deep learning for freshwater water-quality forecasting, disaster early warning and ecological insight**

Austin Jin & Bryan Cheng · Stockholm Junior Water Prize · AI-assisted (see `RESEARCH_PLAN.md` §21)

> **Status (2026-10-22): rebuild in progress — no v2 results yet.** SENTINEL v1 results were
> retracted after an integrity review; v1 code is archived in [`legacy/`](legacy/README.md).
> Every number this project reports will come from `results/registry.jsonl` and is rendered
> into [`RESULTS.md`](RESULTS.md). If a number is not there, it is not a result.

## What v2 is building

| Goal | Component |
|---|---|
| Modality models (water-quality indicators + disasters) | M1 **Sonde** (in-situ sensors) · M2 **Orbit** (satellite) · M3 **Microbe** (16S/ASV) · M4 **ChemTox** (chemical structure → aquatic toxicity) · M5 **BioAssess** (macroinvertebrate/fish communities) · M6 **HydroMet** (weather + catchment) · M7 **Lite** (drone RGB+NIR) |
| Plug-and-play multimodal fusion | **SENTINEL-Fuse** (any subset of modalities, all targets × horizons) |
| Simulation & insight | **SENTINEL-Sim** (physics-guided, metabolism-aware scenarios), causal-discovery claim ladder, interpretability |
| Downstream applications | SafeSwim · ToxinWatch · KillAlert · TapRisk · BioIntegrity · MusselGuard · TroutRefuge · RiverPulse |
| Early warning | **EW-v2** protocol + pre-registered live prospective trial |
| Platform | **SENTINEL Console** (API + web app) |

Full design, benchmarks, baselines and success criteria: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md).
Literature and data landscape: [`LANDSCAPE_SURVEY.md`](LANDSCAPE_SURVEY.md) and [`docs/research/`](docs/research/).

## Layout

```
sentinel/   v2 package (core: registry, manifests, GPU locks, run log · eval: metrics, splits,
            bootstrap, early-warning protocol · data · models · sim · causal · downstream · platform)
scripts/    command-line entry points        tests/   unit + leakage tests
configs/    experiment configs               docs/    research, pre-registration, model cards
results/    registry.jsonl (single source of truth for numbers)
legacy/     SENTINEL v1 (archived; results retracted)
```

Large data and checkpoints live outside the repository: `$SENTINEL_DATA` (default
`~/STOCKHOLM/sentinel_data`) and `$SENTINEL_RUNS` (default `~/STOCKHOLM/sentinel_runs`).

## Quick start

```bash
conda create -n sentinel2 python=3.11 && conda activate sentinel2
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-v2.txt
pip install -e .
pytest                      # unit + leakage tests
python -m sentinel.core.gpu --status
```

## Integrity rules (short form)

1. No number without code that computed it and a registry entry.
2. Figures and tables read numbers only through `sentinel.core.registry.lookup()`.
3. Splits are leakage-checked in code; test sets are scored once, after freezing.
4. Every table includes the strongest simple baseline; results are credited to the model that produced them.
5. Every training run — including failures — is logged automatically in `TRAINING_LOG.md`.

## License

MIT — see [`LICENSE`](LICENSE).
