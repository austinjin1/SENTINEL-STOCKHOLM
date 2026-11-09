# SENTINEL Console — platform guide

The SENTINEL Console is the v2 platform (RESEARCH_PLAN.md §9): a FastAPI service and a Streamlit
web app over the project's local data products, results registry, model cards and pre-registration
documents. It is a scaffold. **No v2 model has been trained yet**, so the Console shows real
observations with their provenance, registry entries exactly as logged, and explicit
"not yet available" states for everything else.

The v1 demo was retracted partly because it displayed generated values labelled as observed and
claimed that every figure was real. The Console is built so that this cannot recur: see
[Integrity guarantees](#integrity-guarantees).

## Run it

Environment: the `sentinel2` conda env (`/home/bcheng/miniconda3/envs/sentinel2/bin/python`); all
platform dependencies are pinned in `requirements-v2.txt` (`pip install -e ".[platform]"` elsewhere).

```bash
# API (http://127.0.0.1:8000, interactive docs at /docs)
python -B -m sentinel.platform.api --port 8000
# or: uvicorn --factory sentinel.platform.api.app:create_app --port 8000

# Web app (http://127.0.0.1:8501); the API is needed only for upload-and-predict and live values
SENTINEL_API_URL=http://127.0.0.1:8000 streamlit run sentinel/platform/app/streamlit_app.py \
    --server.headless true --browser.gatherUsageStats false --server.port 8501

# Docker (CPU): host directories are mounted read-only
export SENTINEL_DATA=~/STOCKHOLM/sentinel_data SENTINEL_RUNS=~/STOCKHOLM/sentinel_runs
docker compose -f docker/docker-compose.yml up --build

# Tests (no network)
python -B -m pytest tests/test_platform.py
```

| Variable | Meaning | Default |
|---|---|---|
| `SENTINEL_DATA`, `SENTINEL_RUNS` | data and run roots (via `sentinel.core.paths`) | `~/STOCKHOLM/sentinel_data`, `~/STOCKHOLM/sentinel_runs` |
| `SENTINEL_API_URL` | where the app finds the API | `http://127.0.0.1:8000` |
| `SENTINEL_API_HOST`, `SENTINEL_API_PORT` | API bind address | `127.0.0.1`, `8000` |
| `SENTINEL_LIVE` | `0` disables the live USGS fetch | enabled |
| `API_USGS_PAT` | optional USGS Water Data API key (or `~/.config/sentinel/keys.env`) | none (anonymous, low rate limit) |

## Architecture

```
            $SENTINEL_DATA (read-only)          repo (read-only)               $SENTINEL_RUNS (read-only)
  processed/sites/sites.parquet         results/registry.jsonl            live/  (live-trial forecast files)
  raw/usgs_dv/dv_series_catalog.parquet docs/model_cards/*.md
  processed/sonde_daily/{index,sites/*} docs/preregistration/*.md
  processed/sonde_hourly/sites/*        RESULTS.md, RESEARCH_PLAN.md
  manifests/*.json                      sentinel/platform/models.yaml
              │                                   │
              ▼                                   ▼
  sentinel/platform/  data_access · provenance · registry_view · model_cards · plugins
                      validation · live (USGS OGC API) · status · hazard_refs · guard
              │                                   │
              ▼                                   ▼
     api/app.py  (FastAPI)  ◄──── HTTP ────  app/streamlit_app.py (Streamlit, 7 pages)
     /health /sites /live /models /registry      Home · Map · Site · Models · Data upload
     /predict/{modality} /alerts(501) ...         Early warning · Integrity
```

Both front ends use the same library modules, so the integrity rules are enforced in one place.
The app reads local products directly (it works without the API) and calls the API for
upload-and-predict and live values. Nothing in `sentinel/platform` writes to the data, runs or
repository directories.

| Module | Role |
|---|---|
| `config.py` | `Settings` built at call time from `sentinel.core.paths`; upload limits; `set_overrides()` for tests |
| `data_access.py` | site catalogue (processed site table cross-checked against the raw USGS catalogue), daily/hourly availability, observations; mtime-invalidated caches |
| `provenance.py` | provenance blocks: manifests (followed upstream), files read, approval counts, synthetic flag |
| `guard.py` | metric guard and the literal-claim scanner |
| `registry_view.py` | read-only registry views, test-split detection, status counts |
| `model_cards.py` | card front matter joined to current registry entries |
| `plugins.py` | `models.yaml` loader, serving gate, plugin output checks |
| `validation.py` | per-modality upload validation |
| `live.py` | USGS Water Data OGC API client (`latest-continuous`, `continuous`) |
| `status.py` | RESULTS.md headline, plan sections, pre-registration docs, live-trial status |
| `hazard_refs.py` | EW-v2 thresholds and anchors from `sentinel/eval/hazards.py` |
| `app/` | Streamlit entry script, pages (`views/`), chart builder, shared UI helpers |

## API

| Endpoint | Purpose | Notes |
|---|---|---|
| `GET /health` | service, data, registry, plugin, live and live-trial status | includes data-consistency warnings |
| `GET /sites` | sites with id, name, lat, lon, type, HUC, variables per tier, record span | filters `var`, `tier`, `huc` (prefix), `state`, `site_type`, `bbox=minLon,minLat,maxLon,maxLat`, `q`, `has_data=true\|false\|any`, `limit`, `offset` |
| `GET /sites/{id}` | one site: metadata, per-variable coverage, hourly range, applicable hazards | `USGS-` prefix accepted |
| `GET /sites/{id}/observations` | observations in long format | `var` (base name such as `do_mgl` or column such as `do_mgl_min`; repeat or comma-separate), `start`, `end`, `tier=daily\|hourly`, `limit`, `offset` |
| `GET /live/{id}` | latest (or `window_hours` ≤ 168) real-time USGS values, labelled "provisional, live" | 503 with `Retry-After` while rate-limited |
| `GET /models` | model cards + current registry entries per card, registry entries without a card, plugin status per modality | |
| `GET /registry` | read-only registry entries (current only) | filters `workstream`, `experiment`, `model`, `split`, `metric`, `target`, `is_baseline` |
| `GET /hazards` | EW-v2 hazard table with regulatory anchors | from `sentinel/eval/hazards.py` |
| `GET /provenance/{name}` | one manifest summary | 404 if the manifest does not exist |
| `POST /predict/{modality}` | upload-and-predict; multipart `file` (+ optional JSON `options`) | 404 unknown modality, 411/413 size, 422 validation errors, 503 `{"status": "model not yet trained"}` or `"model not validated"`, 500 plugin contract violation |
| `GET /alerts`, `POST /simulate` | 501 until EW-v2 alerts and SENTINEL-Sim exist | |
| `GET /forecast/{id}`, `/causal/{id}`, `/downstream/{app}` | 501 until those components exist | |

Every data response carries `provenance`:

```json
{"generated_utc": "...", "datasets": ["sonde_daily"],
 "manifests": [{"dataset": "sonde_daily", "name": "sonde_daily", "status": "present|missing",
                "created_utc": "...", "sources": ["manifest:usgs_dv_raw"], "license": "...",
                "synthetic": false, "retrieval": {...}, "git_commit": "...", "upstream": [...]}],
 "synthetic": false, "files": [{"path": "$SENTINEL_DATA/...", "modified_utc": "...", "bytes": 0}],
 "approval": {"n_values": 0, "n_approved": 0, "n_provisional": 0, "rule": "..."}, "notes": ["..."]}
```

A manifest that does not exist yet is reported with `"status": "missing"`; nothing is filled in.
Observation rows carry a `provisional` flag (USGS qualifier "P" at retrieval time) and, for the
daily tier, the source USGS time-series id; hourly rows carry `n_raw`, the number of 15-min values
in the hour.

## Plugin interface

How a model workstream makes its model available to `POST /predict/{modality}`:

1. **Implement the entry point** anywhere in the `sentinel` package, for example
   `sentinel/models/sonde/serve.py`:

   ```python
   from sentinel.platform.validation import ValidatedUpload

   def predict(upload: ValidatedUpload) -> dict:
       # upload.data: parsed, validated input (see below); upload.options: user options (dict)
       # upload.summary / upload.warnings: validation report; upload.sha256: input hash
       return {
           "predictions": [  # list of JSON records; each names its target and time/location
               {"time_utc": "...", "target": "do_mgl", "horizon_days": 1, "q05": ..., "q50": ..., "q95": ...},
           ],
           "units": {"do_mgl": "mg/L"},   # required: every predicted quantity's unit
           "model_version": "<git commit or checkpoint sha256>",
       }
   ```

   Load weights lazily and cache them (for example with `functools.lru_cache`). The call runs in a
   worker thread; it must be CPU-only and deterministic for a given input.

2. **What `upload.data` contains** (already validated by `sentinel/platform/validation.py`):

   | Modality | `upload.data` |
   |---|---|
   | `sonde` | DataFrame: one time column (`time_utc` UTC, or `date`), SENTINEL variable columns in their native units, missing codes as NaN, optional `site_id`/`*_qualifier` columns, sorted by time |
   | `orbit` | `{"array": (bands, H, W) ndarray, "band_names": [...] or None, "georeference": {crs, transform, nodata}}` |
   | `lite` | `{"rgbn": (4, H, W) ndarray in R, G, B, NIR order, "georeference": {...}}` |
   | `microbe` | `{"counts": DataFrame features × samples (index = feature id), "side": taxonomy/sequence columns or None}` |
   | `chemtox` | list of `{"line", "identifier", "kind": "smiles" or "cas"}` (CAS check digit verified; SMILES parsed by RDKit when installed) |
   | `bioassess` | long DataFrame with `sample_id`, `taxon`, `count` (+ optional `taxonomic_level`, `life_stage`, `date`, `lat`, `lon`, `site_id`) |

3. **Register it** in `sentinel/platform/models.yaml` (one enabled entry per modality):

   ```yaml
   models:
     - modality: sonde
       model_id: sentinel_sonde             # the registry "model" name of this model
       entry_point: "sentinel.models.sonde.serve:predict"
       required_inputs: [time_utc, do_mgl, temp_c]   # sonde: columns; orbit/lite: band names;
                                                     # microbe/bioassess: columns; chemtox: smiles/cas
       registry_experiment: m1_sonde_forecast        # registry experiment with its validated metrics
       model_card: docs/model_cards/sentinel_sonde.md
   ```

4. **Serving gate.** The Console serves the entry only when the model card exists and
   `results/registry.jsonl` holds at least one current entry for (`registry_experiment`,
   `model_id`); otherwise `POST /predict` answers 503 `"model not validated"`. With no entry for a
   modality it answers 503 `"model not yet trained"` (after validating the upload, so users can
   check their files).

5. **Output rules, enforced.** The response must be a dict with `predictions` (list), `units`
   (non-empty mapping) and `model_version`, serialisable as strict JSON (no NaN). Top-level keys
   that report performance (`metrics`, `skill`, `accuracy`, `score`, `nse`, `auc`, `rmse`, ...)
   are rejected with HTTP 500: skill shown next to predictions comes only from the registry,
   under `validated_metrics`, resolved through the metric guard. Every response is labelled
   "MODEL OUTPUT, not an observation" and carries the input sha256, entry point, card and
   registry experiment.

Model cards: copy the template in `docs/model_cards/README.md`.

## Integrity guarantees

- **Registry-only metrics.** `sentinel/platform/guard.py` is the only path to a displayed metric.
  It accepts a registry entry id or registry row, re-reads `results/registry.jsonl`, and refuses
  raw numbers (Python, numpy and Decimal numbers, numeric strings), unknown ids, superseded entries
  and rows whose fields differ from the stored entry. `/models`, `/registry`, `/predict` and the
  Models and Data-upload pages all go through it.
- **No typed-in claims.** `tests/test_platform.py` scans every string in `sentinel/platform`
  (Python string constants, f-string text, YAML, Markdown) plus this guide and the model-card
  template for a metric name followed by a number, percentages, plus-minus intervals, event-count
  fractions and claim labels, and fails on any finding. It also fails if platform code uses a
  random-number generator. Run it by hand with `python -B -m sentinel.platform.guard`.
- **Validated-results banner.** Every page shows "No validated v2 results yet" while the registry
  has no non-baseline entry on a test split (a split name with a `/`, `:` or `|` component that
  starts with `test`, e.g. `test`, `test_a`, `t-F2F/occurrence/test`). Name live-trial splits
  `test_live` so they count.
- **Provenance everywhere.** Data responses and page footers name the manifests (with creation
  dates, sources, licence, code commit, `synthetic` flag, upstream manifests) and the files read;
  a synthetic manifest turns on a red banner.
- **Consistent site metadata.** Rows of `processed/sites/sites.parquet` are cross-checked against
  the raw USGS series catalogue; where the coordinates disagree (or, without coordinates, the
  state), the raw USGS row is shown, the site is flagged `catalog_conflict` and the disagreement is
  reported in `/health` and provenance.
- **Provisional is visible.** USGS provisional values are drawn as a dotted line in a different
  colour from approved values and counted in provenance. Live values are labelled "provisional,
  live", are cached only in memory, and are never used for any reported metric.
- **Hazard thresholds are not typed in.** Reference lines and their regulatory anchors come from
  `sentinel/eval/hazards.py` (the pre-registered definitions) and are labelled as
  criterion-anchored excursions, not regulatory determinations.
- **Empty states name what is missing.** Missing data products, manifests, model cards, models
  and live-trial files are reported as missing; no page or endpoint substitutes sample data.
- **Credentials.** `API_USGS_PAT` is read only from the environment or
  `~/.config/sentinel/keys.env`, sent only to `api.waterdata.usgs.gov` as the `X-Api-Key` header,
  and never logged or returned (tested).

## Live USGS values

`GET /live/{site}` calls the USGS Water Data OGC API (`/ogcapi/v1/collections/latest-continuous`,
or `continuous` with `time=PT<h>H`). At most two upstream requests run concurrently, responses
are cached for five minutes, 5xx responses are retried once, and an HTTP 429 blocks further
upstream calls until its `Retry-After` has passed (the API answers 503 with `Retry-After`
meanwhile). Pagination follows `next` links only on the same host.

## Known gaps

- No models, forecasts, alerts, simulation, causal or downstream results exist yet; those routes
  return 501/503 by design.
- Manifests for `sonde_daily` and `sonde_hourly` are written only when their builds are finalised;
  until then provenance reports them as missing and lists file modification times instead.
- The Docker image has not been built on this server (no Docker here); the Dockerfile and compose
  file are untested.
- The Streamlit app reads the registry and data directly; there is no authentication. Bind it to
  localhost (as the compose file does) or put it behind an authenticating proxy before exposing it.
- The live-trial check only detects forecast files under `$SENTINEL_RUNS/live/`; scoring the trial
  is the EW-v2 workstream's job and appears here only as registry entries.
