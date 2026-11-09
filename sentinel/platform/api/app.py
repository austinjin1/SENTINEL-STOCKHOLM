# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL Console API.

Read-only views of local data products, the results registry and model cards, an optional
live USGS fetch, and upload-and-predict for every modality. Every data response carries a
``provenance`` block; every metric is resolved through :mod:`sentinel.platform.guard`.
Components that do not exist yet answer HTTP 501 (alerts, simulation, forecasts, causal,
downstream) or 503 (no trained model for a modality) with a message saying what is missing.

    python -m sentinel.platform.api --port 8000
    uvicorn --factory sentinel.platform.api.app:create_app
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ... import __version__
from .. import data_access as D
from .. import hazard_refs, live, model_cards, plugins, provenance, registry_view, status, validation
from ..config import MODALITIES, Settings
from ..guard import MetricGuardError

log = logging.getLogger("sentinel.platform.api")
MANIFEST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
FORM_OVERHEAD = 1 << 20


def clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars -> Python, NaN/inf -> None, paths/timestamps -> strings."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def respond(payload: Any, status_code: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(clean(payload), status_code=status_code, headers=headers)


def not_implemented(component: str, detail: str) -> JSONResponse:
    return respond({"status": "not implemented", "component": component, "detail": detail}, 501)


def _split_vars(var: list[str] | None) -> list[str] | None:
    if not var:
        return None
    out = [v.strip() for item in var for v in item.split(",") if v.strip()]
    return out or None


def _bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    try:
        parts = [float(x) for x in raw.split(",")]
    except ValueError as e:
        raise D.BadRequest("bbox must be minLon,minLat,maxLon,maxLat") from e
    if len(parts) != 4 or parts[0] > parts[2] or parts[1] > parts[3]:
        raise D.BadRequest("bbox must be minLon,minLat,maxLon,maxLat")
    return parts[0], parts[1], parts[2], parts[3]


class UploadSizeLimit:
    """ASGI middleware: reject /predict uploads above the modality limit before parsing."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path", "").startswith("/predict/")
        ):
            modality = scope["path"].split("/")[2] if scope["path"].count("/") >= 2 else ""
            limit = self.settings.upload_limits.get(modality, max(self.settings.upload_limits.values()))
            headers = dict(scope.get("headers") or [])
            raw = headers.get(b"content-length")
            if raw is None:
                await JSONResponse(
                    {"status": "length required", "detail": "uploads must send a Content-Length header"},
                    status_code=411,
                )(scope, receive, send)
                return
            try:
                n = int(raw)
            except ValueError:
                n = -1
            if n < 0 or n > limit + FORM_OVERHEAD:
                await JSONResponse(
                    {"status": "upload too large", "modality": modality, "limit_bytes": limit},
                    status_code=413,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None, *, live_client: live.LiveClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="SENTINEL Console API",
        version=__version__,
        description="SENTINEL v2 platform API. Data responses carry provenance; metrics come only from "
        "the results registry; components that do not exist yet return 501/503.",
    )
    app.state.settings = settings
    app.state.live = live_client or live.LiveClient(settings.usgs_api_base, keys_file=str(settings.keys_file))
    app.add_middleware(UploadSizeLimit, settings=settings)

    @app.exception_handler(D.DataUnavailable)
    async def _unavailable(_req: Request, exc: D.DataUnavailable):
        return respond({"status": "not found", "detail": str(exc)}, 404)

    @app.exception_handler(D.BadRequest)
    async def _bad(_req: Request, exc: D.BadRequest):
        return respond({"status": "invalid request", "detail": str(exc)}, 422)

    @app.exception_handler(MetricGuardError)
    async def _guard(_req: Request, exc: MetricGuardError):
        log.error("metric guard refused a value: %s", exc)
        return respond({"status": "metric refused by guard", "detail": str(exc)}, 500)

    def S() -> Settings:
        return app.state.settings

    # ------------------------------------------------------------------ health
    @app.get("/health")
    def health():
        s = S()
        specs, plugin_errors = plugins.load_specs(s)
        _, cat_meta = D.site_catalog(s)
        reg = registry_view.status(s)
        return respond(
            {
                "status": "ok",
                "service": "SENTINEL Console API",
                "version": __version__,
                "time_utc": provenance.utc_now(),
                "data": D.data_status(s),
                "data_warnings": cat_meta["notes"],
                "registry": {
                    k: reg[k]
                    for k in (
                        "registry_path",
                        "exists",
                        "n_entries",
                        "n_current",
                        "n_test_non_baseline",
                        "has_validated_results",
                    )
                },
                "plugins": {"registered": [sp.public() for sp in specs], "errors": plugin_errors},
                "live": {
                    "enabled": s.live_enabled,
                    "api_key_configured": live.read_api_key(s.keys_file) is not None,
                    "rate_limited_for_s": app.state.live.blocked_for(),
                },
                "live_trial": {k: v for k, v in status.live_trial(s).items() if k != "latest"},
            }
        )

    # ------------------------------------------------------------------ sites
    @app.get("/sites")
    def sites(
        var: str | None = None,
        tier: str = Query("any", pattern="^(any|daily|hourly)$"),
        huc: str | None = Query(None, pattern=r"^\d{2,12}$"),
        state: str | None = None,
        site_type: str | None = None,
        bbox: str | None = None,
        q: str | None = None,
        has_data: str = Query("true", pattern="^(true|false|any)$"),
        limit: int = Query(1000, ge=1, le=10000),
        offset: int = Query(0, ge=0),
    ):
        s = S()
        df, meta = D.sites_table(s)
        sel = D.filter_sites(
            df,
            var=var,
            tier=tier,
            huc=huc,
            state=state,
            site_type=site_type,
            bbox=_bbox(bbox),
            q=q,
            has_data=has_data,
        )
        page = sel.iloc[offset : offset + limit]
        notes = list(meta["notes"])
        if df.empty:
            notes.append(
                "no sites: neither a site catalogue nor processed sonde tables were found under "
                f"{s.rel(s.data_root)}"
            )
        return respond(
            {
                "total": int(len(sel)),
                "count": int(len(page)),
                "offset": offset,
                "limit": limit,
                "filters": {
                    "var": var,
                    "tier": tier,
                    "huc": huc,
                    "state": state,
                    "site_type": site_type,
                    "bbox": bbox,
                    "q": q,
                    "has_data": has_data,
                },
                "sites": json.loads(page.to_json(orient="records")) if len(page) else [],
                "provenance": provenance.block(s, meta["datasets"], meta["files"], notes=notes),
            }
        )

    @app.get("/sites/{site_id}")
    def site(site_id: str):
        s = S()
        sid = D.normalize_site_id(site_id)
        rec = D.site_record(s, sid)
        site_row = rec["site"]
        hourly = None
        hf = D.hourly_files(s).get(sid)
        if hf is not None:
            hourly = D.hourly_info(hf)
        cols = list(site_row.get("daily_columns") or [])
        hz = [h for c in cols for h in hazard_refs.lines_for(c, "daily")]
        hz += [h for c in (hourly or {}).get("columns", []) for h in hazard_refs.lines_for(c, "hourly")]
        files = [p for p in (D.daily_sites_dir(s) / f"{sid}.parquet", hf) if p is not None and p.exists()]
        return respond(
            {
                "site": site_row,
                "daily_coverage": D.daily_coverage(s, sid),
                "hourly": hourly,
                "hazards": hz,
                "provenance": provenance.block(
                    s, rec["meta"]["datasets"], [*rec["meta"]["files"], *files], notes=rec["meta"]["notes"]
                ),
            }
        )

    @app.get("/sites/{site_id}/observations")
    def site_observations(
        site_id: str,
        var: list[str] | None = Query(None),
        start: str | None = None,
        end: str | None = None,
        tier: str = Query("daily", pattern="^(daily|hourly)$"),
        limit: int | None = Query(None, ge=1),
        offset: int = Query(0, ge=0),
    ):
        s = S()
        sid = D.normalize_site_id(site_id)
        lim = min(limit or s.default_observation_rows, s.max_observation_rows)
        obs = D.observations(s, sid, tier=tier, variables=_split_vars(var), start=start, end=end)
        page = obs.rows.iloc[offset : offset + lim]
        n_prov = int(obs.rows["provisional"].sum()) if len(obs.rows) else 0
        nxt = offset + lim if offset + lim < obs.total_rows else None
        dataset = "sonde_daily" if tier == "daily" else "sonde_hourly"
        return respond(
            {
                "site_id": sid,
                "tier": tier,
                "start": start,
                "end": end,
                "variables": obs.variables,
                "total_rows": obs.total_rows,
                "count": int(len(page)),
                "offset": offset,
                "limit": lim,
                "next_offset": nxt,
                "truncated": nxt is not None,
                "label": "observed values (USGS), as retrieved and quality-controlled by the "
                "SENTINEL builder",
                "rows": json.loads(page.to_json(orient="records")) if len(page) else [],
                "provenance": provenance.block(
                    s,
                    [dataset],
                    obs.files,
                    approval={
                        "n_values": obs.total_rows,
                        "n_provisional": n_prov,
                        "n_approved": obs.total_rows - n_prov,
                    },
                    notes=obs.notes,
                ),
            }
        )

    # ------------------------------------------------------------------ live
    @app.get("/live/{site_id}")
    def live_values(site_id: str, window_hours: int | None = Query(None, ge=1, le=live.MAX_HOURS)):
        s = S()
        if not s.live_enabled:
            return respond(
                {
                    "status": "live fetch disabled",
                    "data_status": live.DATA_STATUS,
                    "detail": "the live USGS fetch is switched off (SENTINEL_LIVE=0)",
                },
                503,
            )
        try:
            sid = D.normalize_site_id(site_id)
            out = app.state.live.fetch(sid, window_hours=window_hours)
        except live.LiveUnavailable as e:
            hdr = {"Retry-After": str(e.retry_after)} if e.retry_after else None
            return respond(
                {
                    "status": "live data unavailable",
                    "data_status": live.DATA_STATUS,
                    "detail": e.detail,
                    "retry_after_s": e.retry_after,
                },
                e.status_code,
                hdr,
            )
        return respond(out)

    # ------------------------------------------------------------------ models & registry
    @app.get("/models")
    def models():
        s = S()
        mc = model_cards.cards_with_metrics(s)
        specs, errors = plugins.load_specs(s)
        by_mod = {}
        for m in MODALITIES:
            sp, _ = plugins.spec_for(s, m)
            if sp is None:
                by_mod[m] = {"model_id": None, "status": "model not yet trained"}
            else:
                ok, why, _ = plugins.serving_gate(s, sp)
                by_mod[m] = {
                    "model_id": sp.model_id,
                    "status": "served" if ok else "not served",
                    "reason": why,
                }
        msg = None
        if not mc["cards"]:
            msg = (
                f"no model cards in {mc['model_cards_dir']} (template: README.md there); no v2 model has "
                "been trained yet"
            )
        return respond(
            {
                **mc,
                "message": msg,
                "registry": registry_view.status(s),
                "plugins": {
                    "registered": [sp.public() for sp in specs],
                    "errors": errors,
                    "by_modality": by_mod,
                },
            }
        )

    @app.get("/registry")
    def registry(
        workstream: str | None = None,
        experiment: str | None = None,
        model: str | None = None,
        split: str | None = None,
        metric: str | None = None,
        target: str | None = None,
        is_baseline: bool | None = None,
        include_config: bool = False,
    ):
        s = S()
        df = registry_view.filtered(
            s,
            workstream=workstream,
            experiment=experiment,
            model=model,
            split=split,
            metric=metric,
            target=target,
            is_baseline=is_baseline,
        )
        st = registry_view.status(s)
        return respond(
            {
                "filters": {
                    "workstream": workstream,
                    "experiment": experiment,
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "target": target,
                    "is_baseline": is_baseline,
                },
                "count": int(len(df)),
                "current_only": True,
                "registry": {k: st[k] for k in ("registry_path", "exists", "n_entries", "n_current")},
                "entries": registry_view.entries(s, df, include_config=include_config),
                "message": None
                if st["exists"]
                else "results/registry.jsonl does not exist yet: no results logged",
            }
        )

    @app.get("/hazards")
    def hazards():
        return respond(
            {
                "hazards": hazard_refs.hazard_table(),
                "source": "sentinel/eval/hazards.py",
                "note": hazard_refs.NOTE,
            }
        )

    @app.get("/provenance/{name}")
    def manifest(name: str):
        s = S()
        if not MANIFEST_NAME_RE.match(name):
            raise D.BadRequest("invalid manifest name")
        m = provenance.manifest_summary(s, name)
        if m["status"] == "missing":
            return respond({"status": "not found", **m}, 404)
        return respond(m)

    # ------------------------------------------------------------------ predict
    @app.post("/predict/{modality}")
    async def predict(modality: str, file: UploadFile = File(...), options: str | None = Form(None)):
        s = S()
        if modality not in MODALITIES:
            return respond(
                {"status": "unknown modality", "modality": modality, "modalities": list(MODALITIES)}, 404
            )
        limit = s.upload_limits[modality]
        content = await file.read(limit + 1)
        if len(content) > limit:
            return respond({"status": "upload too large", "modality": modality, "limit_bytes": limit}, 413)
        try:
            opts = json.loads(options) if options else {}
            if not isinstance(opts, dict) or len(options or "") > 4096:
                raise ValueError
        except ValueError:
            return respond(
                {
                    "status": "invalid upload",
                    "errors": [
                        {
                            "field": "options",
                            "message": "options must be a JSON object (at most 4096 characters)",
                        }
                    ],
                },
                422,
            )
        spec, plugin_errors = plugins.spec_for(s, modality)
        required = spec.required_inputs if spec else ()
        rep = await run_in_threadpool(
            validation.validate,
            modality,
            file.filename or "",
            content,
            options=opts,
            required_inputs=required,
        )
        if not rep.ok:
            return respond({"status": "invalid upload", **rep.public()}, 422)
        if spec is None:
            return respond(
                {
                    "status": "model not yet trained",
                    "modality": modality,
                    "detail": f"no model is registered for modality '{modality}' in {s.rel(s.models_yaml)}; "
                    "the upload passed validation but no prediction was made",
                    "plugin_errors": plugin_errors,
                    "validation": rep.public(),
                },
                503,
            )
        ok, why, metrics = await run_in_threadpool(plugins.serving_gate, s, spec)
        if not ok:
            return respond(
                {
                    "status": "model not validated",
                    "modality": modality,
                    "model_id": spec.model_id,
                    "detail": f"refusing to serve unvalidated predictions: {why}",
                    "validation": rep.public(),
                },
                503,
            )
        try:
            out = await run_in_threadpool(plugins.run, spec, rep.upload())
        except plugins.PluginError as e:
            log.error("plugin %s failed: %s", spec.model_id, e)
            return respond({"status": "model error", "model_id": spec.model_id, "detail": str(e)}, 500)
        return respond(
            {
                "status": "ok",
                "modality": modality,
                "model_id": spec.model_id,
                "label": "MODEL OUTPUT, not an observation",
                "predictions": out["predictions"],
                "units": out["units"],
                "model_version": out["model_version"],
                "extra": {k: v for k, v in out.items() if k not in ("predictions", "units", "model_version")},
                "validated_metrics": metrics,
                "validation": rep.public(),
                "provenance": {
                    "generated_utc": provenance.utc_now(),
                    "entry_point": spec.entry_point,
                    "model_card": spec.model_card,
                    "registry_experiment": spec.registry_experiment,
                    "input_sha256": rep.sha256,
                },
            }
        )

    # ------------------------------------------------------------------ not yet implemented
    @app.get("/alerts")
    def alerts():
        return not_implemented(
            "EW-v2 alert feed",
            "alerts need frozen EW-v2 thresholds, trained hazard models "
            "and the live forecast pipeline (RESEARCH_PLAN.md section 12); none exist yet",
        )

    @app.post("/simulate")
    def simulate():
        return not_implemented(
            "SENTINEL-Sim",
            "the scenario engine (RESEARCH_PLAN.md section 7.1) has not been built or validated yet",
        )

    @app.get("/forecast/{site_id}")
    def forecast(site_id: str):
        return not_implemented("forecasts", "no SENTINEL forecast model has been trained or frozen yet")

    @app.get("/causal/{site_id}")
    def causal(site_id: str):
        return not_implemented(
            "causal explorer", "no causal-discovery results exist yet (RESEARCH_PLAN.md section 7.2)"
        )

    @app.get("/downstream/{app_name}")
    def downstream(app_name: str):
        return not_implemented(
            "downstream applications", "no downstream application (D1-D8) has been trained or evaluated yet"
        )

    return app
