# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared pieces of the Streamlit app: settings, banners, provenance footer, API client, metric
tables (always through :mod:`sentinel.platform.guard`) and the chart palette.

Palette: categorical slots 1-3 of the SENTINEL dataviz reference palette, validated for both
light and dark surfaces (all-pairs CVD separation); neutral ink for reference lines.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .. import guard, registry_view
from ..config import Settings

SERIES = {"light": ("#2a78d6", "#eb6834", "#1baf7a"), "dark": ("#3987e5", "#d95926", "#199e70")}
NEUTRAL = {"light": "#52514e", "dark": "#c3c2b7"}
MUTED = "#898781"
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}


class ApiUnreachable(RuntimeError):
    pass


def settings() -> Settings:
    return Settings.from_env()


def theme() -> str:
    try:
        t = getattr(st.context.theme, "type", None)
    except Exception:  # noqa: BLE001 — older runtimes / test harness
        t = None
    return "dark" if t == "dark" else "light"


def hex_rgb(h: str, alpha: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


def validated_banner(s: Settings) -> dict:
    st_ = registry_view.status(s)
    if not st_["has_validated_results"]:
        st.warning(
            "**No validated v2 results yet.** The results registry "
            f"({st_['registry_path']}) has no non-baseline entries on a test split. Nothing in this "
            "console is a validated SENTINEL v2 result; observations are shown as retrieved, with their "
            "provenance."
        )
    return st_


def synthetic_banner(block: dict | None) -> None:
    if block and block.get("synthetic"):
        st.error(
            "**SYNTHETIC DATA.** A manifest behind this view is flagged synthetic=True; these values are "
            "not observations."
        )


def _manifest_line(m: dict) -> str:
    if m.get("status") != "present":
        name = m.get("name") or m.get("dataset")
        return f"- `{name}`: **{m.get('status')}** — {m.get('note', '')}"
    parts = [
        f"- `{m['name']}`: created {m.get('created_utc')}",
        f"licence: {m.get('license')}",
        f"synthetic: {m.get('synthetic')}",
    ]
    if m.get("retrieval"):
        parts.append("retrieval: " + ", ".join(f"{k} {v}" for k, v in m["retrieval"].items()))
    if m.get("git_commit"):
        parts.append(f"code {str(m['git_commit'])[:10]}{' (dirty)' if m.get('git_dirty') else ''}")
    line = "; ".join(parts)
    for up in m.get("upstream") or []:
        line += "\n  " + _manifest_line(up)
    return line


def provenance_footer(block: dict | None, *, extra: list[str] | None = None) -> None:
    if not block:
        return
    st.divider()
    names = ", ".join(
        f"{m.get('name') or m.get('dataset')} ({m.get('status')})" for m in block.get("manifests", [])
    )
    st.caption(f"Provenance — manifests: {names or 'none'}; generated {block.get('generated_utc')}")
    with st.expander("Provenance details"):
        lines = [_manifest_line(m) for m in block.get("manifests", [])]
        if lines:
            st.markdown("**Manifests**\n" + "\n".join(lines))
        if block.get("approval"):
            a = block["approval"]
            st.markdown(
                f"**Approval status of the values shown:** {a.get('n_approved')} approved, "
                f"{a.get('n_provisional')} provisional — {a.get('rule')}"
            )
        files = block.get("files") or []
        if files:
            st.markdown(
                "**Files read**\n"
                + "\n".join(
                    f"- `{f['path']}`"
                    + (
                        f" (modified {f['modified_utc']}, {f['bytes']} bytes)"
                        if f.get("exists")
                        else " (missing)"
                    )
                    for f in files
                )
            )
        for n in [*(block.get("notes") or []), *(extra or [])]:
            st.markdown(f"- {n}")


# ------------------------------------------------------------------ API client


def api_request(s: Settings, method: str, path: str, *, timeout: float = 60.0, **kw) -> tuple[int, Any]:
    import requests

    url = f"{s.api_url}{path}"
    try:
        r = requests.request(method, url, timeout=timeout, **kw)
    except requests.RequestException as e:
        raise ApiUnreachable(
            f"the SENTINEL API is not reachable at {s.api_url} ({e.__class__.__name__}); "
            "start it with `python -m sentinel.platform.api`"
        ) from e
    try:
        body = r.json()
    except ValueError:
        body = {"status": "non-JSON response", "detail": r.text[:500]}
    return r.status_code, body


def api_alive(s: Settings) -> bool:
    try:
        code, _ = api_request(s, "GET", "/health", timeout=3.0)
    except ApiUnreachable:
        return False
    return code == 200


# ------------------------------------------------------------------ metrics (guarded)


def metric_frame(s: Settings, refs: list[Any]) -> pd.DataFrame:
    """Table of metrics; every row is re-resolved against the registry by the guard."""
    rows = []
    for ref in refs:
        g = guard.guarded_metric(ref, registry_path=s.registry_path)
        rows.append(
            {
                "metric": g.metric,
                "value": g.value,
                "ci_low": g.ci_low,
                "ci_high": g.ci_high,
                "ci_method": g.ci_method,
                "split": g.split,
                "target": g.target,
                "n": g.n,
                "baseline": g.is_baseline,
                "experiment": g.experiment,
                "model": g.model,
                "registry_id": g.id,
                "logged_utc": g.timestamp,
            }
        )
    return pd.DataFrame(rows)


def show_metric_table(s: Settings, refs: list[Any], *, empty: str) -> None:
    if not refs:
        st.info(empty)
        return
    df = metric_frame(s, refs)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "value": st.column_config.NumberColumn("value", format="%.4g"),
            "ci_low": st.column_config.NumberColumn("CI low", format="%.4g"),
            "ci_high": st.column_config.NumberColumn("CI high", format="%.4g"),
        },
    )
    st.caption(
        "Values are read from results/registry.jsonl and verified by the metric guard; "
        "a missing CI means none was recorded."
    )
