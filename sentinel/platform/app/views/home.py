# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Home: project status (RESULTS.md headline), registry counts, local data, components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import data_access, live, plugins, registry_view, status
from ...config import MODALITIES
from .. import ui


def render() -> None:
    s = ui.settings()
    st.title("SENTINEL Console")
    st.caption(
        "SENTINEL v2 — honest, multimodal deep learning for freshwater water quality (rebuild in progress)"
    )

    st.subheader("Project status")
    head = status.results_headline(s)
    if head["exists"]:
        for k in status.HEADLINE_KEYS:
            st.markdown(f"**{k}:** {head.get(k, '(line not found)')}")
        st.caption(f"Quoted verbatim from {head['path']}.")
    else:
        st.info(f"RESULTS.md not found at {head['path']}; no project status to show.")

    st.subheader("Results registry")
    reg = registry_view.status(s)
    if not reg["exists"]:
        st.info(f"{reg['registry_path']} does not exist yet: no result of any kind has been logged.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Entries (all, incl. superseded)", reg["n_entries"])
        c2.metric("Current entries", reg["n_current"])
        c3.metric("Non-baseline test-split entries", reg["n_test_non_baseline"])
        if reg["by_workstream"]:
            st.dataframe(pd.DataFrame(reg["by_workstream"]), hide_index=True)
        st.caption(
            "Counts are computed from the registry on each page load. A test split is a split "
            "name with a component starting with 'test'."
        )

    st.subheader("Local data")
    ds = data_access.data_status(s)
    _, meta = data_access.sites_table(s)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sites with daily data", ds["daily_sites"])
    c2.metric("Sites with hourly data", ds["hourly_sites"])
    c3.metric("Site catalogue", "present" if (ds["sites_table"] or ds["usgs_dv_catalog"]) else "missing")
    for n in meta["notes"]:
        st.warning(n)
    st.caption(f"Data root: {ds['data_root']}")

    st.subheader("Components")
    specs, errs = plugins.load_specs(s)
    served = {sp.modality for sp in specs if sp.enabled}
    trial = status.live_trial(s)
    rows = [
        {"component": "API", "status": "reachable" if ui.api_alive(s) else f"not reachable at {s.api_url}"},
        {
            "component": "Live USGS fetch",
            "status": ("enabled" if s.live_enabled else "disabled")
            + "; API key "
            + ("configured" if live.read_api_key(s.keys_file) else "not configured (anonymous rate limit)"),
        },
        {
            "component": "Model plugins",
            "status": ", ".join(f"{m}: {'registered' if m in served else 'no model'}" for m in MODALITIES),
        },
        {
            "component": "Live prospective trial",
            "status": f"{trial['n_files']} forecast file(s) found"
            if trial["started"]
            else f"not started (no forecast file under {trial['live_dir']})",
        },
        {"component": "Alerts (EW-v2)", "status": "not implemented (GET /alerts returns 501)"},
        {"component": "Scenario simulation", "status": "not implemented (POST /simulate returns 501)"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for e in errs:
        st.error(f"Plugin registry: {e}")
