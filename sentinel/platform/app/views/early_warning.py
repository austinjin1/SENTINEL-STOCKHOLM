# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Early warning: what EW-v2 is, the pre-registered hazards, and whether the live trial has started."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import hazard_refs, status
from .. import ui


def render() -> None:
    s = ui.settings()
    st.title("Early warning (EW-v2)")
    trial = status.live_trial(s)
    if trial["started"]:
        st.info(
            f"Live forecast files found under {trial['live_dir']}: {trial['n_files']} (most recent "
            f"{trial['latest'][0]['path']}, modified {trial['latest'][0]['modified_utc']}). This page "
            "shows no trial scores: results appear only as registry entries after the pre-specified "
            "analysis dates."
        )
        st.dataframe(pd.DataFrame(trial["latest"]), hide_index=True)
    else:
        st.warning(
            f"**The live prospective trial has not started.** No live-forecast file exists under "
            f"{trial['live_dir']}."
        )
    st.markdown(
        "EW-v2 replaces the retracted v1 early-warning claims. Instead of counting any alarm in a long "
        "look-back window as a detection, it issues daily probabilistic forecasts of hazard *onset*, "
        "scores them against climatology and persistence, freezes alarm thresholds on calibration data "
        "before the test years are touched, and requires placebo and surrogate controls. The live alert "
        "feed (GET /alerts) answers HTTP 501 until frozen models and thresholds exist."
    )
    docs = [d for d in status.prereg_docs(s) if "ew" in d["file"].lower()]
    for d in docs:
        st.markdown(
            f"**Pre-registration:** {d['title']} — status: {d['status'] or 'not stated'} ({d['file']})"
        )

    st.subheader("Pre-registered hazards (sentinel/eval/hazards.py)")
    rows = [
        {
            "hazard": h["key"],
            "variable": h["variable"],
            "event": f"{h['direction']} {h['threshold']} {h['unit'] or ''}",
            "onset rule": f"{h['pre_days']} clean days before (margin {h['pre_margin']}), "
            f"no event in previous {h['refractory_days']} days",
            "sub-daily duration": h["subdaily_min_duration"],
            "anchor": h["anchor"],
        }
        for h in hazard_refs.hazard_table()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Every hazard is a {hazard_refs.NOTE}. Hazards that need per-site derived series (chloride "
        "from conductance, turbidity against a trailing background) are not in this table yet."
    )

    sec = status.plan_section(s, 12)
    st.subheader("Protocol")
    if sec:
        with st.expander("RESEARCH_PLAN.md, section 12 (rendered from the file)", expanded=False):
            st.markdown(sec)
    else:
        st.info(f"Section 12 not found in {s.rel(s.research_plan)}.")
