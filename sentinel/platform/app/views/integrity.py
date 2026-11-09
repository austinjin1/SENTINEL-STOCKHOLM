# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Integrity: the project's integrity rules (from RESEARCH_PLAN.md) and how the Console enforces them."""

from __future__ import annotations

import streamlit as st

from ... import status
from .. import ui

ENFORCEMENT = """
- **Metrics only from the registry.** Every metric shown passes through `sentinel/platform/guard.py`,
  which accepts only results-registry entry ids or registry rows, re-checks each row against
  `results/registry.jsonl`, and refuses raw numbers, unknown ids, superseded entries and edited rows.
- **No typed-in claims.** A test scans the platform source and its documentation for metric names
  followed by numbers, percentages, plus-minus intervals, event-count fractions and claim labels, and
  fails on any; it also fails if platform code calls a random-number generator.
- **Provenance on every data response.** Manifest names, creation dates, sources, licence, the
  `synthetic` flag and the files read are attached to every API response and shown in each page
  footer. A synthetic manifest turns on a red banner.
- **Provisional is visible.** USGS provisional values are drawn differently from approved values, and
  live values are labelled "provisional, live" and never used for any reported metric.
- **Consistent site metadata.** The processed site table is cross-checked against the raw USGS
  catalogue; where coordinates disagree, the raw USGS values are shown and the site is flagged.
- **Empty states say what is missing.** Missing datasets, manifests, model cards, models and
  live-trial files are reported as missing; nothing is filled in with sample data.
- **Unbuilt components refuse.** Upload-and-predict answers 503 "model not yet trained" for a modality
  without a registered, registry-validated model; alerts and simulation answer 501.
"""


def render() -> None:
    s = ui.settings()
    st.title("Integrity")
    sec = status.plan_section(s, 3)
    if sec:
        st.markdown(sec)
        st.caption(f"Rendered from {s.rel(s.research_plan)} at page load.")
    else:
        st.warning(f"Section 3 (integrity rules) not found in {s.rel(s.research_plan)}.")
    st.subheader("How this console enforces them")
    st.markdown(ENFORCEMENT)
    st.subheader("Pre-registration documents")
    docs = status.prereg_docs(s)
    if not docs:
        st.info(f"No pre-registration documents in {s.rel(s.prereg_dir)}.")
    for d in docs:
        with st.expander(f"{d['title']} — {d['status'] or 'status not stated'}"):
            st.caption(d["file"])
            st.markdown(d["text"])
