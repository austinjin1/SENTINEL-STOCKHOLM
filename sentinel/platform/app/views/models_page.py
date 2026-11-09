# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Models: plugin status per modality, model cards, and registry metrics (guarded, with CIs)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import model_cards, plugins
from ...config import MODALITIES
from .. import ui

CARD_FIELDS = (
    "model_id",
    "modality",
    "workstream",
    "version",
    "status",
    "license",
    "registry_models",
    "registry_experiments",
    "intended_use",
    "limitations",
)


def render() -> None:
    s = ui.settings()
    st.title("Models")

    st.subheader("Upload-and-predict plugins")
    specs, errors = plugins.load_specs(s)
    rows = []
    for m in MODALITIES:
        sp, _ = plugins.spec_for(s, m)
        if sp is None:
            rows.append(
                {
                    "modality": m,
                    "model": "none registered",
                    "status": "POST /predict returns 503 'model not yet trained'",
                }
            )
        else:
            ok, why, _ = plugins.serving_gate(s, sp)
            rows.append(
                {"modality": m, "model": sp.model_id, "status": "served" if ok else f"not served: {why}"}
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Plugin registry: {s.rel(s.models_yaml)} ({len(specs)} entr{'y' if len(specs) == 1 else 'ies'})."
    )
    for e in errors:
        st.error(f"Plugin registry: {e}")

    st.subheader("Model cards")
    mc = model_cards.cards_with_metrics(s)
    if not mc["cards"]:
        st.info(
            f"No model cards in {mc['model_cards_dir']} yet (the template is README.md there). No "
            "SENTINEL v2 model has been trained, so there is nothing to describe."
        )
    for card in mc["cards"]:
        with st.container(border=True):
            st.markdown(f"### {card.get('name') or card['file']}")
            if card.get("error"):
                st.error(f"{card['file']}: {card['error']}")
            facts = {k: card.get(k) for k in CARD_FIELDS if card.get(k) not in (None, "", [])}
            if facts:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"field": k, "value": ", ".join(v) if isinstance(v, list) else str(v)}
                            for k, v in facts.items()
                        ]
                    ),
                    hide_index=True,
                )
            ui.show_metric_table(
                s,
                card.get("registry_entries") or [],
                empty="No current registry entries for this card's registry_models.",
            )
            if card.get("body"):
                with st.expander("Card text"):
                    st.markdown(card["body"])

    st.subheader("Registry entries without a model card")
    unc = mc["registry_models_without_card"]
    if not unc:
        st.caption("None.")
    for u in unc:
        with st.expander(f"{u['model']} ({len(u['registry_entries'])} current entries)"):
            ui.show_metric_table(s, u["registry_entries"], empty="No entries.")
