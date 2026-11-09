# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Bring your own data: upload a file for any modality; the API validates it and predicts if it can."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ... import validation
from ...config import MODALITIES
from .. import ui

LABELS = {
    "sonde": "Sonde (in-situ sensor time series)",
    "orbit": "Orbit (satellite chip)",
    "lite": "Lite (drone RGB+NIR image)",
    "microbe": "Microbe (ASV / 16S table)",
    "chemtox": "ChemTox (SMILES / CAS list)",
    "bioassess": "BioAssess (taxa table)",
}


def _issues(title: str, items: list[dict]) -> None:
    if items:
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(items), hide_index=True)


def render() -> None:
    s = ui.settings()
    st.title("Bring your own data")
    st.caption(
        "Files are sent to the SENTINEL API, validated (format, schema, units, size) and passed to the "
        "registered model for that modality, if one exists. Uploads are not stored."
    )
    modality = st.selectbox("Modality", MODALITIES, format_func=lambda m: LABELS[m])
    st.markdown(validation.FORMAT_HELP[modality])
    limit_mib = s.upload_limits[modality] / 2**20
    exts = validation.EXTENSIONS[modality]
    st.caption(f"Accepted: {', '.join(exts)} · size limit {limit_mib:g} MiB")
    opts: dict = {}
    if modality == "lite":
        opts["band_order"] = st.text_input("Band order in the file", value="R,G,B,N")
    f = st.file_uploader("File", type=[e.lstrip(".") for e in exts], key=f"upload_{modality}")
    if f is None or not st.button("Validate and predict", type="primary"):
        return
    try:
        code, body = ui.api_request(
            s,
            "POST",
            f"/predict/{modality}",
            timeout=300.0,
            files={"file": (f.name, f.getvalue())},
            data={"options": json.dumps(opts)} if opts else None,
        )
    except ui.ApiUnreachable as e:
        st.error(str(e))
        return
    val = body.get("validation") or (body if "errors" in body else {})
    if code == 422:
        st.error("The upload failed validation; nothing was sent to a model.")
        _issues("Errors", body.get("errors") or [])
        _issues("Warnings", body.get("warnings") or [])
        return
    if code == 413:
        st.error(f"File too large for {modality} (limit {body.get('limit_bytes')} bytes).")
        return
    if code == 503:
        st.warning(f"**{body.get('status')}** — {body.get('detail', '')}")
        if val.get("ok"):
            st.success("The file passed validation.")
        _issues("Warnings", val.get("warnings") or [])
        if val.get("summary"):
            st.json(val["summary"], expanded=False)
        return
    if code != 200:
        st.error(f"{body.get('status', 'error')} (HTTP {code}): {body.get('detail', '')}")
        return
    st.success(f"Predictions from {body['model_id']} ({body.get('model_version')}).")
    st.warning(f"**{body['label']}.**")
    st.dataframe(pd.DataFrame(body.get("predictions") or []), hide_index=True)
    st.markdown("**Units:** " + ", ".join(f"{k}: {v}" for k, v in (body.get("units") or {}).items()))
    st.markdown("**Validated metrics for this model (from the results registry)**")
    ui.show_metric_table(
        s,
        [m["id"] for m in body.get("validated_metrics") or []],
        empty="The registry returned no metrics for this model.",
    )
    _issues("Validation warnings", val.get("warnings") or [])
    st.caption(
        f"Input sha256 {body['provenance']['input_sha256']} · registry experiment "
        f"{body['provenance']['registry_experiment']} · model card {body['provenance']['model_card']}"
    )
