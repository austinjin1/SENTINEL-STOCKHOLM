# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Map: sites coloured by local data availability for a chosen variable; click opens the site page."""

from __future__ import annotations

import html

import pandas as pd
import pydeck as pdk
import streamlit as st

from ... import data_access, provenance
from ... import variables as V
from .. import ui
from . import PAGES

ANY = "__any__"
CATS = ("daily + hourly", "daily only", "hourly only")
NONE = "no local data"


def _category(row: pd.Series, var: str) -> str:
    if var == ANY:
        d = any(v in V.WQ_NAMES for v in row["daily_variables"])
        h = any(v in V.WQ_NAMES for v in row["hourly_variables"])
    else:
        d, h = var in row["daily_variables"], var in row["hourly_variables"]
    if d and h:
        return CATS[0]
    if d:
        return CATS[1]
    if h:
        return CATS[2]
    return NONE


def _selected_id(event) -> str | None:
    sel = getattr(event, "selection", None)
    if sel is None and isinstance(event, dict):
        sel = event.get("selection")
    if not sel:
        return None
    objs = sel.get("objects", {}) if isinstance(sel, dict) else getattr(sel, "objects", {})
    hits = (objs or {}).get("sites") or []
    return hits[0].get("id") if hits else None


def render() -> None:
    s = ui.settings()
    st.title("Map")
    df, meta = data_access.sites_table(s)
    block = provenance.block(s, meta["datasets"], meta["files"], notes=meta["notes"])
    ui.synthetic_banner(block)
    located = df.dropna(subset=["lat", "lon"]) if not df.empty else df
    if located.empty:
        st.info(
            "No site coordinates available: neither processed/sites/sites.parquet nor "
            f"raw/usgs_dv/dv_series_catalog.parquet exists (or has coordinates) under {s.rel(s.data_root)}."
        )
        ui.provenance_footer(block)
        return

    present = sorted({v for vs in df["variables"] for v in vs}, key=lambda v: (v not in V.WQ_NAMES, v))
    c1, c2 = st.columns([3, 2])
    var = c1.selectbox(
        "Colour sites by local data for",
        [ANY, *present],
        format_func=lambda v: "any water-quality variable" if v == ANY else V.base_label(v),
    )
    show_all = c2.toggle("Include catalogued sites without local data", value=False)
    plot = located if show_all else located[located["has_data"]]
    if plot.empty:
        st.info(
            "No catalogued site has local observations yet (processed/sonde_daily and "
            "processed/sonde_hourly are empty or missing)."
        )
        ui.provenance_footer(block)
        return
    plot = plot.copy()
    plot["category"] = plot.apply(_category, axis=1, var=var)
    mode = ui.theme()
    colours = {c: ui.hex_rgb(h) for c, h in zip(CATS, ui.SERIES[mode], strict=True)}
    colours[NONE] = ui.hex_rgb(ui.MUTED, 150)
    plot["color"] = plot["category"].map(colours)
    plot["radius"] = plot["category"].map(lambda c: 3 if c == NONE else 5)
    plot["type_label"] = plot["site_type_label"].fillna("type unknown")
    plot["vars_label"] = plot["variables"].map(lambda vs: ", ".join(vs) if vs else "no local data")
    plot["span"] = [
        f"{a} to {b}" if a else "no local record"
        for a, b in zip(plot["record_start"], plot["record_end"], strict=True)
    ]
    plot["huc_label"] = plot["huc"].fillna("unknown")
    cols = [
        "id",
        "name",
        "lat",
        "lon",
        "color",
        "radius",
        "category",
        "type_label",
        "vars_label",
        "span",
        "huc_label",
    ]
    layer_data = plot[cols].copy()
    for c in ("id", "name", "category", "type_label", "vars_label", "span", "huc_label"):  # tooltip is HTML
        layer_data[c] = layer_data[c].fillna("").astype(str).map(html.escape)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=layer_data,
        id="sites",
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        radius_units="pixels",
        pickable=True,
        stroked=True,
        get_line_color=ui.hex_rgb(ui.SURFACE[mode]),
        line_width_min_pixels=1,
    )
    view = pdk.ViewState(latitude=float(plot["lat"].median()), longitude=float(plot["lon"].median()), zoom=3)
    tooltip = {
        "html": "<b>{name}</b><br/>{id} · {type_label} · HUC {huc_label}<br/>{category}: {vars_label}"
        "<br/>{span}"
    }
    event = st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip),
        on_select="rerun",
        selection_mode="single-object",
        key="site_map",
    )

    counts = plot["category"].value_counts()
    legend = " &nbsp; ".join(
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:5px;"
        f"background:{h};margin-right:4px'></span>{c} ({int(counts.get(c, 0))})"
        for c, h in [*zip(CATS, ui.SERIES[mode], strict=True), (NONE, ui.MUTED)]
        if c != NONE or show_all
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.caption(
        "Colour = which local tier holds the selected variable (daily tier: USGS daily values; hourly "
        "tier: aggregated 15-min values). Click a site to open its page."
    )

    sid = _selected_id(event)
    if sid:
        st.session_state["site_id"] = sid
        if "site" in PAGES:
            st.switch_page(PAGES["site"])

    with st.expander("Table view"):
        st.dataframe(
            plot[["id", "name", "category", "type_label", "huc_label", "vars_label", "span"]], hide_index=True
        )
        pick = st.selectbox("Open site", [""] + plot.loc[plot["category"] != NONE, "id"].tolist())
        if pick and "site" in PAGES:
            st.session_state["site_id"] = pick
            st.switch_page(PAGES["site"])
    ui.provenance_footer(block)
