# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Time-series figure for the Site page.

One panel per variable (shared time axis, never two y-scales on one panel). Approved values are
a solid line in categorical slot 1; provisional values a dotted line in slot 2, so they differ by
colour *and* dash. Gaps in the record are left as gaps. EW-v2 hazard thresholds are drawn as
dashed neutral reference lines with a text label; the anchors are listed in the table under
the chart.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import ui

PANEL_HEIGHT = 250
DENSE = 4000  # switch to WebGL traces above this many points per series
SPARSE = 200  # draw markers below this many points (isolated values stay visible)


def _regular(times: pd.Series, tier: str) -> pd.DatetimeIndex:
    t = pd.to_datetime(times, utc=(tier == "hourly"))
    freq = "D" if tier == "daily" else "h"
    return pd.date_range(t.min(), t.max(), freq=freq)


def timeseries(
    rows: pd.DataFrame, variables: list[dict], tier: str, mode: str, hazards: dict[str, list[dict]]
) -> go.Figure:
    tcol = "date" if tier == "daily" else "time_utc"
    names = [v["name"] for v in variables if v["n"] > 0]
    fig = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[next(v["label"] for v in variables if v["name"] == n) for n in names],
    )
    blue, orange = ui.SERIES[mode][0], ui.SERIES[mode][1]
    shown = {"approved": False, "provisional": False}
    for i, name in enumerate(names, start=1):
        meta = next(v for v in variables if v["name"] == name)
        g = rows[rows["variable"] == name]
        idx = _regular(g[tcol], tier)
        t = pd.to_datetime(g[tcol], utc=(tier == "hourly"))
        val = pd.Series(g["value"].to_numpy(), index=t)
        prov = pd.Series(g["provisional"].to_numpy(bool), index=t)
        val, prov = val.reindex(idx), prov.reindex(idx, fill_value=False).astype(bool)
        for key, mask, colour, dash, label in (
            ("approved", ~prov, blue, "solid", "Approved"),
            ("provisional", prov, orange, "dot", "Provisional (subject to revision)"),
        ):
            y = val.where(mask)
            n = int(y.notna().sum())
            if n == 0:
                continue
            trace = go.Scattergl if n > DENSE else go.Scatter
            hover = f"%{{x}}<br>{label}: %{{y}} {meta['unit'] or ''}<extra>{name}</extra>"
            fig.add_trace(
                trace(
                    x=idx,
                    y=y.to_numpy(),
                    name=label,
                    legendgroup=key,
                    showlegend=not shown[key],
                    mode="lines+markers" if n < SPARSE else "lines",
                    connectgaps=False,
                    line={"color": colour, "width": 2, "dash": dash},
                    marker={"size": 6, "color": colour},
                    hovertemplate=hover,
                ),
                row=i,
                col=1,
            )
            shown[key] = True
        for hz in hazards.get(name, []):
            fig.add_hline(
                y=hz["threshold"],
                row=i,
                col=1,
                line={"color": ui.NEUTRAL[mode], "width": 1, "dash": "dash"},
                annotation_text=f"{hz['key']}: {hz['direction']} {hz['threshold']} {hz['unit'] or ''}",
                annotation_position="top left",
                annotation_font_size=11,
            )
        fig.update_yaxes(title_text=meta["unit"] or "", row=i, col=1)
    fig.update_layout(
        height=PANEL_HEIGHT * max(1, len(names)) + 60,
        hovermode="x unified",
        margin={"l": 60, "r": 20, "t": 40, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig
