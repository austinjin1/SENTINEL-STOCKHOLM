# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Site page: local daily/hourly observations, hazard reference lines, live values, provenance."""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import streamlit as st

from ... import data_access, hazard_refs, live, provenance
from ... import variables as V
from .. import charts, ui

DEFAULT_DAYS = {"daily": 3 * 365, "hourly": 30}
MAX_DAYS = {"daily": 40 * 366, "hourly": 2 * 366}
MAX_VARS = 6


def _default_column(cols: list[str], tier: str) -> str:
    stats = V.DAILY_STATS if tier == "daily" else V.HOURLY_STATS
    hz_cols = [c for c in cols if hazard_refs.lines_for(c, tier)]
    if hz_cols:
        return hz_cols[0]
    wq_means = [c for c in cols if c.endswith("_mean") and V.split_column(c, stats)[0] in V.WQ_NAMES]
    return (wq_means or cols)[0]


def _date(x: str | None, fallback: _dt.date) -> _dt.date:
    return pd.Timestamp(x).date() if x else fallback


def _live_panel(s, sid: str) -> None:
    with st.expander("Latest real-time values from USGS (provisional, live)"):
        st.caption(live.NOTICE)
        if not live.USGS_SITE_RE.match(sid):
            st.info("Live values are available only for USGS site numbers.")
            return
        if not st.button("Fetch latest values", key=f"live_{sid}"):
            return
        try:
            code, body = ui.api_request(s, "GET", f"/live/{sid}", timeout=45.0)
        except ui.ApiUnreachable as e:
            st.error(str(e))
            return
        if code != 200:
            st.error(f"{body.get('status', 'error')}: {body.get('detail', '')}")
            return
        st.warning(
            "**PROVISIONAL, LIVE** — fetched from USGS at request time; not quality-assured; never used "
            "for any reported metric."
        )
        vals = pd.DataFrame(body.get("values") or [])
        if vals.empty:
            st.info("USGS returned no continuous values for this site.")
        else:
            st.dataframe(
                vals[
                    [
                        "parameter_code",
                        "variable",
                        "time_utc",
                        "value",
                        "unit",
                        "approval_status",
                        "qualifier",
                        "statistic_id",
                    ]
                ],
                hide_index=True,
            )
        p = body.get("provenance") or {}
        st.caption(
            f"{p.get('source')} · collection {p.get('collection')} · retrieved {p.get('retrieved_utc')} · "
            f"{'authenticated' if p.get('authenticated') else 'anonymous'} request · cache {p.get('cache')}"
        )


def render() -> None:
    s = ui.settings()
    st.title("Site")
    df, _ = data_access.sites_table(s)
    with_data = df[df["has_data"]] if not df.empty else df
    if with_data.empty:
        st.info(
            "No local observations to show: the daily tier (processed/sonde_daily) and the hourly tier "
            f"(processed/sonde_hourly) are missing or empty under {s.rel(s.data_root)}."
        )
        return
    ids = with_data["id"].tolist()
    wanted = st.session_state.pop("site_id", None) or st.query_params.get("site")
    if wanted in ids:
        st.session_state["site_select"] = wanted
    names = dict(zip(with_data["id"], with_data["name"], strict=True))
    sid = st.selectbox(
        "Site", ids, key="site_select", format_func=lambda i: f"{i} — {names.get(i) or 'name unknown'}"
    )
    st.query_params["site"] = sid
    rec = with_data[with_data["id"] == sid].iloc[0]

    st.subheader(rec["name"] or sid)
    facts = [
        f"{rec['agency'] or 'agency unknown'} {sid}",
        rec["site_type_label"] or "type unknown",
        f"HUC {rec['huc'] or 'unknown'}",
        rec["state"] or "state unknown",
    ]
    if pd.notna(rec["lat"]):
        facts.append(f"{rec['lat']:.5f}, {rec['lon']:.5f}")
    st.markdown(" · ".join(str(f) for f in facts))
    st.caption(
        f"Local record {rec['record_start']} to {rec['record_end']} · variables: "
        f"{', '.join(V.base_label(v) for v in rec['variables'])}"
    )
    if rec.get("catalog_conflict"):
        st.warning(
            "processed/sites/sites.parquet disagrees with the raw USGS catalogue for this site; the raw "
            "USGS metadata are shown."
        )

    tiers = [t for t in ("daily", "hourly") if len(rec[f"{t}_columns"])]
    tier = st.radio(
        "Tier",
        tiers,
        horizontal=True,
        format_func=lambda t: (
            "Daily (USGS daily values)" if t == "daily" else "Hourly (aggregated 15-min values, UTC)"
        ),
    )
    cols = list(rec[f"{tier}_columns"])
    describe = V.describe if tier == "daily" else V.describe_hourly
    sel = st.multiselect(
        "Variables",
        cols,
        default=[_default_column(cols, tier)],
        max_selections=MAX_VARS,
        format_func=lambda c: f"{describe(c)['label']} ({describe(c)['unit'] or 'unit unknown'})",
    )
    if not sel:
        st.info("Select at least one variable.")
        return
    lo = _date(rec[f"{tier}_start"], _dt.date(2000, 1, 1))
    hi = _date(rec[f"{tier}_end"], _dt.date.today())
    sel_hi = hi
    if tier == "daily":  # default window ends where the selected variables' own records end
        cov = {c["name"]: c for c in data_access.daily_coverage(s, sid)}
        lasts = [cov[c]["last"] for c in sel if c in cov and cov[c]["last"]]
        if lasts:
            sel_hi = min(hi, _date(max(lasts), hi))
    start_default = max(lo, sel_hi - _dt.timedelta(days=DEFAULT_DAYS[tier]))
    rng = st.date_input(
        "Period",
        value=(start_default, sel_hi),
        min_value=lo,
        max_value=hi,
        key=f"period_{sid}_{tier}_{'-'.join(sel)}",
    )
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        st.info("Pick a start and an end date.")
        return
    start, end = rng
    if (end - start).days > MAX_DAYS[tier]:
        start = end - _dt.timedelta(days=MAX_DAYS[tier])
        st.info(f"The {tier} view is limited to {MAX_DAYS[tier]} days per request; showing from {start}.")

    obs = data_access.observations(s, sid, tier=tier, variables=sel, start=str(start), end=str(end))
    n_prov = int(obs.rows["provisional"].sum()) if len(obs.rows) else 0
    dataset = "sonde_daily" if tier == "daily" else "sonde_hourly"
    block = provenance.block(
        s,
        [dataset],
        obs.files,
        notes=obs.notes,
        approval={"n_values": obs.total_rows, "n_provisional": n_prov, "n_approved": obs.total_rows - n_prov},
    )
    ui.synthetic_banner(block)
    hz = {c: hazard_refs.lines_for(c, tier) for c in sel}
    if obs.total_rows == 0:
        st.info("No values for the selected variables in this period.")
    else:
        fig = charts.timeseries(obs.rows, obs.variables, tier, ui.theme(), hz)
        st.plotly_chart(fig, key="ts")
        st.caption(
            "Solid line: approved values. Dotted line: provisional values (subject to revision by USGS). "
            "Dashed reference lines: EW-v2 hazard thresholds from sentinel/eval/hazards.py "
            f"({hazard_refs.NOTE}). Gaps are missing data."
        )
    rows = [h for c in sel for h in hz[c]]
    if rows:
        st.markdown("**Hazard thresholds drawn**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "hazard": h["key"],
                        "applies to": h["applies_to"],
                        "event": f"{h['direction']} {h['threshold']} {h['unit'] or ''}",
                        "rule": h["rule"],
                        "regulatory anchor": h["anchor"],
                    }
                    for h in rows
                ]
            ),
            hide_index=True,
        )
        st.caption(hazard_refs.NOTE.capitalize() + ".")
    with st.expander("Table view"):
        st.dataframe(obs.rows, hide_index=True)
    _live_panel(s, sid)
    ui.provenance_footer(block)
