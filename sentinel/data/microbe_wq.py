# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Join NEON surface-water 16S communities with co-located water quality (SENTINEL-Microbe, M3).

Inputs (processed by the NEON data workstream, each with a manifest): ``neon_microbe_16s``
(samples / counts / otus [+ ``_v1``]), ``neon_chem`` (``chem_event_wide``, ``chem_long``),
``neon_sonde_daily`` (``sites/<SITE>.parquet``), ``neon_sites``.

Rules (docs/preregistration/microbe_protocol.md §2.2, fixed before targets were inspected):

* **Community event** = 16S dnaSampleIDs of one site with the same local-standard-time collection
  date (one release; the event's primer set is that of its first dnaSampleID — mixed-primer events
  are counted in the audit); counts summed over replicates; OTU/ASV names mapped to the sha1 of
  their representative sequence (``seq_id``) so identical sequences coincide; events with fewer
  than ``min_reads`` reads dropped; provisional records dropped.
* **Grab chemistry:** the chemistry event with the *same parentSampleID* as any of the event's
  genetic samples (same field visit; closest in time if several); otherwise the nearest chemistry
  event at the same site AND the same named location within ±24 h (``chem_match = 'time'``);
  otherwise missing. Censored values: the published value,
  else the detection limit (substitution recorded in ``<target>_censored``).
* **Sonde:** daily mean on the local collection date at the co-located station — wadeable
  streams HOR 102 (S2, downstream), lakes and non-wadeable rivers HOR 103 (buoy) — lowest VER
  (shallowest) when several; ±1-day mean kept as the ablation column ``<var>_pm1d``. Only events
  whose 16S named location is the S2 station or the buoy get sonde targets (reach / inflow /
  littoral events: sonde targets missing).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

PRIMER_SHORT = {"16S_V4_515F-806R": "V4", "16S_V3V4_Pro341F-Pro805R": "V3V4", "16S_V4V5_515FY-926R": "V4V5"}
CHEM_ANALYTES = {"raw_tn_mgl": "TN", "raw_tp_mgl": "TP", "raw_doc_mgl": "DOC", "raw_no3_mgl": "NO3+NO2 - N"}
SONDE_VARS = {"raw_sc_uscm": "spcond_uscm", "raw_turb_fnu": "turb_fnu", "raw_do_mgl": "do_mgl",
              "raw_chla_ugl": "chl_ugl"}
SONDE_HOR = {"stream": "102", "lake": "103", "river": "103"}
RANKS = ("domain", "phylum", "class", "order", "family", "genus")


def weighted_median(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return np.nan
    o = np.argsort(x[ok])
    cw = np.cumsum(w[ok][o])
    return float(x[ok][o][np.searchsorted(cw, 0.5 * cw[-1])])


def primer_from_length(mean_len: float) -> str:
    """Primer set from the read-weighted MEDIAN amplicon length after primer trimming (robust to
    long chloroplast reads, review W3): V4 (515F-806R)
    ~ 253 bp, V4V5 (515F-Y/926R) ~ 372 bp, V3V4 (Pro341F/Pro805R) ~ 400 bp on NEON data."""
    if not np.isfinite(mean_len):
        return "unknown"
    if mean_len < 300:
        return "V4"
    if mean_len < 386:
        return "V4V5"
    if mean_len < 440:
        return "V3V4"
    return "unknown"


def seq_sha1(seq: str) -> str:
    return hashlib.sha1(str(seq).upper().encode()).hexdigest()


def local_date(time_utc: pd.Series, offset_h: pd.Series) -> pd.Series:
    t = pd.to_datetime(time_utc, utc=True).dt.tz_convert(None)
    return (t + pd.to_timedelta(offset_h.astype(float), unit="h")).dt.normalize()


def build_events(samples: pd.DataFrame, counts: pd.DataFrame, otus: pd.DataFrame, sites: pd.DataFrame,
                 offsets: dict, *, min_reads: int = 1000, source: str = "DP1.20141.002"):
    """-> (events, tokens, sequences, audit)."""
    audit = {}
    s = samples.copy()
    audit["n_dna_samples"] = len(s)
    if "provisional" in s:
        s = s[~s["provisional"].fillna(False).astype(bool)]
    audit["n_after_provisional"] = len(s)
    s["primer_set_full"] = s["primer_set"].astype(str)
    s["primer_set"] = s["primer_set_full"].map(PRIMER_SHORT).fillna("unknown")
    s["primer_source"] = np.where(s["primer_set"] != "unknown", "pcr_record", "none")
    s["offset_h"] = s["site"].map(offsets)
    if s["offset_h"].isna().any():
        raise ValueError(f"missing UTC offsets for {sorted(s.loc[s['offset_h'].isna(), 'site'].unique())}")
    s["collect_date"] = local_date(s["time_utc"], s["offset_h"])
    s["event_key"] = s["site"].astype(str) + "|" + s["collect_date"].dt.strftime("%Y-%m-%d")
    # counts -> seq ids
    o = otus.dropna(subset=["sequence"]).copy()
    o["seq_id"] = [seq_sha1(x) for x in o["sequence"]]
    c = counts[counts["dnaSampleID"].isin(s["dnaSampleID"])].merge(o[["sequenceName", "seq_id"]],
                                                                    on="sequenceName", how="inner")
    audit["count_rows_without_sequence"] = int(counts["dnaSampleID"].isin(s["dnaSampleID"]).sum() - len(c))
    # primer fallback from the amplicon itself (read-weighted mean length), never from the date
    o_len = o.set_index("sequenceName")["sequence"].str.len()
    cl = c.assign(L=c["sequenceName"].map(o_len).to_numpy(float))
    wlen = cl.groupby("dnaSampleID")[["L", "count"]].apply(lambda g: weighted_median(g["L"], g["count"]))
    s["amplicon_len"] = s["dnaSampleID"].map(wlen)
    unk = (s["primer_set"] == "unknown") & s["amplicon_len"].notna()
    s.loc[unk, "primer_set"] = s.loc[unk, "amplicon_len"].map(primer_from_length)
    s.loc[unk, "primer_source"] = "amplicon_length"
    audit["primer_source"] = s["primer_source"].value_counts().to_dict()
    audit["mixed_primer_events"] = int((s.groupby("event_key")["primer_set"].nunique() > 1).sum())
    pcr = s[s["primer_source"] == "pcr_record"]
    if len(pcr) and pcr["amplicon_len"].notna().any():
        agree = (pcr["amplicon_len"].map(primer_from_length) == pcr["primer_set"])[pcr["amplicon_len"].notna()]
        audit["length_rule_agreement_with_pcr_records"] = {"n": int(agree.size), "agree": float(agree.mean())}
    c = c.merge(s[["dnaSampleID", "event_key"]], on="dnaSampleID")
    tok = c.groupby(["event_key", "seq_id"], as_index=False)["count"].sum()
    tok = tok[tok["count"] > 0]
    reads = tok.groupby("event_key")["count"].sum()
    nseq = tok.groupby("event_key").size()
    g = s.groupby("event_key")
    ev = pd.DataFrame({
        "site_id": g["site"].first(), "domain_id": g["domain"].first(), "collect_date": g["collect_date"].first(),
        "time_utc": g["time_utc"].min(), "primer_set": g["primer_set"].first(),
        "primer_set_full": g["primer_set_full"].first(), "primer_source": g["primer_source"].first(),
        "amplicon_len": g["amplicon_len"].mean(),
        "named_location": g["named_location"].first() if "named_location" in s else pd.NA,
        "parent_sample_ids": g["parentSampleID"].agg(lambda x: "|".join(sorted(set(map(str, x.dropna()))))),
        "dna_sample_ids": g["dnaSampleID"].agg(lambda x: "|".join(sorted(set(map(str, x))))),
        "n_dna_samples": g.size(),
        "release": g["release"].first() if "release" in s else pd.NA,
        "instrument_model": g["instrument_model"].first() if "instrument_model" in s else pd.NA,
    })
    ev["n_reads"] = reads.reindex(ev.index).fillna(0).astype(int)
    ev["n_seqs"] = nseq.reindex(ev.index).fillna(0).astype(int)
    audit["n_events_all"] = len(ev)
    ev = ev[ev["n_reads"] >= min_reads]
    audit["n_events_min_reads"] = len(ev)
    st = sites.set_index("site")
    ev["aquatic_type"] = ev["site_id"].map(st["aquatic_class"])
    ev["latitude"] = ev["site_id"].map(st["station_lat"] if "station_lat" in st else st["site_lat"]).astype(float)
    ev["longitude"] = ev["site_id"].map(st["station_lon"] if "station_lon" in st else st["site_lon"]).astype(float)
    ev["latitude"] = ev["latitude"].fillna(ev["site_id"].map(st["site_lat"]).astype(float))
    ev["longitude"] = ev["longitude"].fillna(ev["site_id"].map(st["site_lon"]).astype(float))
    ev["elevation"] = ev["site_id"].map(st["elev_m"]).astype(float)
    ev["source_product"] = source
    ev.index.name = "event_key"
    ev = ev.reset_index()
    ev["sample_id"] = [hashlib.sha1((source + "|" + k).encode()).hexdigest()[:16] for k in ev["event_key"]]
    tok = tok[tok["event_key"].isin(ev["event_key"])].merge(ev[["event_key", "sample_id"]], on="event_key")
    tax = o.drop_duplicates("seq_id").set_index("seq_id")
    for r in RANKS:
        if r in tax:
            tok[r] = tok["seq_id"].map(tax[r]).astype("string")
    seqs = tax.loc[tok["seq_id"].unique(), ["sequence"]].reset_index()
    seqs["length"] = seqs["sequence"].str.len()
    return ev, tok[["sample_id", "seq_id", "count", *[r for r in RANKS if r in tok]]], seqs, audit


def chem_var_for(units: pd.DataFrame, analyte: str) -> str:
    u = units[units["analyte"].astype(str).str.strip() == analyte]
    u = u[u["unit"].astype(str) == "milligramsPerLiter"]
    if len(u) != 1:
        raise ValueError(f"expected one mg/L chemistry variable for {analyte!r}, got {u.to_dict('records')}")
    return str(u["var"].iloc[0])


VISIT_SUFFIX = r"(?:\.(?:FIL|RAW|PCN|POM|CN|ANC|NUT|\d{1,2}))+$"   # replicate numbers 1-2 digits (dates have 8)


def visit_key(parent_id: pd.Series) -> pd.Series:
    """Field-visit key of a NEON chemistry/genetic parentSampleID: strips fraction / replicate suffixes
    such as ``.FIL.NUT``, ``.RAW.NUT.2``, ``.RAW.3`` (``ARIK.SS.20200916.FIL.NUT`` -> ``ARIK.SS.20200916``).
    Needed because ``neon_chem/chem_event_wide`` keeps those suffixes (one wide row per bottle), so
    post-2019 TN/TP/NO3 sit on ``*.RAW.NUT`` / ``*.FIL.NUT`` rows, not on the bare visit row."""
    return parent_id.astype(str).str.replace(VISIT_SUFFIX, "", regex=True)


def collapse_visits(w: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """One row per field visit: value = mean over the visit's bottles/replicates that report it
    (each analyte is reported on its own fraction), censored = any, time = earliest."""
    w = w.copy()
    w["visit"] = visit_key(w["parentSampleID"])
    cen = [f"{v}__censored" for v in value_cols if f"{v}__censored" in w]
    agg = {v: "mean" for v in value_cols if v in w}
    agg.update({c: "max" for c in cen})
    agg.update({"site": "first", "named_location": "first", "time_utc": "min"})
    for c in cen:
        w[c] = w[c].astype("float")
    out = w.groupby("visit").agg(agg)
    for c in cen:
        out[c] = out[c] > 0
    out["n_bottles"] = w.groupby("visit").size()
    out.index.name = "parentSampleID"
    return out.reset_index()


def match_chemistry(ev: pd.DataFrame, chem_wide: pd.DataFrame, units: pd.DataFrame, chem_long: pd.DataFrame | None,
                    window_h: float = 24.0) -> pd.DataFrame:
    w = chem_wide.copy()
    if "provisional" in w:
        w = w[~w["provisional"].fillna(False).astype(bool)]
    w["time_utc"] = pd.to_datetime(w["time_utc"], utc=True)
    var = {k: chem_var_for(units, a) for k, a in CHEM_ANALYTES.items()}
    w = collapse_visits(w, list(var.values()))
    dl = {}
    if chem_long is not None and "detection_limit" in chem_long:
        cl = chem_long.dropna(subset=["detection_limit"])
        cl = cl.assign(parentSampleID=visit_key(cl["parentSampleID"]))
        dl = {v: cl[cl["var"] == v].groupby("parentSampleID")["detection_limit"].max() for v in var.values()}
    by_parent = w.set_index("parentSampleID")
    rows = []
    for e in ev.itertuples(index=False):
        parents = [p for p in visit_key(pd.Series(str(e.parent_sample_ids).split("|"))).unique()
                   if p and p in by_parent.index]
        how, dt_h, rec = "none", np.nan, None
        if parents:
            rec = by_parent.loc[parents]
            if isinstance(rec, pd.DataFrame):   # several parents / duplicate rows: closest in time
                dtt = (rec["time_utc"] - pd.Timestamp(e.time_utc)).abs()
                rec = rec.iloc[int(np.argmin(dtt.to_numpy()))]
            how = "parentSampleID"
            dt_h = abs((rec["time_utc"] - pd.Timestamp(e.time_utc)).total_seconds()) / 3600
        else:
            cand = w[w["site"] == e.site_id]
            if "named_location" in cand and pd.notna(e.named_location):
                cand = cand[cand["named_location"].astype(str) == str(e.named_location)]   # same station (W4)
            if len(cand):
                d = (cand["time_utc"] - pd.Timestamp(e.time_utc)).abs().dt.total_seconds() / 3600
                j = d.idxmin()
                if d[j] <= window_h:
                    rec, how, dt_h = cand.loc[j], "time", float(d[j])
        out = {"sample_id": e.sample_id, "chem_match": how, "chem_dt_h": dt_h,
               "chem_parent": None if rec is None else rec.name if how == "parentSampleID" else rec["parentSampleID"]}
        for k, v in var.items():
            val = np.nan if rec is None else float(rec.get(v, np.nan))
            cen = False if rec is None else bool(pd.notna(rec.get(f"{v}__censored")) and rec.get(f"{v}__censored"))
            if cen and not np.isfinite(val) and v in dl:
                val = float(dl[v].get(out["chem_parent"], np.nan))
            out[k] = val
            out[k.replace("raw_", "") + "_censored"] = cen
        rows.append(out)
    return pd.DataFrame(rows)


def match_sonde(ev: pd.DataFrame, sonde_dir, units_ok: dict | None = None) -> pd.DataFrame:
    from pathlib import Path

    rows = []
    cache = {}
    for e in ev.itertuples(index=False):
        if e.site_id not in cache:
            p = Path(sonde_dir) / f"{e.site_id}.parquet"
            cache[e.site_id] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        d = cache[e.site_id]
        out = {"sample_id": e.sample_id, "sonde_hor": None, "sonde_ver": None}
        hor = SONDE_HOR.get(str(e.aquatic_type))
        loc = str(e.named_location).split(".AOS.")[-1] if pd.notna(e.named_location) else ""
        if not (loc == "S2" or loc.startswith("buoy")):   # sonde only co-located at S2 / buoy (audit O7)
            hor = None
        out["sonde_station_ok"] = hor is not None
        if len(d) and hor is not None:
            dd = d[(d["hor"].astype(str) == hor)]
            if "provisional" in dd:
                dd = dd[~dd["provisional"].fillna(False).astype(bool)]
            if len(dd):
                ver = sorted(dd["ver"].astype(str).unique())[0]
                dd = dd[dd["ver"].astype(str) == ver]
                dd = dd.assign(date=pd.to_datetime(dd["date"]).dt.normalize()).set_index("date")
                day = pd.Timestamp(e.collect_date)
                out["sonde_hor"], out["sonde_ver"] = hor, ver
                for k, v in SONDE_VARS.items():
                    col = f"{v}_mean"
                    if col not in dd:
                        out[k] = np.nan
                        out[k + "_pm1d"] = np.nan
                        continue
                    out[k] = float(dd[col].get(day, np.nan)) if day in dd.index else np.nan
                    win = dd.loc[(dd.index >= day - pd.Timedelta(days=1)) & (dd.index <= day + pd.Timedelta(days=1)), col]
                    out[k + "_pm1d"] = float(win.mean()) if win.notna().any() else np.nan
        for k in SONDE_VARS:
            out.setdefault(k, np.nan)
            out.setdefault(k + "_pm1d", np.nan)
        rows.append(out)
    return pd.DataFrame(rows)
