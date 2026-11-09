#!/usr/bin/env python3
"""EXPAND step 3 -- build mc_fusion_expanded.parquet (schema-identical + huc2).

Uses the SAME spectral_features() as analyze_microcystin_fusion.py so the 26
sf0..sf25 columns are numerically identical in definition to the current
mc_fusion_dataset.parquet. Merges:
  - satellite sf* from mc_expand_tiles (new) AND mc_tiles (original, re-featurized)
  - in-situ pigment CHLA/PC from wqp_hab_indicators.parquet (+/-3d nearest, same as original)
  - WQP drivers temp/P/N from data/raw/wqp_drivers/ (nearest by site/date; honest coverage)
  - clean huc2 region column

Writes mc_fusion_expanded.parquet + mc_fusion_expanded_manifest.json.
Pools the ORIGINAL dataset rows too (with huc2 backfilled) so the expanded set is
a proper superset for leave-one-region-out.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/bcheng/SENTINEL")
OUT_DIR = ROOT / "data/processed/biology/noaa_habs"
IND = OUT_DIR / "wqp_hab_indicators.parquet"
ORIG = OUT_DIR / "mc_fusion_dataset.parquet"
EXP_TARGETS = OUT_DIR / "mc_expand_targets.parquet"
EXP_TILES = OUT_DIR / "mc_expand_tiles"
WHO_THR = 8.0
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11"]


def spectral_features(img):
    """IDENTICAL to analyze_microcystin_fusion.py -> 26 features, sf0..sf25."""
    b = {name: img[i] for i, name in enumerate(BANDS)}
    c = slice(22, 42)
    feats = []
    for i, name in enumerate(BANDS):
        arr = img[i]
        feats += [float(arr.mean()), float(arr[c, c].mean()), float(arr.std())]
    eps = 1e-6
    ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + eps)
    ndci = (b["B05"] - b["B04"]) / (b["B05"] + b["B04"] + eps)
    ndwi = (b["B03"] - b["B08"]) / (b["B03"] + b["B08"] + eps)
    gr = b["B03"] / (b["B04"] + eps)
    for idx in (ndvi, ndci, ndwi, gr):
        feats += [float(idx.mean()), float(idx[c, c].mean())]
    return np.array(feats, dtype=np.float32)


def load_pigment_agg():
    ind = pd.read_parquet(IND, columns=[
        "MonitoringLocationIdentifier", "ActivityStartDate",
        "CharacteristicName", "ResultMeasureValue"])
    ind["date"] = pd.to_datetime(ind["ActivityStartDate"], errors="coerce").dt.normalize()
    ind["val"] = pd.to_numeric(ind["ResultMeasureValue"], errors="coerce")
    ind = ind.dropna(subset=["date", "val"])
    ind["grp"] = np.where(ind["CharacteristicName"].str.contains("hycocyanin"), "PC", "CHLA")
    ind = ind.rename(columns={"MonitoringLocationIdentifier": "Location_Identifier"})
    agg = (ind.groupby(["Location_Identifier", "date", "grp"])["val"].mean()
           .unstack("grp").reset_index())
    for c in ("CHLA", "PC"):
        if c not in agg.columns:
            agg[c] = np.nan
    return agg


def merge_pigment(df, agg):
    parts = []
    for sid, g in df.groupby("site"):
        a = agg[agg["Location_Identifier"] == sid][["date", "CHLA", "PC"]].sort_values("date")
        gg = g.sort_values("date")
        if len(a):
            gg = pd.merge_asof(gg, a, on="date",
                               tolerance=pd.Timedelta("3D"), direction="nearest")
        else:
            gg["CHLA"] = np.nan
            gg["PC"] = np.nan
        parts.append(gg)
    return pd.concat(parts).reset_index(drop=True)


def build_expanded_rows():
    mc = pd.read_parquet(EXP_TARGETS).reset_index(drop=True)
    mc["date"] = pd.to_datetime(mc["date"]).dt.normalize()
    rows = []
    for f in EXP_TILES.glob("*.npz"):
        idx = int(f.stem)
        if idx >= len(mc):
            continue
        try:
            d = np.load(f, allow_pickle=True)
            img = d["image"]
            if img.shape != (6, 64, 64):
                continue
            sf = spectral_features(img)
        except Exception:
            continue
        r = mc.iloc[idx]
        rows.append({
            "site": r["Location_Identifier"], "date": r["date"],
            "mc": float(r["val"]),
            "cloud": float(d["cloud"]) if "cloud" in d else np.nan,
            "huc2": r["huc2"], "state": r.get("state", None),
            **{f"sf{i}": v for i, v in enumerate(sf)},
        })
    return pd.DataFrame(rows)


def main():
    agg = load_pigment_agg()

    # --- new expanded rows ---
    exp = build_expanded_rows()
    print(f"expanded tiles featurized: {len(exp)} rows, "
          f"{exp['huc2'].dropna().nunique() if len(exp) else 0} HUC2")
    if len(exp):
        exp = merge_pigment(exp, agg)

    # --- original rows, re-used as-is (already have sf*, CHLA, PC), backfill huc2 ---
    orig = pd.read_parquet(ORIG)
    # site->huc2 from expand targets + cyanotoxin table
    cy = pd.read_parquet(OUT_DIR / "wqp_hab_cyanotoxins.parquet",
                         columns=["Location_Identifier", "Location_HUCEightDigitCode",
                                  "Location_Latitude", "Location_Longitude", "Location_State"])
    h8 = cy["Location_HUCEightDigitCode"].astype(str).str.replace(".0", "", regex=False)
    cy["huc2"] = np.where(h8.str.match(r"^\d+$"), h8.str.zfill(8).str[:2], None)
    site2huc = cy.dropna(subset=["huc2"]).groupby("Location_Identifier")["huc2"].first().to_dict()
    site2lat = cy.groupby("Location_Identifier")["Location_Latitude"].first().to_dict()
    site2lon = cy.groupby("Location_Identifier")["Location_Longitude"].first().to_dict()

    def geo_huc2(site):
        lat = pd.to_numeric(site2lat.get(site), errors="coerce")
        lon = pd.to_numeric(site2lon.get(site), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            return None
        # deterministic coarse fallback for CA/tribal (unmapped) sites
        if lon < -119 and 32 <= lat <= 43:
            return "18"   # California region
        return None

    orig["huc2"] = orig["site"].map(site2huc)
    m = orig["huc2"].isna()
    orig.loc[m, "huc2"] = orig.loc[m, "site"].map(geo_huc2)
    orig["state"] = None
    orig["source"] = "original"
    if len(exp):
        exp["source"] = "expanded"

    # unify schema: keep the ORIGINAL columns + huc2 (+state,source helpers)
    sf_cols = [f"sf{i}" for i in range(26)]
    keep = ["site", "date", "mc", "cloud"] + sf_cols + ["CHLA", "PC", "huc2", "state", "source"]
    for c in keep:
        if c not in orig.columns:
            orig[c] = np.nan
        if len(exp) and c not in exp.columns:
            exp[c] = np.nan
    frames = [orig[keep]]
    if len(exp):
        frames.append(exp[keep])
    full = pd.concat(frames, ignore_index=True)

    # dedup on (site,date) keeping original where duplicated
    full["date"] = pd.to_datetime(full["date"]).dt.normalize()
    full = full.sort_values("source")  # 'expanded' < 'original' -> keep original on dup
    full = full.drop_duplicates(subset=["site", "date"], keep="last").reset_index(drop=True)
    full["label"] = (full["mc"] > WHO_THR).astype(int)
    full["idx"] = np.arange(len(full))

    # attach drivers (honest coverage): nearest wqp driver by site+date within 30d
    def attach_driver(full, fname, colname):
        p = ROOT / "data/raw/wqp_drivers" / fname
        if not p.exists():
            full[colname] = np.nan
            return 0
        try:
            dv = pd.read_csv(p, low_memory=False)
        except Exception:
            full[colname] = np.nan
            return 0
        # identify site/date/value columns (exact names first; avoid grabbing
        # OrganizationIdentifier via a loose endswith fallback)
        scol = next((c for c in ("MonitoringLocationIdentifier", "Location_Identifier")
                     if c in dv.columns), None)
        dcol = next((c for c in dv.columns if "StartDate" in c or c == "date"), None)
        vcol = next((c for c in dv.columns if "ResultMeasureValue" in c or c == "Result_Measure"
                     or c == "val"), None)
        if not (scol and dcol and vcol):
            full[colname] = np.nan
            return 0
        dv = dv[[scol, dcol, vcol]].copy()
        dv.columns = ["site", "date", "v"]
        dv["date"] = pd.to_datetime(dv["date"], errors="coerce").dt.normalize()
        dv["v"] = pd.to_numeric(dv["v"], errors="coerce")
        dv = dv.dropna(subset=["date", "v"])
        vals = np.full(len(full), np.nan)
        by = {s: g.sort_values("date") for s, g in dv.groupby("site")}
        for i, r in full.iterrows():
            g = by.get(r["site"])
            if g is None:
                continue
            dd = (g["date"] - r["date"]).abs()
            j = dd.values.argmin()
            if dd.iloc[j] <= pd.Timedelta("30D"):
                vals[i] = g["v"].iloc[j]
        full[colname] = vals
        return int(np.isfinite(vals).sum())

    cov_t = attach_driver(full, "temp.csv", "temp_C")
    cov_p = attach_driver(full, "phosphorus.csv", "TP")
    cov_n = attach_driver(full, "nitrogen.csv", "TN")

    out = OUT_DIR / "mc_fusion_expanded.parquet"
    full.to_parquet(out, index=False)

    # manifest
    def reg_table(d):
        g = d.groupby("huc2").apply(lambda x: pd.Series({
            "samples": int(len(x)), "sites": int(x.site.nunique()),
            "pos_mc1.6": int((x.mc > 1.6).sum()), "pos_mc0.3": int((x.mc > 0.3).sum()),
            "pos_mc8": int((x.mc > 8).sum()),
            "chla_cov": int(x.CHLA.notna().sum()), "pc_cov": int(x.PC.notna().sum()),
        }), include_groups=False)
        return {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in g.iterrows()}

    n_reg_bar = 0
    for huc, x in full.groupby("huc2"):
        if len(x) >= 30 and int((x.mc > 1.6).sum()) >= 6:
            n_reg_bar += 1

    manifest = {
        "dataset": "mc_fusion_expanded.parquet",
        "schema_columns": list(full.columns),
        "sf_columns_match_original": sf_cols,
        "n_total": int(len(full)),
        "n_sites": int(full.site.nunique()),
        "n_huc2_regions": int(full["huc2"].dropna().nunique()),
        "n_expanded_new_rows": int((full["source"] == "expanded").sum()),
        "driver_coverage": {"temp_C": cov_t, "TP": cov_p, "TN": cov_n},
        "regions_meeting_DG_bar (>=30 samp & >=6 mc>1.6 pos)": int(n_reg_bar),
        "per_region": reg_table(full),
        "per_region_by_source_expanded_only": reg_table(full[full.source == "expanded"])
        if (full.source == "expanded").any() else {},
    }
    (OUT_DIR / "mc_fusion_expanded_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWROTE {out}")
    print(f"total={len(full)} sites={full.site.nunique()} HUC2={full['huc2'].dropna().nunique()}")
    print(f"regions meeting DG bar (>=30 samp, >=6 mc>1.6 pos): {n_reg_bar}")
    print(f"driver coverage temp={cov_t} TP={cov_p} TN={cov_n}")
    pd.set_option("display.width", 200)
    tab = pd.DataFrame(manifest["per_region"]).T
    print(tab)


if __name__ == "__main__":
    main()
