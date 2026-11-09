# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the SENTINEL-Sonde daily builder (synthetic rows exercise code paths only)."""

import numpy as np
import pandas as pd

from sentinel.data.sonde_daily import clean_long, splice_site
from sentinel.data.sources.usgs_legacy import _wide_to_long, read_rdb


def _rows(ts, dates, values, quals, parm="00300", stat="00003"):
    return pd.DataFrame({"site_no": "01", "parm_cd": parm, "stat_cd": stat, "ts_id": ts,
                         "value": np.asarray(values, "float32"), "qualifier": quals,
                         "date": pd.to_datetime(dates).date})


def test_qualifier_and_range_masking():
    d = pd.date_range("2020-01-01", periods=6)
    df = _rows("1", d, [8, 8, 8, 30, 8, 8], ["A", "A:e", "A:[4]", "A", "P", "A:<"])
    clean, counts = clean_long(df)
    assert counts["masked_qualifier"] == 3 and counts["masked_implausible"] == 1
    assert len(clean) == 2 and clean["provisional"].tolist() == [False, True]


def test_splice_fills_gaps_only_when_consistent():
    d1 = pd.date_range("2020-01-01", periods=60)
    d2 = pd.date_range("2020-01-31", periods=60)  # 30-day overlap
    good = pd.concat([_rows("1", d1, np.full(60, 8.0), "A"), _rows("2", d2, np.full(60, 8.1), "A")])
    wide, log = splice_site(clean_long(good)[0])
    assert wide["do_mgl_mean"].notna().sum() == 90 and log[0]["used"]
    bad = pd.concat([_rows("1", d1, np.full(60, 8.0), "A"), _rows("2", d2, np.full(60, 12.0), "A")])
    wide, log = splice_site(clean_long(bad)[0])
    assert wide["do_mgl_mean"].notna().sum() == 60 and not log[0]["used"]
    assert set(wide["do_mgl_mean_ts"].unique()) == {"1"}


def test_rdb_missing_codes_become_nan():
    text = ("# comment\nagency_cd\tsite_no\tdatetime\t123_00300_00003\t123_00300_00003_cd\n"
            "5s\t15s\t20d\t14n\t10s\nUSGS\t01\t2020-01-01\t8.1\tA\nUSGS\t01\t2020-01-02\tEqp\tP\n"
            "USGS\t01\t2020-01-03\t-999999\tA\n")
    long = _wide_to_long(read_rdb(text), "datetime", has_stat=True)
    assert long["value"].notna().sum() == 1
    assert "Eqp" in long.loc[long["value"].isna(), "qualifier"].iloc[0]
