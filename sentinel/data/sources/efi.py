# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Ecological Forecasting Initiative (EFI) challenge targets: neon4cast aquatics, VERA, usgsrc4cast.

These are independent, publicly scored forecasting benchmarks (CRPS vs climatology /
persistence). Targets need no account; submitting forecasts needs a registered model id.
"""

from __future__ import annotations

import pandas as pd

TARGETS = {
    "neon4cast_aquatics": "https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/"
                          "project_id=neon4cast/duration=P1D/aquatics-targets.csv.gz",
    "vera4cast_daily": "https://amnh1.osn.mghpcc.org/bio230121-bucket01/vera4cast/targets/"
                       "project_id=vera4cast/duration=P1D/daily-insitu-targets.csv.gz",
    "usgsrc4cast_chl": "https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/"
                       "project_id=usgsrc4cast/duration=P1D/river-chl-targets.csv.gz",
}


def read_targets(path) -> pd.DataFrame:
    """Long-format targets: site_id, datetime (UTC, daily), variable, observation[, depth_m]."""
    df = pd.read_csv(path)
    df["variable"] = df["variable"].astype(str).str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df
