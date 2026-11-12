# Datasheet — national NAIP × WQP matchups, Secchi ±7 d (`naip_matchups_national_pm7`)

| | |
|---|---|
| **Built** | 2026-11-09 by M7 (SOTA lever 1(a)) with `scripts/data/build_naip_matchups_window.py`. It is generated from `build_naip_matchups.py`, which is unchanged; the only change is the window rule. |
| **Window** | Secchi: \|sample − NAIP date\| ≤ 7 d (lake Secchi is slow-varying). Turbidity, TSS, SSC and chl-a: ≤ 1 d (unchanged). S2 teacher: nearest usable S2 within ±1 d of the NAIP date (unchanged). |
| **Scope** | All states in the WQP table; samples 2010-01-01 – 2024-01-31; NAIP 2010–2023 (Planetary Computer) |
| **Outputs** | `$SENTINEL_DATA/processed/naip_matchups/national_pm7/{matchups.parquet, coverage.json, naip_scene_stats.parquet, build.log}`. Chips are appended to `processed/naip_chips/wqp` and `processed/s2_chips/naip_teacher`. |
| **Licences** | as `naip_matchups.md` |
| **Synthetic** | no |

## Counts (`coverage.json`)
- 51,381 matched samples: Secchi 38,061; chl-a 5,552; turbidity 4,433; TSS 2,879; SSC 456.
- 33,863 NAIP chips OK, of which 22,575 are new (26 `outside_image`). 16,461 samples have an S2 teacher chip
  (10,694 S2 chips).
- The Secchi counts equal the counts-only pre-declaration (`$SENTINEL_RUNS/lite/counts_national2/counts.json`).

## Relation to earlier builds
- All 11,288 previously extracted NAIP chips and 3,607 S2 chips keep their chip ids and shards, checked by request
  key against `processed/naip_matchups/snapshot_national_20261108/`. The one `outside_image` row (no chip) also
  compares as unchanged.
- Disk added: NAIP chips 1.5 → 4.3 GB; S2 teacher chips 133 → 388 MB (about +3.1 GB total). Free disk was 108 GB
  before and 95 GB after (the disk is shared).

## Use
Training rows of the Lite national-2 split only (`lite_national_protocol.md` N4/N5). New chips enter as `train`
only; every evaluation block keeps its national-1 chips and ±1-day labels.

## Known issues
- A ±7-day Secchi label pairs imagery and the measurement up to a week apart, which adds label noise.
- Evaluation labels are unaffected (±1 d).
