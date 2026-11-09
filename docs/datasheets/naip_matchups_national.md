# Datasheet — national NAIP × WQP matchups (`naip_matchups_national`)

| | |
|---|---|
| **Built** | 2026-11-08 by M7 with `scripts/data/build_naip_matchups.py --national` (code unchanged; same rules as `naip_matchups.md`) |
| **Scope** | every state in the WQP surface-sample table (52 incl. DC/territories with samples), WQP samples 2010-01-01 – 2024-01-31, NAIP 2010–2023 |
| **Licences** | as `naip_matchups.md` (NAIP public domain, see the caveat there; WQP public; Copernicus Sentinel data free and open) |
| **Outputs** | `$SENTINEL_DATA/processed/naip_matchups/national/{matchups.parquet, naip_scene_stats.parquet, coverage.json, build.log}`; chips appended to the existing stores `processed/naip_chips/wqp/` and `processed/s2_chips/naip_teacher/` |
| **Manifest** | `naip_matchups_national` |
| **Synthetic** | no |

## Counts (`coverage.json`)
- 6,493,304 WQP target samples at 186,419 sites; 78,223 sites pass the JRC water test.
- ±1-day NAIP matchups: 21,455 samples (Secchi 8,135; chl-a 5,552; turbidity 4,433; TSS 2,879; SSC 456) at
  11,287 (site, DOQQ) pairs; 11,288 NAIP chips OK; 7,027 matched samples have an S2 teacher chip (3,607 S2 chips).
- Lake/reservoir samples: Secchi 6,788; chl-a 3,580; turbidity 1,781; TSS 1,162.

## Relation to the 4-state build (frozen inputs of the original Lite track)
- The canonical `processed/naip_matchups/{matchups.parquet, coverage.json, naip_scene_stats.parquet, build.log}` and
  `manifests/naip_matchups.json` were snapshotted before the run (`processed/naip_matchups/snapshot_4state_20261024/`,
  `SHA256SUMS`) and restored byte-identical afterwards.
- The chip stores are **supersets**: all 6,195 previously indexed NAIP chips and 2,130 S2 chips keep their chip ids
  and shards (checked by request key). Their `index.parquet` files therefore differ from the hashes recorded in the
  old `naip_matchups` manifest; the pre-build indices are kept in the snapshot directory.
- Disk added ≈ 0.8 GB (NAIP chips 0.78 → 1.5 GB; S2 teacher chips 77 → 133 MB).

## Use
Lite national track (`docs/preregistration/lite_protocol.md` §12): labels `processed/lite_labels/labels_national.parquet`
(manifest `lite_labels_national`), split `docs/preregistration/lite_split_national.json`. Frozen 4-state test rows
keep their protected roles; new rows never enter training if their water body holds a frozen evaluation chip.

## Known issues
As `naip_matchups.md` (NAIP is not reflectance; per state-year colour balancing differs across 52 states — a larger
radiometric domain shift than the 4-state set; no time of day). Turbidity units: Lite uses NTU only.
