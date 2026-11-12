# SOTA campaign: every individual model (lead directive, 2026-11-09)

**User directive:** "we need all of our individual models to be SOTA, so perhaps harder tasks, more data".
This applies to M1 Sonde/EW, M2 Orbit, M3 Microbe, M4 ChemTox, M5 BioAssess, M6 HydroMet and M7 Lite.

## Status at directive time

| Model | Status | Bar |
|---|---|---|
| M5 | SOTA (ensemble C, chloride/pH/conductivity, Test A) | met |
| M1 EW | Test-A underpowered, no claim | not met |
| M4 | competitive | not met |
| M6 | competitive | not met |
| M2 | competitive on val | not met |
| M3 | pending | — |
| M7 | negative on val | not met |

## What counts (RESEARCH_PLAN §15.1 is unchanged)

A SOTA label still requires all of the following:
- a paired CI that excludes 0 against both the strongest published method and the strongest simple baseline;
- Holm correction;
- at least 3 seeds;
- a single look at an UNSEEN test block, frozen beforehand.

A harder task or more data is a means to that end, never a way to lower the bar.

## Rules, to keep this honest (garden-of-forking-paths control)

1. **No reuse of spent test blocks.** A spent block is never used for training, tuning or a second claim (§15.5 item 7). Every new track names an unseen test block: future data, a new geography, or a new source.
2. **Justify new tracks before any result.** Each harder track is pre-registered with a practical justification *before* any validation result on it exists. Examples:
   - out-of-region transfer;
   - extremes or rare conditions;
   - sparse-data sites;
   - new sensors;
   - longer horizons.

   Choosing a track because validation showed a win is forbidden. The protocol must record when, and why, each track was chosen.
3. **Every track is reported, wins and losses.** A SOTA label is scoped to its track ("SOTA on out-of-region chloride"), never generalised. Originals are never replaced (§15.5 items 5–7).
4. **More data comes first.** Usually the most honest lever is to add training data the baselines get too: more sites, years, states and countries. Comparator parity is mandatory (same inputs, same data).
5. **Diagnose before adding capacity.** Where a GBM ties or wins, first ask: does the learned model have information or inductive bias the GBM lacks? If not, more model capacity will not help. Instead consider:
   - sequence or spatial context;
   - upstream graphs;
   - multi-task learning;
   - pretraining on larger related corpora;
   - physics constraints.
6. **Caps.** Each new track has a pre-declared configuration cap. All configurations are counted.
7. **Single-look guard with the latest hardening on every new track:**
   - a sealed registry row;
   - rescore refused after scoring;
   - access closed at sealing;
   - all artifacts verified before the claim;
   - BaseException aborts;
   - the claim tied to pid/host;
   - a full import-closure code hash.

   Write-ups go in `*_test_results.md`, never into frozen files.
8. **Shared limits:**
   - GPUs 0/1 only, through the lock helper, at most 2 h per job;
   - CPU at most 6 threads per job;
   - no git commits by agents;
   - never read `keys.env`.

## Deliverable per workstream (in its handoff, before running anything new)

A short **SOTA plan**:
- **Diagnosis:** why the model is not SOTA now.
- **Levers:** data, task and method, ranked by expected gain per GPU-hour.
- **Track definitions:** each new or harder track, with its unseen test block and its power check.
- **Caps and ETA.**

The lead reviews each plan before compute is spent.

## Disk rule (lead, 2026-11-09)
The host disk is 98% full: about 100 GB free, shared with another user.
- Before any download or build, run `df -h /home/bcheng`. Stop and report to the lead if free space would drop below 60 GB.
- New-data caps per workstream:

  | Workstream | Cap |
  |---|---|
  | M6 (EA England + CH/Bavaria) | 20 GB |
  | M7 | 10 GB |
  | M4 | 10 GB |
  | M1 | 10 GB |
  | M2 | as already built |
  | M3 | 5 GB |
  | M5-X | 5 GB |

- Delete intermediate caches when they are no longer needed.

## Reserved unseen blocks (lead, 2026-11-09), never used for training or tuning by ANY workstream
- `docs/preregistration/sonde_newsites_list.json` (sha 6200a77c…): 729 USGS sites reserved as the EW-v3 / M1-N test block. About 470 of them also have DV files in the shared `processed/sonde_daily`, so any loader reading that tier must exclude them.
- `docs/plans/ea_england_allocation.md`: EA values from 2023-10-01 onward, M6's test-role groups, and M1's EA block.
- Each workstream's own spent and reserved test blocks, as listed in its protocol.

## Shared-input drift note (lead, 2026-11-10)
- `processed/forcings_daily` was rebuilt on 2026-10-27, adding ERA5 weather at sites that had none before.
- Any model trained or validated on weather BEFORE that date sees different inputs today.
- Before re-scoring or reusing such a model, check input identity, for example with M1's `--weather-exclude-sources era5_seamless`, or re-pin the inputs.
- Open-Meteo has a shared daily budget across all workstreams. Any single puller is capped at about 60 % of it.

## GPU rule update (lead, 2026-11-10, user instruction "you can use all of the other gpus")
- The cap file `.gpu_locks/ALLOWED` now reads `0,1,2,3,4,5,6,7,8,9`.
- Always go through `sentinel/core/gpu.py` with `--min-free-mb`. Never start on a GPU another user's process occupies.
- Remove any hard-coded `SENTINEL_GPUS=0,1`; widen it to `0-9`, or leave it unset so the cap applies.
- Keep jobs at ≤ 2 h (chunk or checkpoint) so GPUs free up quickly if other users return.
- 2026-11-10: the user deleted the unrelated REVENANT/ECHO/SCARLET projects, leaving about 184 GB free on `/`. The 60 GB floor and the per-workstream caps are unchanged.

## Reserved unseen blocks added 2026-11-10 (lead; architecture round v2, before any result on them)
- **M2 Test-S** (`testxarchive_lake`): in-lake, non-buffer rows of test-role water bodies (orbit split v3) dated
  2015-06-23..2023-12-31, all chip sources. Reserved from 2026-11-10. Never used to train or tune any model that
  takes Sentinel-2 input or that is fused with M2. Other workstreams exclude these site-dates where practical and
  report any overlap (counts only) to the lead.
- **M5-F France:** Naïades / Hub'eau biology operations dated 2023–2025 and their paired physico-chemistry.
  Reserved before download. Lock-on-download as for `sonde_newsites`.
- **M1 StreamPULSE (M1-SP) and M7 P2:** each locks its test block on download or at index build, in its own
  amendment, before any label value is read.
- **M4 `T1_test`, `T6a_test`, `T6b_test`** (SENTINEL-Toxin, architecture round v2): the rows with those values of
  the block column in `chemtox_t1t6_split_v2.parquet`, sha256
  `ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3`. Rows / hold-out units: T1_test 11,363 / 546,
  T6a_test 11,249 / 694, T6b_test 4,106 / 289. Reserved at the lead stamp of 2026-11-10
  (`docs/preregistration/chemtox_arch_v2_lead_rulings.md`, C-4). No workstream trains or tunes on those
  `result_id`s; any overlap is reported to the lead as counts. The first-text split
  (`chemtox_t1t6_split_v1.parquet`) is superseded and is not a reservation.
- Decisions and scope: `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`.
