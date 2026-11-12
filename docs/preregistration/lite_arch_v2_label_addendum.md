# SENTINEL-Lite architecture round v2 — label addendum (counts only)

- **Written:** 2026-11-10 19:02 EST (2026-11-11 00:02 UTC) by the M7 workstream agent, on the lead's order of 2026-11-10 15:05 EST (confirmed
  18:45 EST).
- **Status:** for the lead's stamp, together with `lite_arch_v2_amendment.md` v0.3 (sha256
  `2b369018d2fc2a340594790756b3edbef3b1de28a38cf6073eb5908111908a56`, not edited), `lite_arch_v2_deviations.md` and the lead's
  `lite_arch_v2_lead_rulings.md`. The amendment text is not changed by this file. Where a count here differs from a
  count in the amendment, this file is the count of record (ruling on the missed point 2 of the verified review).
- **Counts only.** This file holds no label value and no per-row statistic of a validation, test or lock row. Counts
  on protected roles were made by row membership (keys site, date, target, characteristic, unit) and by
  present/absent flags, never by comparing values. No model was fitted for this file, and no validation number of any
  kind exists on the new label version.
- **Sources of every number** (`$SENTINEL_RUNS/lite/counts_arch_v2/`):

  | File | sha256 | Written by |
  |---|---|---|
  | `labels2b_counts.json` | `f3f3702f24e56be858c721a304dbbbdd93cdeeb7ad7ba06fc1ec7a9c98f328c6` | `scripts/data/litev2_build_labels2b.py` |
  | `frozen_rederive_stopcheck.json` | `070f776d29e89e2af780372fb155e631d8f3ebf2a95980c6d4696498997b2ea9` | the same script (4-state chips, STOP check of order 2(b)) |
  | `stamp_counts.json` (0444) | `2ab6a81210d848ae2d234a12ff26dc149dccd332cb065a25c13debd62bb08a4e` | `scripts/data/litev2_stamp_counts.py` |
  | `addendum_counts.json` | `9318b9eb03ef8f0d82f048110a984a266ed6e038d4446d3b4692860b2b1050a1` | `scripts/data/litev2_addendum_counts.py` |

Sections (i)–(x) follow the numbering of item 1 of the lead's order.

## (i) Rows kept, dropped and changed, per build

### The rule (ruling F3; one project rule)
Every WQP label comes from the harmonised, QC-passed tables only (`wqp_discrete/variable=*/`). A surface result has
depth ≤ 1.0 m or unknown depth; a result flagged "bottom" is dropped at any depth; Secchi is exempt. The sample label
is the median of those results over (site, date, target, characteristic, unit). A sample with no such result is
dropped, with the reason `no_qc_passed_result` or `no_surface_result`. The reader is M2's
`sentinel.models.orbit.wqp_surface.apply`, imported and not edited. Sonde labels are unchanged.

"Changed" below means a sample that is kept with a different number of results than as built (its median may or may
not differ). On validation and test roles values were not compared, so this is the only measure of change given.

### (i-a) The 32,820 national-2 chips, by role and target (chips)
Chip roles and water-body ids are unchanged (asserted on the `chip_role` mapping alone; unit test
`tests/test_litev2_stamp.py`). No as-built WQP label remains in the new version: each of the four original labels of
every chip was re-derived through the rule.

| Role | Chips | Secchi: as built → corrected (lost / gained) | Turbidity (NTU): as built → corrected (lost / gained) | TSS: as built → corrected (lost / gained) | Chl-a: as built → corrected (lost / gained) | Chips left with no original label |
|---|---|---|---|---|---|---|
| `train` | 10,535 | 9,371 → 9,371 (0 / 0) | 1,555 → 1,453 (104 / 2) | 1,251 → 1,206 (46 / 1) | 2,059 → 1,876 (184 / 1) | 107 |
| `val` | 1,278 | 1,163 → 1,163 (0 / 0) | 132 → 130 (2 / 0) | 161 → 160 (1 / 0) | 428 → 422 (6 / 0) | 2 |
| `val_n` | 780 | 365 → 365 (0 / 0) | 483 → 467 (16 / 0) | 303 → 291 (12 / 0) | 364 → 300 (64 / 0) | 36 |
| `test` | 1,178 | 1,094 → 1,094 (0 / 0) | 92 → 91 (1 / 0) | 98 → 98 (0 / 0) | 310 → 303 (7 / 0) | 0 |
| `test_n` | 773 | 379 → 379 (0 / 0) | 486 → 460 (26 / 0) | 270 → 262 (8 / 0) | 422 → 350 (72 / 0) | 39 |
| `test_time` | 747 | 602 → 602 (0 / 0) | 190 → 190 (0 / 0) | 166 → 166 (0 / 0) | 361 → 361 (0 / 0) | 0 |
| `test_time_n` | 426 | 205 → 205 (0 / 0) | 258 → 248 (10 / 0) | 146 → 144 (3 / 1) | 223 → 202 (21 / 0) | 13 |
| `excluded` | 17,103 | 16,952 → 16,952 (0 / 0) | 146 → 133 (13 / 0) | 54 → 38 (16 / 0) | 114 → 101 (13 / 0) | 31 |
| all roles | 32,820 | 30,131 → 30,131 (0 / 0) | 3,342 → 3,172 (172 / 2) | 2,449 → 2,365 (86 / 2) | 4,281 → 3,915 (367 / 1) | 228 |

- "Gained" (5 labels on 3 targets) are chips where the as-built median was rejected by the frozen bounds and the
  surface-only median is not.
- Whole-version effect (the lead's finding (a); ruled 18:45 EST: no change, the rows go):
  - `test_n` 39 of 773 chips (5.0 %), `val_n` 36 of 780 (4.6 %), `test_time_n` 13 of 426 (3.1 %), `train` 107 of
    10,535 (1.0 %), `val` 2 of 1,278 (0.2 %), `test` 0, `test_time` 0 are left with no original label.
  - Training labels: chl-a 2,059 → 1,876 (−8.9 %), turbidity 1,555 → 1,453 (−6.6 %), TSS 1,251 → 1,206 (−3.6 %),
    Secchi 9,371 → 9,371.
- STOP check of order 2(b), on the 5,998 4-state chips alone: `train` 1 of 2,795 chips (0.04 %), `val` 2 of 1,278
  (0.16 %), `test` 0 of 1,178, `test_time` 0 of 747 lose every label; the largest loss of a target's 4-state training
  labels is chl-a, 12 of 994 (1.2 %). Both limits (5 % of a role's chips, 10 % of a target's training rows) are met.

Chips left with no original label, by origin and reason:

| Origin of the chip | Role | Chips | With no original label left | no QC-passed result | no surface result | result kept, rejected by the frozen label rules (unit, bounds, censoring) |
|---|---|---|---|---|---|---|
| 4-state build | `train` | 2,795 | 1 | 0 | 0 | 1 |
| 4-state build | `val` | 1,278 | 2 | 1 | 0 | 1 |
| 4-state build | `test` | 1,178 | 0 | 0 | 0 | 0 |
| 4-state build | `test_time` | 747 | 0 | 0 | 0 | 0 |
| national ±1 d build | `train` | 2,036 | 106 | 17 | 64 | 25 |
| national ±1 d build | `val_n` | 780 | 36 | 28 | 6 | 2 |
| national ±1 d build | `test_n` | 773 | 39 | 29 | 8 | 2 |
| national ±1 d build | `test_time_n` | 426 | 13 | 0 | 12 | 1 |
| national ±1 d build | `excluded` | 564 | 31 | 10 | 20 | 1 |
| Secchi ±7 d build | `train` | 5,704 | 0 | 0 | 0 | 0 |
| Secchi ±7 d build | `excluded` | 16,539 | 0 | 0 | 0 | 0 |

### (i-b) Sample rows of the 4-state build (re-derived by key; 10,432 row keys)
Key: organisation-qualified site id, date, target, characteristic. No as-built label was kept, and M2's side table
was not used for these keys (order 2(b)).

| Target | Role of the chip | Rows | Kept, same results | Kept, different number of results | Dropped: no QC-passed result | Dropped: no surface result |
|---|---|---|---|---|---|---|
| Secchi | `train` | 2,583 | 2,570 | 12 | 1 | 0 |
| Secchi | `val` | 1,196 | 1,168 | 23 | 5 | 0 |
| Secchi | `test` | 1,116 | 1,104 | 6 | 6 | 0 |
| Secchi | `test_time` | 621 | 608 | 2 | 11 | 0 |
| **Secchi** | **all** | **5,516** | **5,450** | **43** | **23** | **0** |
| Turbidity (NTU) | `train` | 689 | 654 | 27 | 8 | 0 |
| Turbidity (NTU) | `val` | 147 | 140 | 0 | 7 | 0 |
| Turbidity (NTU) | `test` | 95 | 91 | 2 | 2 | 0 |
| Turbidity (NTU) | `test_time` | 207 | 207 | 0 | 0 | 0 |
| **Turbidity (NTU)** | **all** | **1,138** | **1,092** | **29** | **17** | **0** |
| TSS | `train` | 569 | 551 | 5 | 12 | 1 |
| TSS | `val` | 176 | 161 | 0 | 15 | 0 |
| TSS | `test` | 103 | 98 | 1 | 4 | 0 |
| TSS | `test_time` | 168 | 168 | 0 | 0 | 0 |
| **TSS** | **all** | **1,016** | **978** | **6** | **31** | **1** |
| Chl-a | `train` | 1,268 | 1,245 | 0 | 23 | 0 |
| Chl-a | `val` | 600 | 582 | 0 | 18 | 0 |
| Chl-a | `test` | 393 | 382 | 2 | 9 | 0 |
| Chl-a | `test_time` | 501 | 501 | 0 | 0 | 0 |
| **Chl-a** | **all** | **2,762** | **2,710** | **2** | **50** | **0** |

### (i-c) Sample rows of the national ±1 d build (21,455 rows; 1,430 dropped, 20,025 kept)
Role = national-2 role of the row's chip; `none` = the row's chip has no national-2 role. Rows of 4-state chips
that the national build also holds appear under `train`, `val`, `test` and `test_time`.

| Target | Role of the chip | Rows | Kept, same results | Kept, fewer results | Dropped: no QC-passed result | Dropped: no surface result |
|---|---|---|---|---|---|---|
| Secchi | `train` | 3,751 | 3,697 | 47 | 7 | 0 |
| Secchi | `val` | 1,196 | 1,168 | 23 | 5 | 0 |
| Secchi | `val_n` | 374 | 370 | 3 | 1 | 0 |
| Secchi | `test` | 1,116 | 1,103 | 7 | 6 | 0 |
| Secchi | `test_n` | 382 | 375 | 4 | 3 | 0 |
| Secchi | `test_time` | 621 | 608 | 2 | 11 | 0 |
| Secchi | `test_time_n` | 206 | 203 | 3 | 0 | 0 |
| Secchi | `excluded` | 425 | 421 | 3 | 1 | 0 |
| Secchi | `none` | 64 | 34 | 0 | 30 | 0 |
| **Secchi** | **all** | **8,135** | **7,979** | **92** | **64** | **0** |
| Turbidity (NTU) | `train` | 1,848 | 1,392 | 278 | 54 | 124 |
| Turbidity (NTU) | `val` | 148 | 105 | 35 | 8 | 0 |
| Turbidity (NTU) | `val_n` | 518 | 409 | 92 | 8 | 9 |
| Turbidity (NTU) | `test` | 99 | 90 | 4 | 3 | 2 |
| Turbidity (NTU) | `test_n` | 547 | 427 | 78 | 22 | 20 |
| Turbidity (NTU) | `test_time` | 209 | 147 | 60 | 0 | 2 |
| Turbidity (NTU) | `test_time_n` | 332 | 255 | 44 | 6 | 27 |
| Turbidity (NTU) | `excluded` | 153 | 136 | 2 | 0 | 15 |
| Turbidity (NTU) | `none` | 579 | 407 | 86 | 72 | 14 |
| **Turbidity (NTU)** | **all** | **4,433** | **3,368** | **679** | **173** | **213** |
| TSS | `train` | 1,466 | 1,168 | 135 | 44 | 119 |
| TSS | `val` | 176 | 147 | 14 | 15 | 0 |
| TSS | `val_n` | 337 | 279 | 36 | 4 | 18 |
| TSS | `test` | 105 | 95 | 4 | 4 | 2 |
| TSS | `test_n` | 312 | 232 | 65 | 7 | 8 |
| TSS | `test_time` | 170 | 114 | 54 | 0 | 2 |
| TSS | `test_time_n` | 160 | 134 | 20 | 4 | 2 |
| TSS | `excluded` | 61 | 39 | 4 | 10 | 8 |
| TSS | `none` | 92 | 66 | 6 | 11 | 9 |
| **TSS** | **all** | **2,879** | **2,274** | **338** | **99** | **168** |
| Chl-a | `train` | 2,605 | 2,072 | 194 | 128 | 211 |
| Chl-a | `val` | 600 | 550 | 32 | 18 | 0 |
| Chl-a | `val_n` | 488 | 325 | 32 | 69 | 62 |
| Chl-a | `test` | 398 | 372 | 12 | 9 | 5 |
| Chl-a | `test_n` | 551 | 391 | 31 | 86 | 43 |
| Chl-a | `test_time` | 504 | 455 | 46 | 1 | 2 |
| Chl-a | `test_time_n` | 260 | 199 | 16 | 35 | 10 |
| Chl-a | `excluded` | 128 | 107 | 4 | 3 | 14 |
| Chl-a | `none` | 18 | 15 | 0 | 2 | 1 |
| **Chl-a** | **all** | **5,552** | **4,486** | **367** | **351** | **348** |
| SSC | `train` | 50 | 47 | 2 | 1 | 0 |
| SSC | `val_n` | 42 | 42 | 0 | 0 | 0 |
| SSC | `test_n` | 32 | 28 | 2 | 2 | 0 |
| SSC | `test_time_n` | 20 | 19 | 0 | 0 | 1 |
| SSC | `excluded` | 2 | 2 | 0 | 0 | 0 |
| SSC | `none` | 310 | 286 | 14 | 2 | 8 |
| **SSC** | **all** | **456** | **424** | **18** | **5** | **9** |

### (i-d) Sample rows of the Secchi ±7 d build (51,381 rows; 1,609 dropped, 49,772 kept)
The ±7 d build differs from the ±1 d build for Secchi only (checked: every other target has identical counts).

| Target | Role of the chip | Rows | Kept, same results | Kept, fewer results | Dropped: no QC-passed result | Dropped: no surface result |
|---|---|---|---|---|---|---|
| Secchi | `train` | 11,751 | 11,569 | 173 | 9 | 0 |
| Secchi | `val` | 1,734 | 1,702 | 27 | 5 | 0 |
| Secchi | `val_n` | 471 | 464 | 6 | 1 | 0 |
| Secchi | `test` | 1,666 | 1,653 | 7 | 6 | 0 |
| Secchi | `test_n` | 455 | 448 | 4 | 3 | 0 |
| Secchi | `test_time` | 832 | 819 | 2 | 11 | 0 |
| Secchi | `test_time_n` | 274 | 271 | 3 | 0 | 0 |
| Secchi | `excluded` | 20,419 | 20,075 | 341 | 3 | 0 |
| Secchi | `none` | 459 | 248 | 6 | 205 | 0 |
| **Secchi** | **all** | **38,061** | **37,249** | **569** | **243** | **0** |

### (i-e) national-2b additions (chips without a national-2 role)
- Role-less chips in the ±7 d build with a kept row: 763. With a recovered label under the rule: 719. Dropped because
  the site is in the reserved new-site block: 5. Missing from the NAIP chip store: 0. **Added: 714 chips.**
- Roles of the 714 (national-1 rules, unchanged): **144 `train`** (in 77 water bodies), **570 `excluded`**. Reasons
  for `excluded` (a chip can fail more than one rule): water-body rule 467, 2-km buffer 110, block rule 70, year rule
  (NAIP year > 2021) 87. 559 of the 714 lie in a water body that already holds a national-2 chip. 44 blocks had no
  national-2 role and were drawn by the national-1 salted rule: 26 train, 11 val, 7 test.
- No chip was added to `val`, `val_n` or to any test role, and no original-target label was built on an added chip.
- **national-2b: 33,534 chips** = 32,820 + 714. Roles: train 10,679; val 1,278; val_n 780; test 1,178; test_n 773;
  test_time 747; test_time_n 426; excluded 17,673.

Label columns of national-2b by role (chips with a present label; recovered columns exist on `train`, `val` and
`val_n` only, NTRU and censored rows on `train` only, as §3.4 of the amendment and Q2 say; asserted in the build):

| Role | Chips | With a label | Secchi | Turbidity (NTU) | TSS | Chl-a | Turbidity FNU | Turbidity NTRU | SSC | Censored Secchi | Censored TSS | Censored chl-a | Censored turbidity | Censored SSC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `train` | 10,679 | 10,595 | 9,371 | 1,453 | 1,206 | 1,876 | 120 | 41 | 135 | 47 | 83 | 51 | 3 | 0 |
| `val` | 1,278 | 1,276 | 1,163 | 130 | 160 | 422 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `val_n` | 780 | 744 | 365 | 467 | 291 | 300 | 13 | 0 | 41 | 0 | 0 | 0 | 0 | 0 |
| `test` | 1,178 | 1,178 | 1,094 | 91 | 98 | 303 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `test_n` | 773 | 734 | 379 | 460 | 262 | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `test_time` | 747 | 747 | 602 | 190 | 166 | 361 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `test_time_n` | 426 | 413 | 205 | 248 | 144 | 202 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `excluded` | 17,673 | 17,072 | 16,952 | 133 | 38 | 101 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Training rows by label column and origin:

| Label column | As built (national-2) | national-2b file | of these on national-2 chips | on added chips | At load (4 reserved-site chips out) | Water bodies at load |
|---|---|---|---|---|---|---|
| Secchi | 9,371 | 9,371 | 9,371 | 0 | 9,371 | 2,822 |
| Turbidity (NTU) | 1,555 | 1,453 | 1,453 | 0 | 1,453 | 411 |
| TSS | 1,251 | 1,206 | 1,206 | 0 | 1,203 | 390 |
| Chl-a | 2,059 | 1,876 | 1,876 | 0 | 1,875 | 638 |
| Turbidity FNU | 0 | 120 | 110 | 10 | 117 | 41 |
| Turbidity NTRU | 0 | 41 | 23 | 18 | 41 | 26 |
| SSC | 0 | 135 | 45 | 90 | 134 | 67 |
| Censored Secchi | 0 | 47 | 14 | 33 | 47 | 39 |
| Censored TSS | 0 | 83 | 71 | 12 | 83 | 56 |
| Censored chl-a | 0 | 51 | 49 | 2 | 51 | 29 |
| Censored turbidity (NTU) | 0 | 3 | 3 | 0 | 3 | 3 |
| Censored SSC | 0 | 0 | 0 | 0 | 0 | 0 |

- "At load": 4 `train` chips of national-2 lie at 3 sites of the reserved new-site block (deviation 3). Their role is
  unchanged in the pin file, and no loader of this round returns them. Chips at reserved sites in every other role:
  0 (val 0, val_n 0, test 0, test_n 0, test_time 0, test_time_n 0, excluded 0); P2 lock sites and index rows at
  reserved sites: 0 in each P2 role.
- v2 training rows of the national table at load: 10,591 chips with at least one label, in 3,197 water bodies.

### (i-f) P2 (first build → rebuild under the rule)
Only the WQP label filter changed. The lock file and every constant of Q6 are unchanged.

- Lock sites 1,963, of which 4 are excluded at load (deviation 2) → 1,959; 1,934 have a NAIP image (13,686
  site–image-date pairs); 514 pairs have a label; 511 index rows (first build: 513).
- `test_p2`: 2 chips are in the first build only (their WQP sample has no QC-passed surface result); label present
  206 → 204; usable 189 → 187. Found by key and present/absent flag; no `test_p2` value was read.
- `train_p2`: 145 → 145 rows, 0 label values changed, 0 label sources changed, usable 128 → 128.
- `val_p2`: 162 → 162 rows, 0 label values changed, 0 label sources changed, usable 153 → 153.
- Censored site-dates: 0 in every role, as before.

## (ii) Corrected table of section 3.1 (sample rows of the ±7 d build; role = national-2 role of the row's chip)

| Recovered group | Rows v0.3 → recount | On train-role chips | On chips with no national-2 role | On val / val_n | On the four test roles | On `excluded` chips |
|---|---|---|---|---|---|---|
| Turbidity FNU | 471 → **402** | 128 → **115** | 237 → **201** | 16 → **15** | 87 → **68** | 3 → **3** |
| Turbidity NTRU, uncensored | 319 → **309** | 25 → **25** | 260 → **260** | 13 → **13** | 19 → **9** | 2 → **2** |
| SSC mg/L, uncensored | 448 → **439** | 49 → **49** | 306 → **298** | 41 → **41** | 50 → **49** | 2 → **2** |
| Censored Secchi | 316 → **285** | 32 → **28** | 245 → **220** | 9 → **9** | 11 → **9** | 19 → **19** |
| Censored TSS | 282 → **214** | 111 → **73** | 73 → **70** | 44 → **23** | 48 → **43** | 6 → **5** |
| Censored chl-a | 162 → **143** | 70 → **67** | 17 → **14** | 43 → **33** | 26 → **25** | 6 → **4** |
| Censored turbidity (NTU) | 26 → **20** | 3 → **3** | 16 → **10** | 1 → **2** | 6 → **5** | 0 → **0** |
| Censored SSC | 3 → **3** | 0 → **0** | 2 → **2** | 1 → **1** | 0 → **0** | 0 → **0** |
| Censored turbidity FNU / NTRU (not recovered) | 4 → **5** | **0** | **4** | **0** | **1** | **0** |

- The verifier's reading "at most 115 now, at most 316" for FNU is confirmed by the recount: 115 rows on train-role
  chips, and 115 + 201 = 316 if every role-less row had become a training row.
- What was built (chips, not rows): **120 FNU training chips** (110 on national-2 train chips + 10 on added chips;
  117 at load), 41 NTRU, 135 SSC (134 at load). So the FNU training gain is 120 chips, not 316 and not the 365 of
  v0.3. No text may quote the v0.3 counts as the training gain.
- Role-less recovered rows: 1,079 (v0.3: 1,156), on the 763 role-less chips above (v0.3: at most 1,043).
- Uncensored recovered rows on non-test chips rejected by bounds: FNU 1, NTRU 1, SSC 2.
- A row can move between groups (for example the two censored-turbidity rows on val / val_n chips against one in
  v0.3) because the surface-only results of a sample can differ in censoring from all its results.
- Turbidity units under the rule in the ±7 d build: NTU 3,331 rows, FNU 402, NTRU 314 (309 + 5 censored).

## (iii) The new national label version

| Item | Value |
|---|---|
| Version | `national-2b` |
| Pin file | `docs/preregistration/lite_split_national2b.json` |
| Pin file sha256 | `03f0114a8683aa8fcc0d2b49901dc580c376ec3fdcdd2a8f41d1ee2242395663` |
| Canonical split hash (`labels2b.split_sha256`) | `29f0dd009bb69a5e36a7829d3bf5b8c7c4301f683a87b6a1b6ddd2fd1ca08092` |
| `labels_sha256` (all label columns of the version) | `f46129816768fccbe202fae11ec14a508b5d3cb9258a9538de5843dec16c34d2` |
| `labels_original4_sha256` (the four original label columns) | `c9267f2ec1de4bdb6355b61375fe0272398eace2f700a08db57ac165ff2d48ab` |
| Label table | `$SENTINEL_DATA/processed/lite_labels/labels_national2b.parquet`, file sha256 `2969c729dc96dc9f796deb2c8a108f9e5e8d400ec987d9662a24c400917227e4` |
| Registry row (before any run on the version) | `6bebe3649244` (M7-lite, `lite/label_version`, 2026-11-10 23:27 UTC) |
| As-built version, kept only to reproduce numbers already seen | canonical split hash `976b5be7ed7c76e9417d526f41e7b6899f88e49e3dc00b7dba077bf78f40ebe2`, `labels_sha256` `b5a61143c024fd4f867f8b42a07d1559c9deba3fc3bd5e50c752db3f210d6825` (reproduced from the builds before the rule was applied) |

## (iv) The rebuilt P2 index

- Index sha256 **`5c404be9251a22ae1b1d082446879e1466fa1c3fd41658f1f9649230be7f31b2`** (pinned in
  `sentinel/models/litev2/p2_index.py`; registry row `3ea898947cbd`, workstream `lite`, 2026-11-10 23:30 UTC).
  First-build hash `68688fac1af2c4e147070562f3dde8a035eb150360dd49759619456baf23ff44` (row `e6a469d28fe1`) is kept
  for the record only.
- Lock file `docs/preregistration/lite_p2_lock.json`, sha256
  `f3d4c63ebe503ac1774959bfcf943b45bc140883fe5b30933ee03f0cacca52be`, unchanged, 0444.
- The index stores no `test_p2` value (checked by a unit test and at every load).

| Role | Index rows (label present, NAIP image) first build → rebuild | Not eligible | **Usable chips** first build → rebuild | Water bodies | Sites | Effective water bodies | Effective sites | Label from sonde / WQP | Chips in the largest water body |
|---|---|---|---|---|---|---|---|---|---|
| `test_p2` | 206 → 204 | 17 | 189 → **187** | 38 | 74 | 10.08 | 50.75 | 155 / 32 | 46 |
| `val_p2` | 162 → 162 | 9 | 153 → **153** | 21 | 70 | 2.72 | 50.34 | 142 / 11 | 91 |
| `train_p2`, before F7 | 145 → 145 | 17 | 128 → **128** | 42 | – | 22.38 | – | – | – |
| `train_p2`, after F7 (the v2 training rows) | – | – | **60** | 25 | 32 | 14.63 | 20.69 | 46 / 14 | 8 |

- Effective number = (Σn)² / Σn² over water bodies (or sites). Registry row `e89036650629`.
- All 187 usable `test_p2` chips are river/stream sites (NAIP years 2010–2023). `val_p2`: 152 river/stream, 1
  estuary. `train_p2` after F7: 56 river/stream, 4 lake/reservoir.
- The floor of section 8.6 (≥ 100 chips in ≥ 40 water bodies) is still not met: 187 chips in 38 water bodies. P2 stays
  descriptive in this round (ruling F8).

## (v) F7 removals from `train_p2` (counts; registry row `e89036650629`)

Block roles are those of every Lite split so far (4-state, national-1, national-2, and the 44 blocks first seen in
national-2b). A `train_p2` water body is removed whole when it has an index row **or a `train_p2` lock site** in a
25-km block of a val or test role (deviation 5).

| Step | Index rows | Usable rows |
|---|---|---|
| `train_p2` before F7 | 145 | 128 |
| Rows in a val-role block / in a test-role block | 13 / 13 | 13 / 12 |
| Water bodies with an index row in such a block: 8; lock water bodies with a site in such a block: 35; **water bodies of the index removed: 12** | 41 | 38 |
| Rows in the 42 blocks without a Lite role (drawn by the national-1 salted rule: 28 train, 8 val, 6 test) | 111 | 96 |
| Removed by the draw (block not train) | 38 | 30 |
| **Removed in all** | 79 | 68 |
| **Left: the `train_p2` rows of every v2 arm and comparator** | – | **60** (25 water bodies, 32 sites; sonde 46, WQP 14) |

The lock file is not changed. `p2_index.load_index` applies the removal for every caller.

## (vi) Test-S mask count, reconciled with M2's 242 (registry row `a7349f26339f`)

Rule (F4): a teacher pairing on an M2 Test-S site-date (orbit-v3 test-role lake, in-lake, not buffer, 2015-06-23 to
2023-12-31) carries KD weight 0 and supplies no teacher-derived feature. Built from keys for national-2b
(`$SENTINEL_RUNS/lite/tests_mask_national2b/flags.parquet`, sha256
`ab5baac3c96888b47fe7f2ee5b1ef247b5cfd72f4ff64396f58a3e844c22272d`).

| Role | Chips | On an orbit-v3 test-role lake (in-lake) | On a Test-S site-date | Teacher pairing masked (KD weight 0) |
|---|---|---|---|---|
| `train` | 10,679 | 1,337 | 778 | 308 |
| `val` | 1,278 | 191 | 114 | 41 |
| `val_n` | 780 | 35 | 18 | 9 |
| `test` | 1,178 | 213 | 142 | 67 |
| `test_n` | 773 | 72 | 33 | 20 |
| `test_time` | 747 | 84 | 84 | 39 |
| `test_time_n` | 426 | 18 | 18 | 16 |
| `excluded` | 17,673 | 2,328 | 1,575 | 779 |

- **308** = NAIP chips of the national `train` role whose teacher pairing lies on a Test-S site-date, over both
  teacher stores (the count on national-2 was also 308).
- **242** (M2's count) = distinct Sentinel-2 chips of the **4-state** teacher store on Test-S points, over **all**
  Lite roles.
- The two counts differ in three ways: unit (NAIP-chip pairings against distinct S2 chips), store (both stores
  against the 4-state store) and roles (train only against all roles). Counted in M2's unit, the same rule gives
  1,273 distinct S2 chips over all roles: **242 in the 4-state store** and 1,031 in the national ±7 d store. So the
  two counts agree. In M2's unit the `train` role holds 304 distinct S2 chips for its 308 pairings.
- 114 `val` and 18 `val_n` chips are on Test-S site-dates (the CIR-Lite-R gate cells exclude them); 778 `train`
  chips are on Test-S site-dates and stay in M7 training with their labels (the "777" of ruling F4 is now 778).

## (vii) Existing models re-run, and the compute ceiling restated

**Re-runs on national-2b (order 2(c)): logged re-runs, not new configurations, not counted against the cap of 8.**
Every setting, seed and code path is unchanged; only the label table differs (`labels2b.activate_rerun`). They start
only after the stamp. Registry experiment `lite_val_national2b`.

| # | Configuration (config file in `configs/lite/`) | As-built run |
|---|---|---|
| 1 | crop128 KD, `mnv4s_raw_crop128_kd03_national2.yaml` | `natstudent_mnv4s_raw_crop128_kd03_national2_20261110-004644` |
| 2 | crop192 KD, `mnv4s_raw_crop192_kd03_national2.yaml` | `natstudent_mnv4s_raw_crop192_kd03_national2_20261110-012653` |
| 3 | crop256 KD, `mnv4s_raw_crop256_kd03_national2.yaml` | `natstudent_mnv4s_raw_crop256_kd03_national2_20261110-020817` |
| 4 | SSL init, crop128, `mnv4s_raw_crop128_kd03_ssl_national2.yaml` | `natstudent_mnv4s_raw_crop128_kd03_ssl_national2_20261110-061734` |
| 5 | SSL init, crop256, `mnv4s_raw_crop256_kd03_ssl_national2.yaml` | `natstudent_mnv4s_raw_crop256_kd03_ssl_national2_20261110-115345` |
| 6 | crop256 KD + scene/lat/lon/DOY FiLM, `mnv4s_raw_crop256_kd03_ctx_national2.yaml` | `natstudent_mnv4s_raw_crop256_kd03_ctx_national2_20261110-115406` |
| twin of 3 | no KD, `mnv4s_raw_crop256_nokd_national2.yaml` | `natstudent_mnv4s_raw_crop256_nokd_national2_20261110-115426` |
| twin of 6 | no KD, `mnv4s_raw_crop256_nokd_ctx_national2.yaml` | `natstudent_mnv4s_raw_crop256_nokd_ctx_national2_20261110-123816` |
| baselines | the frozen baseline list, refitted on the corrected rows | `natbaselines_20261109-211150` |

- 3 seeds each, as built. Selection rule S is applied once to the corrected validation rows; the as-built and the
  corrected validation tables are both reported; nothing is changed between them.
- "Unchanged" has two stated exceptions, both forced by rules of the campaign: the re-runs train on **4 fewer
  chips** (the 4 `train` chips at reserved new-site sites; deviation 3), and their label table is national-2b
  (107 train chips have no original label left and so drop out of the four-target loss).
- The SSL initial weights of configs 4 and 5 (label-free SimSiam run) and the KD target file (teacher outputs on
  train-role chips, written before this round) are reused as built. No teacher inference is run for a re-run.
- No model is re-scored on a spent test role.

**Compute ceiling (replaces the estimate line of section 10; every job ≤ 2 h through the GPU lock helper).**

| Block | Runs | Estimate, GPU-h |
|---|---|---|
| v2 networks (section 6: ANCHOR-Lite, ANCHOR-Set, A1, A3, A4, A5) | 6 configs × 3 seeds | 5 (as v0.3) |
| DenseNet121 P3a + P3b, both backbone learning rates × 3 seeds (ruling F8; v0.3 had 8 runs) | 12 | 16.5 (v0.3: 11 for 8 runs) |
| Re-runs of configs 1–6 and both no-KD twins | 8 configs × 3 seeds | 5 (the as-built runs took 290 min of wall time in all, 31–41 min each) |
| No-KD twins of slots 7 and 8 (ruling F5) | 2 configs × 3 seeds | 2 |
| S10 TabPFN: memory test on training rows, then one model per target × 3 seeds | – | 2 |
| S11 GBM-stacked: pooled embeddings of the config 3 network (PCA and GBM on CPU) | 3 seeds | 0.5 |
| GBMs (S1–S9, P1, P2a/b, refitted baselines) | – | 0 (CPU, < 2 CPU-h) |
| **Sum of estimates** | | **31** |
| **Hard ceiling for this round, restated** | | **45 GPU-h** (v0.3: 30) |

- The gated or deferred slots of section 6 (CIR-Lite-R, SparK) are not in the estimate; if one is opened it must fit
  under the ceiling.
- S10 (ruling F2, order item 6): row cap 10,000. No target has more than 10,000 uncensored training rows (largest:
  Secchi, 9,371), so the salted water-body subsample removes no row unless the memory test fails. The memory test
  (10,000 → 8,000 → 6,000 rows × 98 features on one 11 GB GPU, training rows only, no validation row loaded) is the
  first GPU job after the stamp, and its result is registered before any validation number.

## (viii) S9 target list (registry row `2e952b94162f`)

The amendment sets the minimum at l.276 (S9 row of section 5): "Run for a target only if it has ≥ 20 censored
training rows (count)". Censored training rows of national-2b under the label rule (a chip counts only if it has no
uncensored value of that target; right-censored chips with several distinct limits are not used):

| Target | Censored training rows | of these left- / right-censored | S9 |
|---|---|---|---|
| Secchi | 47 | 2 / 45 | **run** |
| TSS | 83 | 83 / 0 | **run** |
| Chl-a | 51 | 51 / 0 | **run** |
| Turbidity (NTU) | 3 | 3 / 0 | not run |
| SSC | 0 | – | not run |
| Turbidity FNU, NTRU | 0 (censored FNU / NTRU rows are not recovered) | – | not run |

**S9 runs for Secchi, TSS and chl-a only.** Any minimum from 4 to 47 would give the same list. S9 and the Tobit term
do not exist for FNU, on the national track or on P2 (P2 has 0 censored site-dates). Chl-a is a secondary target and
never enters selection.

## (ix) `test_p2` stratum "HUC8 and 25-km block hold no v2 training row" (registry row `68dfce1e2a72`)

v2 training rows = national-2b `train` chips with at least one label at load (10,591; all have a HUC8) plus the 60
`train_p2` rows left after F7. They occupy 1,589 blocks and 691 HUC8 basins.

| Stratum of usable `test_p2` chips | Chips | Water bodies | Sites | Effective water bodies |
|---|---|---|---|---|
| All | 187 | 38 | 74 | 10.08 |
| **HUC8 and 25-km block hold no v2 training row** (the reported stratum) | 135 | 22 | 49 | 5.95 |
| Block holds none, HUC8 holds some | 28 | 10 | 14 | 5.6 |
| HUC8 holds none, block holds some | 7 | 3 | 3 | 2.33 |
| HUC8 and block both hold training rows | 17 | 6 | 8 | 4.07 |

Every `test_p2` chip has a HUC8. Counting national-2b training rows alone gives the same 135 chips. Every P2 number
is reported for all 187 chips and for this stratum, with the words "unseen water bodies", not "unseen basins".

## (x) Erratum for the spent 4-state look of 2026-11-09 (counts only)

The single look of the 4-state track (negative result for H-G6) was scored on as-built labels. Under the label rule, of its
1,178 `test` chips, 7 of 310 chl-a labels and 1 of 92 turbidity labels have no QC-passed surface result (Secchi 0 of
1,094, TSS 0 of 98; no chip loses every label), and of its 747 `test_time` chips no label is affected. In sample
rows of the 4-state test roles, 21 `test` rows and 11 `test_time` rows have no QC-passed result, and 11 and 2 rows
are kept with a different number of results. Of the training labels of those models, 12 of 994 chl-a labels and 0
of the other targets would drop. The results of `lite_test_results.md` stay reported as they are; no spent test row
was or will be re-scored without the lead's authorisation, and no value was compared to make these counts.

## Power statements restated on the corrected counts (section 8.6 of the amendment)

- **P2 counts:** `test_p2` 187 usable chips (v0.3: 189), 38 water bodies, 74 sites (75), sonde 155 / WQP 32 (34),
  largest water body 46 chips (48), effective number of water bodies 10.08; stratum of (ix): 135 chips, 22 water
  bodies, effective 5.95. `val_p2` 153 chips, 21 water bodies, effective 2.72 (nothing is selected on it; F8).
  `train_p2` 60 chips after F7 (v0.3: 128), 25 water bodies, effective 14.63.
- **Reference comparison.** The half-width of about 22 MdSA points in section 8.6 came from the as-built national
  validation turbidity comparison (615 chips, 162 water bodies). On the corrected labels that comparison has 597
  chips in 155 water bodies (effective 29.86). Its half-width on the corrected labels does not exist yet (no
  validation number before the stamp) and will be reported with the re-runs.
- **Expected precision on P2**, by the same scaling as v0.3 (half-width ∝ 1/√water bodies, constant from the
  as-built comparison): about ±45 points at the 38 test water bodies and about ±88 at the effective number of 10; in
  the stratum of (ix) about ±60 at 22 water bodies and about ±115 at the effective number of 6. The v0.3 values
  "±44 at 40, ±28 at 100, ±22 at 160" describe counts P2 does not have.
- **So P2 can show only a very large difference, and a null result will not show equivalence.** It is descriptive in
  this round and earns no label (F8). Every P2 number carries "rivers and streams only", 38 test water bodies and
  the effective number 10.
- **National validation rows on the corrected labels** (chips with a present label; counts only):

| Target | `val` chips / water bodies | `val_n` chips / water bodies | val ∪ val_n chips / water bodies / effective water bodies | As-built val ∪ val_n chips |
|---|---|---|---|---|
| Secchi | 1,163 / 370 | 365 / 121 | 1,528 / 491 / 91.99 | 1,528 |
| Turbidity (NTU) | 130 / 26 | 467 / 129 | 597 / 155 / 29.86 | 615 |
| TSS | 160 / 36 | 291 / 100 | 451 / 136 / 29.39 | 464 |
| Chl-a | 422 / 130 | 300 / 108 | 722 / 238 / 43.58 | 792 |
| Turbidity FNU | 2 / 2 | 13 / 4 | 15 / 6 / 2.92 | 0 (not built) |
| SSC | 0 / 0 | 41 / 13 | 41 / 13 / 7.34 | 0 (not built) |

  FNU validation (15 chips in 6 water bodies) is below any useful size and SSC validation exists on `val_n` only; N2
  of `lite_national_protocol.md` (≥ 10 chips per target, ≥ 5 water bodies for a paired comparison) applies unchanged.
- **Not recounted here:** the event counts of N3 of `lite_national_protocol.md` (positives of turbidity > 50 NTU and
  of TSS on `val_n`) need validation label values. They are recounted by the national scorer after the stamp, under
  N3's rule unchanged.

## Conflict C2 and the rule on Sentinel-2-input models (lead rulings of 2026-11-10 15:05, 18:26 and 18:45 EST; M2 stamp)

New text only; nothing frozen is edited by it. The rule is binding on M7 through the M2 stamp (registry row
`51a841e48437`; `docs/preregistration/orbit_arch_v2_lead_rulings.md`, R-1 and C-1):

> Until M2S/test is sealed, no workstream, the M7 national look included, scores a Sentinel-2-input model on a row of
> an M2 test-role water body dated ≤ 2023-12-31. Conflict C2 applies to M2 as written for M7.

- **What happened (C2).** The spent 4-state look of 2026-11-09 scored the Sentinel-2-input Lite-safe teacher on rows
  of M2 test-role lakes. Its six registry rows (`b41eab495b6c`, `bc70b284498d`, `b9c6bb9ffafe`, `d89d96326ac8`,
  `d544efe3f909`, `b9f03a45c672`) stay as they are and are disclosed in every M7 and M2 write-up.
- **The keys-only list.** `$SENTINEL_RUNS/lite/counts_arch_v2/c2_teacher_scored_m2_test_lake_keys.csv`, sha256
  `4786aa121eb673ced502d487fb4af714fff37027677677048d2024c960cfc699`, 110 data rows (a superset by keys: 106 in the
  Test-S window and 4 buffer rows; no label, prediction or error column), with its `.meta.json` (sha256
  `ea2b0b11bc0d804ef2701e9456a84e0050928769de997357d52242d35c2d5d94`). Both files are 0444. The csv is pinned in the
  M2 stamp; M7 does not edit, regenerate or move it. A corrected list, if ever needed, is a new file, announced to
  the lead first. M2 reads this list and nothing else of M7. It was written from keys files; no look prediction file
  was opened for it.
- **How M7 keeps the rule.**
  1. No v2 arm, ablation, comparator or re-run runs a Sentinel-2-input model on any row. The students and every
     comparator take NAIP chips (and scene / location context) as input. KD uses the teacher-target file written
     before this round (teacher outputs on train-role chips only); no teacher inference is run in this round (§6 of
     the amendment: "no new teacher target is computed, none on any P2 chip"). S8 and P3b learn the teacher mean as
     an auxiliary target from that file and never call the teacher.
  2. In every v2 arm, ablation and comparator a teacher pairing on a Test-S site-date carries KD weight 0 and
     supplies no teacher-derived feature (ruling F4; 308 train pairings, section (vi)).
  3. The validation and comparison code of the national track scores students and baselines only; it has no teacher
     row.
  4. The teacher step of the national look (`scripts/eval/national_lite_score_test.py`) drops every row of an M2
     test-role water body dated ≤ 2023-12-31 (`tests_mask.teacher_score_blocked`: buffer rows of the water body
     included, no lower date bound, a row without a date is blocked) and hard-fails if one reaches the teacher call
     (`tests_mask.assert_teacher_scorable`). Unit tests cover both. This stays until the lead says M2S/test is sealed.
  5. The CIR-Lite-R gate (Sentinel-2 water cells) is not run in this round; if it is ever opened, its cells exclude
     the same rows.
  6. No M7 output is an input to anything scored on Test-S, and no Lite output is an input to an M1, M2, M6 or Fuse
     model scored on its own test roles (rulings F4, F10).
- NAIP-input students are trained and validated on chips of those water bodies (778 `train`, 114 `val` and 18
  `val_n` chips on Test-S site-dates), as ruling F4 allows; that is not a score of a Sentinel-2-input model.

## Registry rows of the stamp package

| Row | Experiment | Content |
|---|---|---|
| `6bebe3649244` | `lite/label_version` | national-2b pin and hashes |
| `3ea898947cbd` | `lite/p2_index` (workstream `lite`) | rebuilt index, 187 usable `test_p2` chips |
| `e89036650629` | `lite/p2_f7` | F7 removals, 60 `train_p2` rows left; usable rows and effective water bodies |
| `a7349f26339f` | `lite/tests_mask` | 308 masked train pairings; 242 in M2's unit |
| `68dfce1e2a72` | `lite/p2_stratum` | 135 of 187 chips; reserved-site counts |
| `2e952b94162f` | `lite/s9_targets` | S9 list: Secchi, TSS, chl-a |
| `e7ebd23c8ede` | `lite/pretrained_weights` | DenseNet121 IMAGENET1K_V1 file sha256 |
