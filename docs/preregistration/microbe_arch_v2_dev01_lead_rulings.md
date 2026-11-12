# M3 SENTINEL-Microbe — architecture round v2, deviations file 01: lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10. The binding stamp time is the timestamp of the registry row
(`M3-microbe` / `microbe/prereg_snapshot`) that carries the sha256 of this file.
**Stamps:** `docs/preregistration/microbe_arch_v2_deviations_01.md`
- sha256 `a3fdc1ed74468123c96e744b88a25958762487e2f5106178b50d6f9a96245a58`
- 135 lines, 9,051 bytes, last modified 2026-11-10 15:23:39 EST.

The deviations file is stamped **as written, byte for byte**, together with the rulings below. Where this file and
the deviations file differ, **this file wins**. Where this file is silent, the stamped amendment
(`microbe_arch_v2_amendment.md`, sha256 `022841f3…eb6a`) and its rulings file (`microbe_arch_v2_lead_rulings.md`,
sha256 `e148a09d…cd95`, registry row `421854a9d225`) apply unchanged. Both files are read-only (0444) from the
stamp onward. Line numbers (L…) refer to the stamped deviations file.

This file is deliberately **not** named `microbe_arch_v2_deviations_*.md`: that pattern is read by
`sentinel/models/microbe/stamp.py::release_status` as a deviations file, and this file releases nothing by itself.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash | equals the hash reviewed (`a3fdc1ed…5a58`), re-computed by the lead immediately before the stamp |
| Full re-read | all 135 lines re-read by the lead on 2026-11-10 before writing this file, with `stamp.py` (150 lines) and the gate function of `archv2.py` |
| Independent review | one independent reviewer on this exact hash (start and end of the review), verdict `stamp_with_lead_rulings`, 2026-11-10 23:33 UTC. Every finding of that review is closed by a ruling in §2 |
| A development number of an affected run | none. The registry holds no row for `niche_s_a0`, `xgb_v4clr_mo`, `niche_s_qmp` or a `calib_*` metric, and no QMP confound output or GEO power file exists (asserted by the stamp script; counts in the stamp row) |
| Development numbers that do exist at the stamp | rows of experiment `microbe_arch_v2_cv` for the tier-0 baselines on family G, released by the amendment stamp and started after the review of this file ended. The lead has not read any of their values. No ruling below depends on a development value: each one adds a comparator pair, takes the stricter figure, closes a code hole or fixes wording |
| A bar is lowered | no. No hypothesis, family, Holm denominator, cap, split or scorer changes (L28 holds) |
| Second stamp row and the amendment stamp | checked by the reviewer on a scratch registry and re-checked by the stamp script: a second `microbe/prereg_snapshot` row leaves the amendment stamped (row `421854a9d225`) and releases exactly the five keys of L10 |
| Living files (sha256 at the stamp, in the stamp row) | `sentinel/models/microbe/stamp.py`, `archv2.py`, `niche.py`, `v4mr.py`, `scripts/train/microbe_arch_v2_cv.py`, `scripts/eval/microbe_qmp_confound.py`, `scripts/eval/microbe_geo_power.py`, `tests/test_microbe_archv2.py`. They changed after the deviations file was written (L132–133 are out of date) and will change again under §2 |

## 2. Rulings (binding)

- **D1 — +QMP twins and the gates of their base model.** No +QMP twin is run before its base model's own gate is
  satisfied: the release key of the base name (for example `XGB_V4CLR_MO` for `xgb_v4clr_mo_qmp`) and, for
  `xgb_v4mr_stack_qmp`, the slot-#8 row of the lead's D1-gate evaluation. M3 changes `archv2.gate` so that it
  applies the release key and the slot-#8 check of `<base>` to `<base>_qmp`, with unit tests that cover
  `xgb_v4clr_mo_qmp` and `xgb_v4mr_stack_qmp`, **before** the confound script is run.
- **D2 — P1, holdout scaling (replaces L119–122).** SE_holdout = SE · √(1,215 / 214) for every target. This is
  the upper end of the range the amendment states (§7.3) and it bounds every target's true ratio. It is label-free.
- **D3 — P1, pairs (adds to L113–115).** Each finished entry is paired with every comparator that amendment §7.4
  names for it: `tier0_best`, `rf_clr`, `xgb_clr`, `rf_clr_tuned`, `xgb_clr_tuned`, and, for an entry that reads a
  context column (G1 and G8), also `rf_clr_context` and `xgb_clr_context`. The power figure of a target is still
  the largest SE over its pairs.
- **D4 — P1, details.** The SE of a pair is the width of the 95 % percentile interval divided by 2 · z₀.₉₇₅, as in
  the reference report (`microbe_split_lodo_x001_report.json`). An entry among G1, G3, G7, G8, G9 that is
  reported as not run (with its registry row or log entry) contributes no pair and does not block the file; an
  entry that is neither finished nor reported as not run does block it. The power script is brought into line
  with D2–D4, with tests, before it runs. It runs once.
- **D5 — R17, what the rule decides (replaces L104).** If the rule fires, every +QMP result is descriptive only:
  G10 and every +QMP twin are reported, G10 is not selectable as M\*_GEO and supports no inferential claim, and it
  still counts as a GEO entry (the cap does not move). The rule does not decide whether a +QMP model runs.
- **D6 — R17, non-finite share.** A non-finite laboratory share of `qmp_load` on either event set (missing
  laboratory, fewer than 2 groups, zero variance) means the rule has fired. M3 changes the script accordingly,
  with a test, before it runs.
- **D7 — R17, one run.** The confound script is run once, after this stamp and after the code changes of D1 and
  D6. It refuses to run if its output exists. Its first output stands.
- **D8 — R14, condition enforced.** `niche_s_qmp` is not run before the mechanical G10 choice (ruling B8) is
  logged and names G1. The code requires the registry row id of that choice, and `niche_s_qmp` is added to the
  counted models of `scripts/train/microbe_arch_v2_cv.py`, so that a re-run is refused.
- **D9 — wording of L79, L82–83 and L135.** A0 and the calibration diagnostic are prerequisites of the gate
  evaluation (ruling B6). The gate outcome is decided by the two conditions of amendment §2.6 as amended by the
  rulings file, alone. No calibration statistic is a gate condition.
- **D10 — R15, intercept.** With no context prior, the intercept step fills a missing training dimension with 0
  (the training mean after standardisation). This is what "no context prior" implies and it is accepted; L35
  ("intercept unchanged") is read with this exception.
- **D11 — who writes stamp rows.** M3 writes no row of experiment `microbe/prereg_snapshot`. M3 changes
  `stamp.py` so that only rows with `extra.written_by == "lead"` count, in `stamp_status` and in
  `_stamped_shas`, with a test. Both existing rows carry that field.
- **D12 — V1 and C1.** Accepted as written. V1 follows ruling B5 and replaces the per-family, mean-filled reading
  of the amendment. C1 selects nothing and is in no code path a gate condition.

## 3. What M3 may do from the stamp onward

- **Now:** `niche_s_a0` (key `A0`), `xgb_v4clr_mo` (key `XGB_V4CLR_MO`) and the calibration diagnostic (key
  `CALIBRATION`), on `lodo`, under the amendment's rules for counted models, seeds and reruns.
- **After the code changes of D1, D6, D7 and D11 pass their tests:** the QMP confound script, once.
- **After G1, G3, G7, G8 and G9 are finished or reported as not run, and the script matches D2–D4:** the GEO power
  file, once (key `GEO_POWER`).
- **After the B8 choice row exists and names G1:** `niche_s_qmp` (key `NICHE_S_QMP`).
- **Still not released:** D2 NicheGeo-Phys, the added EA family, H4, GAS-3 (each needs its own reviewed and
  stamped deviations file), `niche_s_doy`, and `xgb_v4mr_stack` before the lead's D1-gate row.
- A code change that alters a prediction of a model that already has a development number is not allowed under
  this file. The changes ordered above touch gates, counters and scripts that have not run.
