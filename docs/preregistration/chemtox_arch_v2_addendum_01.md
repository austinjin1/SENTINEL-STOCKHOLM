# M4 ChemTox arch-v2 — addendum 01: one more sanctioned reader for auxiliary rows

**Dated 2026-11-10 (19:40 EST). A separate file, per clarification C-7: nothing is appended to the
stamped amendment or to the rulings file.** Written by M4 under lead **ruling N2-A** of 19:39 EST.

**Parents.** `docs/preregistration/chemtox_arch_v2_amendment.md` sha256
`fb52cd4b072fa19c8c1e19b9fe1374a0e8abb71d9492eff764f2034cb26d293b` (stamped, 0444) and
`docs/preregistration/chemtox_arch_v2_lead_rulings.md` sha256
`553be413bef252e5900023dc1dc983dd8f228a5825194ca70505cf6649c7f060` (0444), which wins over the
amendment wherever they differ. This addendum is subordinate to both and changes exactly one thing.

**Status: proposed. It takes effect only when the lead reviews it with the guard and the tests and
writes the pins file and the registry row (C-2, C-7).**

---

## 1. The stamped text

Ruling **N2** of the rulings file:

> **N2 [note].** 'Physically cannot' is read as 'does not through the guard'; any read of
> `ecotox_aquatic.parquet` for T1/T6 work outside `load_training_pool()` and
> `build_protected_labels()` is a protocol breach that voids the block concerned.

Amendment §10.2 simultaneously requires the auxiliary row sets H2, H3, H4 and H5 to be built, with
seven counted exclusions, for `ted`, R2, R3, `mmoe`, A4 and A5, and §9 admits ablation A5 on the
BCF-family frame.

## 2. The collision

`sentinel.models.chemtox.arch_v2.auxiliary_rows()` reads `ecotox_aquatic.parquet` **directly**
(`pd.read_parquet(ecotox, columns=_USABLE_COLUMNS)`), and
`scripts/eval/chemtox_arch_v2_h4_report.py` reads it, `raw/ecotox/release_2026-09-15/results.parquet`
and `tests.parquet` directly. Neither is `load_training_pool()` nor `build_protected_labels()`, so
under N2 **running either is a protocol breach that voids the block concerned**. Ruling **B1** fixes
`arch_v2.py` at sha256 `39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980`, so the
reader cannot be changed inside it.

As stamped, therefore, the auxiliary and ablation row sets the amendment requires **cannot lawfully
be built**. M4 reported the collision rather than running the loader; it has not been run since the
stamp, and the four tests that exercise it are gated behind an environment variable and skipped.

## 3. The change

**N2's list of sanctioned readers of `ecotox_aquatic.parquet` for T1/T6 work becomes three:**

1. `arch_v2_guard.load_training_pool()` — unchanged;
2. `arch_v2_guard.build_protected_labels()` — unchanged;
3. **`arch_v2_guard.load_auxiliary_rows(target, track)`** and
   **`arch_v2_guard.load_h4_rows(track)`** — new, and the only route to an auxiliary or
   BCF-family row.

The two new functions are in the guard, which is a living file under B1 and is reviewed and pinned
in the same pass. They:

* **refuse unless `verify_freeze()` passes**, so neither can run before the lead writes the pins
  file (S2, C-2);
* **fix every path to the pinned module constants.** No caller-supplied `ecotox`, `prospective`,
  `registry`, `splits`, `cluster_table`, `results.parquet` or `tests.parquet` path is accepted; a
  caller that passes one gets a hard error. `arch_v2.auxiliary_rows()` is called with those
  constants, so the frozen loader is used exactly as stamped and its seven counted exclusions
  still apply;
* **re-check the three leakage properties independently of `auxiliary_rows()`' own filters**, after
  the call, and raise on any hit, reporting counts only: no returned `result_id` is a `result_id` of
  `T1_test`, `T6a_test` or `T6b_test`; no returned `(test_cas, latin_name, duration_h)` cell is a
  sealed cell; no returned `test_cas` maps to a held-out hold-out unit;
* **write the exclusion report (counts only)** beside the pool outputs, under
  `$RUNS/chemtox/arch_v2/aux_exclusions/`.

`scripts/eval/chemtox_arch_v2_h4_report.py` is rewritten to call `load_h4_rows()` and to **read no
parquet itself**; the BCF join and QC rules move unchanged into a new pure module
`sentinel/models/chemtox/arch_v2_h4.py`, which takes frames as arguments and performs no I/O.

## 4. What does not change

**No bar, no split, no block, no comparator, no statistic, no cap and no outcome wording.** In
particular:

* the hashes fixed by rulings §1.1 are unchanged — `arch_v2.py`
  `39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980`, `arch_v2_power.py`
  `0841903004361e52d0010b8be8e2c38745195e8d0629b0dd7e7c0c01243d4f7a`, the amendment, the split
  `ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3`, the cluster table
  `a906aac1f9dd97577ff473981a1fce5aae1a509e3fb2dcb42f8726dea2d821f5`, the spent-id list
  `3fa94b68ad96c9018737886430ae6d1efd3b1c52dc33b2d98b4857a87f5bd1d8` and
  `ecotox_aquatic.parquet` `beec1f81769aecad423e952c21bca2d0b3c899b2f0f40d80143fdab3cbf4fc9e`;
* the seven AUX-LEAK exclusions, their order and their counters are those of the stamped
  `auxiliary_rows()`; this addendum adds re-checks, it removes none;
* the filtered auxiliary counts of amendment §10.1 (H2 5,388 / 4,455; H3 121,678 / 45,661; H4
  10,050 / 2,984) are **not re-asserted here**: they were measured before the stamp with the same
  frozen loader, and they are re-measured through the guarded route once the pins file exists. Until
  then they are quoted as pre-stamp measurements;
* all three blocks remain **descriptive only** (F1). Nothing here creates a route to a §15.1 label;
* `unseen_taxa.py`, `unseen_taxa_guard.py` and `gpboost_model.py` are untouched.

## 5. State at the time of writing

No pins file, no FREEZE file, no freeze-sha record, no `AUTHORISED.json`, no protected-labels
artefact; `verify_freeze()` raises; `registry_rows()` is 0 for all three blocks. Neither
`auxiliary_rows()` nor the H4 report has been run on real data since the stamp. The new guard
functions are tested on synthetic frames in temporary directories only.
