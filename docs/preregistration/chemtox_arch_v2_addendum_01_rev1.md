# M4 ChemTox arch-v2 — addendum 01, revision 1: the guarded readers

**Dated 2026-11-10. A separate file (C-7).** It **replaces `chemtox_arch_v2_addendum_01.md` in
full**. That file, sha256 `9a195e643c420c39b7fd4451053fd3f482e738441d3ca626e85659cd342c9f2e`, is
kept byte for byte as the text the independent review read, and is **not** registered (G-12).

**Status: proposed.** It takes effect only when the lead reviews it with the guard and the tests and
writes the pins file and the registry row that carries its sha256 (C-2, C-7, G-11).

**Parents, in order of authority.** The stamped amendment (`fb52cd4b…293b`) and the stamped lead
rulings (`553be413…f060`) win over everything here. The lead's guard rulings
(`chemtox_arch_v2_lead_guard_rulings_01.md`, sha256
`63b7184c6733c6ed62382c6f66b7098bdd792beb98705d0ce1d52b2424fa1a9a`) win over this file. Ruling
**N2-A** is recorded in the annex of that file.

---

## 1. Every change from the reviewed text

The independent review of 2026-11-10 20:10 EST returned `fix_then_rereview` on findings F1–F14.
This revision changes the following, and nothing else.

| # | Change | Why |
|---|---|---|
| 1 | **"N2's list becomes three" → FIVE sanctioned readers**, named in §3: `load_training_pool`, `build_protected_labels`, `load_test_block`, `load_auxiliary_rows`, `load_h4_rows`. | F10: the reviewed text said "three" while item 3 named two functions, the guard docstring said "four", and `load_test_block` — which also reads `ecotox_aquatic.parquet` — was in neither N2 nor the addendum. |
| 2 | **The claim that the H4 counts came from "the same frozen loader" is struck and corrected** (§5). | F3: false. The frozen `auxiliary_rows()` **raises** on H4 (`arch_v2.py:592-593`); the H4 counts came from the old report script, which applied **four** steps, not seven. |
| 3 | **H4 now applies all seven exclusions of amendment §10.2**, in the stamped order, each with its own counter, and its re-check is separate code (§4). | F3 / G-3. The reviewed text said "the seven" while the reader applied four. |
| 4 | **The pre-stamp H4 counts 10,050 (T1) and 2,984 (T6) are marked SUPERSEDED** and are not quoted as training-row counts anywhere (§5). | G-3: they were measured with four steps. H4 is re-measured once, through the guard, after the pins exist. |
| 5 | **Each reader's returned columns and refused targets are stated** (§3.1). | F4: the reviewed text said nothing about what the readers return, and the H2/H3 reader in fact returned **no target at all**. |
| 6 | **The guard now joins the auxiliary target itself**, for exactly the ids it returns, after the re-checks; **H5 comes from `load_training_pool()` rows only** and has no supervised target (§3.2). | F4 / G-4. |
| 7 | **"pinned module constants" is corrected**: the four inputs that were *not* pinned — `prospective.parquet`, the ADORE split assignments, the raw `results.parquet` and `tests.parquet` — are now pinned by hash and written into every report (§6). | F7: the phrase was inaccurate for exactly those four. |
| 8 | **The gate is described as requiring the lead**, not merely as requiring a file to exist (§2). | F1: a two-key JSON any workstream could write passed `verify_freeze()`. |
| 9 | **The environment switch `M4_ALLOW_DIRECT_ECOTOX_READ` is removed** (§7). | G-10: the 7 test items it opened stay in the file and skip **always**, with the reason. |
| 10 | **The unguarded importable readers are named** as breaches for T1/T6 work (§7). | F10 / G-10. |
| 11 | **Reports are never overwritten** (§6). | F8: the reviewed text promised a report "next to the pool outputs"; it was silently rewritten on every call. |
| 12 | **The single-look rule is stated to survive a different runs root** (§8). | F5: a copy of the lead's FREEZE in a fresh `SENTINEL_RUNS` allowed a second look. |
| 13 | **The protected-labels artefact is write-once and hash-verified** (§8). | F6: it was trusted by path; a planted or swapped file was served. |
| 14 | **The pre-rewrite v2 H4 script could NOT be recovered byte for byte** and is not restored (§5). | G-12: it is not rewritten from memory. |

Three small corrections the review also noted (F13), made here: the reviewed file's mode was 0664,
not 0444; its "Dated 19:40" was a minute after its mtime; and "the four tests" of its item (c) were
four functions covering seven parametrised items.

---

## 2. The stamped text, and the collision

Ruling **N2** of the stamped lead rulings:

> **N2 [note].** 'Physically cannot' is read as 'does not through the guard'; any read of
> `ecotox_aquatic.parquet` for T1/T6 work outside `load_training_pool()` and
> `build_protected_labels()` is a protocol breach that voids the block concerned.

Amendment §10.2 simultaneously requires the auxiliary row sets H2, H3, H4 and H5 to be built, with
seven counted exclusions, for `ted`, R2, R3, `mmoe`, A4 and A5; §9 admits ablation A5 on the
BCF-family frame.

`arch_v2.auxiliary_rows()` reads `ecotox_aquatic.parquet` **directly**, and
`scripts/eval/chemtox_arch_v2_h4_report.py` read it and the two raw release tables directly. Neither
was a sanctioned reader, so under N2 running either was a breach. Ruling **B1** fixes `arch_v2.py`
at sha256 `39ecc9cc39fb1c945a843ffa8e99d38eb73c3dce378d24d8a4566550e64fe980`, so the reader cannot
be changed inside it. As stamped, therefore, the auxiliary and ablation row sets the amendment
requires **could not lawfully be built**.

**The gate itself.** `verify_freeze()` now requires the **lead**, not merely a file: the pins must
carry no write bit, a `chemtox/prereg_snapshot` row with `extra.written_by == "lead"` must carry
`extra.pins_sha256` equal to the sha256 of the pins on disk, every pinned key must be present with a
64-hex value, and no unknown entry or null value is tolerated but the protected-labels path before
the first look (G-1). M4 writes neither the pins file nor any row of that experiment.

---

## 3. The change: five sanctioned readers

**N2's list of sanctioned readers of a protected table for T1/T6 work becomes five**, all in
`sentinel/models/chemtox/arch_v2_guard.py`:

1. `load_training_pool()` — pool rows;
2. `build_protected_labels()` — materialises the sealed targets, write-once;
3. `load_test_block()` — the only reader of a sealed block;
4. **`load_auxiliary_rows(target, track)`** — H2 and H3;
5. **`load_h4_rows(track)`** — the BCF-family frame for ablation A5.

All five refuse unless `verify_freeze()` passes, so none can run before the lead writes the pins
file (S2, C-2). Numbers 4 and 5 additionally fix every path to the pinned constants — no
caller-supplied `ecotox`, `prospective`, `registry`, `splits`, `cluster_table`, `split`, `results`,
`tests`, `out` or `root` is accepted — re-check all seven exclusions in code separate from the
exclusions themselves, and write an exclusion report that cannot be overwritten.

### 3.1 What each reader returns, and what it refuses

| Reader | Rows | Target returned | Targets refused |
|---|---|---|---|
| `load_training_pool` | `block == "train_pool"` only | `log10_conc_mol_L` when asked for | — |
| `load_test_block` | one sealed block, after authorisation | `log10_conc_mol_L` plus the artefact's `*_protected` copy | — |
| `load_auxiliary_rows("H2"/"H3", track)` | the frozen loader's survivors, after the re-check | **`log10_conc_mol_L`**, joined by the guard for exactly the returned `result_id`s | `log10_bcf`; and the reader refuses `"H4"`, `"H5"` and any other name |
| `load_h4_rows(track)` | the BCF family, after all seven exclusions and the re-check | **`log10_bcf`** (its own target, from the raw table) | `log10_conc_mol_L` |
| `load_h5_pool_pairs()` | `load_training_pool()` rows only | `log10_conc_mol_L` | every auxiliary row; it takes no arguments |

`load_auxiliary_rows` raises if any returned row has no target in the pinned frame, so a silent
missing label is impossible.

### 3.2 H5 (the acute-to-chronic ratio)

Amendment §10.1 makes H5 **a reported output, never a §15.1 label**, and G-4 fixes that it is
computed from `load_training_pool()` rows only: **no guarded H3 row and no other row enters it.**
The pool is the acute endpoint, so **there is no supervised ACR target and none is fabricated**: the
candidate emits `log10_acr` (non-negative by construction, see `arch_v2_timelaw.AcuteChronicHead`)
and it is reported with an interval. `load_h5_pool_pairs()` returns the pool's (chemical, species)
keys with the acute target and takes no arguments, so auxiliary rows cannot be passed in.

---

## 4. The seven exclusions, and the re-check

Amendment §10.2's list, in the stamped order, applies to H2, H3, H4, H5 and any future auxiliary
target. For H2 and H3 the frozen `auxiliary_rows()` applies them; for H4 the guard applies them
itself, each with its own counter:

1. T2's reserved classes — for H4 this needed a **class join** the old reader lacked (species
   number → `class`, from the pinned frame);
2. spent prospective `result_id`s — **added for H4**;
3. spent ADORE `result_id`s — **added for H4**;
4. chemistry in a held-out hold-out unit;
5. chemistry absent from the split (its relation to the hold-out is *unknown*, so it is excluded);
6. a row sharing a sealed cell;
7. on T6 only, a row outside 24–96 h.

**The sealed-cell rule for H4 is the (CAS, species) pair, not the (CAS, species, duration) triple.**
H4's `obs_duration_mean` is an *observation* duration, not the exposure duration the sealed rows
carry, so the triple would compare incomparable quantities. The pair is a strict superset of the
triple, so it removes **more** rows. The choice is recorded in every H4 report and is the one place
this revision departs from the literal wording of §10.2's item 6; it is conservative, and it is
disclosed rather than silent.

**The re-check is separate code and fails closed (G-9).** It takes no context from the caller: it
rebuilds the sealed ids, the held units, the CAS→unit maps and both spent-id lists from the pinned
files itself, so one broken derivation cannot blind both the exclusion and its check. It compares
CAS keys with a normalising key, so `' 400'`, `'400.0'`, `'0400'` and `'40-0'` all resolve to the
same chemical — the forms that slipped past the reviewed version's bare string compare. A surviving
row whose chemistry maps to **no** hold-out unit **raises**; it is not merely counted. Any hit raises
with counts only in the message.

---

## 5. The H4 counts, corrected

The reviewed text said the H4 counts "were measured before the stamp with the same frozen loader".
**That is false and is struck.** The frozen `auxiliary_rows()` raises on `"H4"`; the counts came from
the old report script, which applied **four** of the seven exclusions, and the pinned JSON report
shows those four steps.

**The pre-stamp counts 10,050 (T1) and 2,984 (T6) are therefore SUPERSEDED** and may not be quoted
as training-row counts. H4 is re-measured **once**, through `load_h4_rows()`, after the pins exist,
and the lead compares the new counts with the pinned v2 report (G-3).

**The pre-rewrite v2 H4 script could not be recovered byte for byte.** Its sha256
`5653546fbfc398fb4b2895420af3465ed49e26dd44f7fd6d394eebb2f4f36e84` is recorded in
`$RUNS/chemtox/arch_v2/SHA256SUMS_v2`, but the file was overwritten in place and a search of the
repository, the runs tree and the session scratch directory finds no copy with that hash; the
repository has no commit containing it. **It is not restored and it is not rewritten from memory.**
The v1 script is kept at `$RUNS/chemtox/arch_v2/h4_bcf_join_qc.py`. Against that v1 copy the rules
are the same except that `DIMENSIONLESS` gained `'<NA>'`, and the new steps drop the v2 report's
`shared_with_train_pool_chemicals` counter — both differences are recorded here rather than
asserted away.

---

## 6. Pinned inputs and the reports

**Every input that decides which rows come back is pinned by hash and written into every report**
(G-7): the split, the cluster table, the spent-id list, `ecotox_aquatic.parquet`,
**`prospective.parquet`**, **the ADORE split assignments**, **the raw `results.parquet`** and **the
raw `tests.parquet`**. The reader also compares the spent-ADORE set it derives from the registry
with the pinned spent-id list and **refuses on any difference**, so a partition scored since the
freeze closes every reader until the lead rules. The lead sets the four newly pinned data files 0444
when the pins are written; M4 does not rewrite them.

Pinned code gains, besides the files already covered: the H4 report script,
`sentinel/models/chemtox/prospective.py`, the three test files, this addendum, and the twelve
configuration, scorer, arm, comparator and report-template files listed in M4's report.

**No report is overwritten** (G-8). Each exclusion report is written with exclusive creation and a
counter (`aux_H3_T1.json`, `aux_H3_T1.1.json`, …) under `$RUNS/chemtox/arch_v2/aux_exclusions/`,
mode 0444, and carries the pins sha256, the time, the process id and every input hash. The H4
report script takes **no output path**.

---

## 7. Readers outside the guard are breaches

Per **G-10**, these remain importable and are **not** guarded. Using any of them for T1/T6 work is a
breach under N2 and **voids the block concerned**:

* `arch_v2.load_usable_acute`, `arch_v2.build_universe`, `arch_v2.build_split`, and
  `arch_v2.auxiliary_rows` **called directly** rather than through `load_auxiliary_rows`;
* `aux.ecotox_auxiliary`;
* `prospective.curate_ecotox`, `prospective.augmentation_rows`.

The environment switch `M4_ALLOW_DIRECT_ECOTOX_READ` is **removed**. The seven test items it used to
open stay in `tests/test_chemtox_arch_v2.py` as the record of what they asserted and **skip
always**, with the reason. What they asserted is now asserted by the guard's own re-check on every
real call and is tested on synthetic frames in `tests/test_chemtox_arch_v2_guard.py`.

---

## 8. The single look

**The single-look rule survives a different runs root** (G-5). `make_pins()` records the resolved
runs root and `verify_freeze()` refuses a different one; and before any sealed row is read
`authorise()` writes a claim row to the registry (experiment `chemtox_arch_v2_look_claim` — **not**
`chemtox/prereg_snapshot`, of which M4 writes none), so every later `authorise()` for that block
refuses from **any** runs root while the later steps of the same look proceed normally.

**The protected-labels artefact is write-once and hash-verified** (G-6). `build_protected_labels()`
refuses a pre-existing file this guard did not write; it records the sha256 of what it wrote in a
write-once marker and in `AUTHORISED.json`; `load_test_block()` verifies that sha256 and asserts
that the artefact's target equals the pinned frame's target for every returned `result_id`. A
legitimate second or third look reuses the recorded artefact; a planted or swapped one is refused.

---

## 9. What does not change

**No bar, no split, no block, no comparator, no statistic, no cap and no outcome wording.** In
particular: the hashes fixed by §1.1 of the stamped rulings are unchanged — `arch_v2.py`
`39ecc9cc…fe980`, `arch_v2_power.py` `08419030…d4f7a`, the amendment `fb52cd4b…293b`, the split
`ab71aace…23a3`, the cluster table `a906aac1…821f5`, the spent-id list `3fa94b68…d1d8` and
`ecotox_aquatic.parquet` `beec1f81…fc9e`. All three blocks remain **descriptive only** (F1);
nothing here creates a route to a §15.1 label. `unseen_taxa.py`, `unseen_taxa_guard.py` and
`gpboost_model.py` are untouched.

## 10. State at the time of writing

No pins file, no FREEZE file, no freeze-sha record, no `AUTHORISED.json`, no look claim and no
protected-labels artefact exist; `verify_freeze()` raises; `registry_rows()` is 0 for all three
blocks. Neither `auxiliary_rows()` nor any H4 reader has been run on real data since the stamp, and
the new guard functions are tested on synthetic frames in temporary directories only.
