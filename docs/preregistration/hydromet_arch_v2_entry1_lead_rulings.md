# M6 SENTINEL-Hydro — architecture round v2, deviations entry 1: lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10. The binding stamp time is the timestamp of the registry row
(`M6-hydromet` / `hydromet_arch_v2/prereg_snapshot`, second row of that experiment) that carries the sha256 of this
file.
**Stamps:** `docs/preregistration/hydromet_arch_v2_deviations.md` (entry 1)
- sha256 `65a1599aa604924aeb2dab5af841760983027c609f3454f24dccb9e40d02b048`
- 356 lines, 32,377 bytes, last modified 2026-11-10 19:12:38 EST.

Entry 1 is stamped **as written, byte for byte**, together with the rulings below. Where this file and the
deviations file differ, **this file wins**. Entry 1 changes the stamped amendment and its rulings
(`hydromet_arch_v2_lead_rulings.md`, sha256 `b144974a96c2140c933ccbe1e8d3247d9a1ffa0ad6663ee958237e33eb4179b0`)
only where it says so explicitly: the text corrections of its §6 (ordered by rulings L5) and the lead ruling L11
quoted verbatim in its §7, which this stamp makes binding (L11 adds a comparator and raises the bar; it amends
rulings §2.2 in the conservative direction only). Everywhere else the amendment and its rulings stand unchanged, and
nothing in this file lowers or replaces a stamped rule. The deviations file and this file are read-only (0444) from
the stamp onward. The lead changes the mode; the engineer never does.

**The stamp is partial.** It releases exactly what the head of entry 1 asks for: the pin check; the references G1,
G2, P, R-TD and R-TD-W (both label forms each); then the D1 grid, selection, seeds, ablation (a), gate G-D1a,
ablation (b) and refits; the stacks when their members exist. Inner rows only (train-role inner groups × train
block), in the order of §3 below. It does **not** release: any validation-, test-block or England number; any D2
grid run or the DO stage; any Powell-derived row or prior; Tier 2; a freeze or a look.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash | equals the hash reviewed (`65a1599a…b048`), re-computed by the lead immediately before the stamp |
| Full re-read | all 356 lines re-read by the lead on 2026-11-10 after the review, before writing this file; §8–§11 read a second time while writing §2 |
| Independent review | one independent reviewer on this exact hash; verdict `stamp_with_lead_rulings`, no blocking finding, twelve findings F1–F12. Report copied unchanged to `docs/preregistration/reviews/arch_v2_20261110/M6e1_review.json` (sha256 `a92d6b9251a6c47917af3895d861f25f3dbeeef9248db3f9ac076a7212272281`) and `M6e1_review.md` (sha256 `79a2c9c0c65d310fe1627c762cad9ab457c135f5e654b75cbaa0233532294c0c`) |
| Items the reviewer did not check | the source documents of the constants and of the R-TD and R-TD-W recipes (only entry, module, YAML and L11 were compared with each other); equality of `scan_cube` and the frozen reader on real site tables; a recomputation of the pool fingerprints; `scoring_code_sha256()`; the test statement of entry §10 for the 19:08 tree; the declarations of entry §5 (they cannot be verified from disk); smoke timings and memory figures; whether GBM, P and stack run records store the pin hash; the D2 network and its config; any code state after 19:43 EST. The stamp does not certify these items. The lead closed two of them: the scoring-code hash (next row) and the code state (row "Code moved"); §2.2 rules on the rest |
| Frozen M6 scoring code | `sentinel.models.hydromet.guard.scoring_code_sha256()` recomputed by the lead before the stamp: `5bd7d3292a29aa9beb46cd1da5e480ca58b3856eeb909dd7fd43b0e3561fd60b`, equal to entry §10 |
| A validation, test or England number on a new arm | none found. Registry at the stamp: 0 rows `M6V2/inner_v1`; 0 rows with an experiment starting `M6EA` or `M6/test_b`; 1 row `M6V2/pins_v1`; 1 row `M6V2/decisions_v1` (the train-only timing record of §2.2 F7). The counts are in the stamp row |
| Pinned artefacts | `configs/hydromet_sota/v2/pins.json` sha256 `84958df5acf2e468a8ec926a5f59d779901f50a72375188b63723c71bff1c5d9`, read-only, equal to entry §2.1 |
| Code moved after the entry was written | yes (reviewer F2): `gso_mt/runner.py`, the neural driver and the new `thermox/nets.py` were written or edited between 19:12 and 19:39 EST, all outside `pins.json`. No arch-v2 code, test or config file changed after 19:39:30 EST. The lead ran the arch-v2 tests (`tests/test_hydromet_sota_v2_*.py`, `tests/test_reserved_loaders.py`) on CPU from 19:51:18 to 19:52:45 EST: 125 tests, all passed. The sha256 of every arch-v2 code, test and config file was identical before and after the run, and is asserted again by the stamp script; the list is in the stamp row (`code_sha256_at_stamp`) |
| A bar is lowered | no. §15.1, campaign rules 1–8, the floors of Q5 and the gates of rulings §2.2 are unchanged. The rulings below add conditions only |
| Stamp guard | `gso_mt/runner.py::deviations_status` needs a row of workstream `M6-hydromet` whose experiment ends in `/prereg_snapshot`, written by a script whose name does not contain `sota_hydromet_v2`, whose `extra.sha256` maps the amendment path to the amendment's sha256 and the deviations path to the sha256 of the file as it is on disk, with the file read-only. Before the stamp the guard refuses; the stamp script asserts both states |
| Living files | none is relied on by this file |

## 2. Rulings (binding)

### 2.1 The four decisions of entry §11

**Decision 1 — N2.** Confirmed. Both N2 endpoints (T and DO) are below the G_eff floor on metadata, so both are
**descriptive, with p = 1** in the N2 family of 2 (rulings §2.1 Q5 and Q6). The status is fixed now, from the first
column of the entry's §3 table (any eligibility row); the proxy-block column changes nothing. The realised-cell check
still runs at the look; it can confirm the status or add a failure, and can never promote an endpoint. Gate G-D1b
for N2 stays **exactly as stamped** (rulings §2.2): both N2 endpoints, against every member of the track's
comparator set, a no-go final for the track. A go on N2 permits only the descriptive N2 table of the frozen model.
No N2 claim of any kind is possible in this round.

**Decision 2 — the `PIN CHANGE` route.** Accepted, with these conditions.
1. The route is inert at this stamp: the stamped file holds no `PIN CHANGE` line, so every pinned file must equal
   its pin.
2. One state per label. From the first `PIN CHANGE` line for a label onward, only the sha256 of the **last** line
   for that label is valid; the original pin and every earlier line for that label are void. The code does not
   enforce this yet (reviewer F5) and the routine sits in the pinned `gso_mt/data.py`. Therefore the entry that
   carries the first `PIN CHANGE` line also carries the fix and a `PIN CHANGE` line for `gso_mt/data.py` itself.
   Until the code enforces it, a run made on a superseded state is void by this ruling.
3. Every non-smoke run record and registry row carries `changed_under_stamp` as `verify_pins` returns it, together
   with the deviations sha256 and the id of the lead row that released the run (`assert_released` already returns
   them). This is done in the unpinned code before the first non-smoke fit (§3).
4. Every line is a counted configuration (rulings §2.3). Its entry names each run that is repeated because of it.
   A change to a D1, GBM, P or stack config after any inner number of that arm exists counts against the cap of 12.
5. A line for `sentinel/hydromet_sota/sites.py`, either `reserved.py`, `gso_mt/data.py`, `gso_mt/constants.py` or
   `configs/hydromet_sota/v2/pins.json#values` is stamped only after the lead has read the diff, and the lead's
   rulings on that entry say so.
6. Data tables, the split, the role tables and the pool fingerprint have no change route, as the entry states.

**Decision 3 — `scan_cube`.** Accepted as described in entry §9.1: an in-memory substitution for one call, no file
of the frozen or lever code changed. Conditions: every non-smoke run record names the reader used; the pool
fingerprint check of `load_v2_train(require_pins=True)` is mandatory for every non-smoke pool read, with no other
path to the pool (reviewer F8). The lever engineer is not asked for a `cube_loader=` parameter now; if the lever
loader is next changed for another reason, the parameter is added then, under a `PIN CHANGE` entry.

**Decision 4 — pin the code.** Yes. Before the first non-smoke fit the engineer writes a second pin record:
- path `docs/preregistration/hydromet_arch_v2_code_pins.json`, read-only (0444), written once. It is **not** placed
  in `configs/hydromet_sota/v2/`: every file of that directory is a pin target of `pins.json`, so a new file there
  would fail `verify_pins`;
- content: the sha256 of `gso_mt/runner.py`, `nets.py`, `trainer.py`, `criterion.py`, `gbm_ref.py`, `gates.py`,
  `gso_mt/__init__.py`, `gso_mt/synthetic.py` if any non-smoke path imports it, every `stack/*.py`,
  `thermox/__init__.py`, `thermox/nets.py`, the four train drivers, the inner evaluation script and the pin script;
- a registry row under `M6V2/pins_v1` with a metric name different from the existing row's. It is never a row whose
  experiment ends in `/prereg_snapshot`;
- every non-smoke entry point verifies the record and writes its sha256 to the run record and the registry row;
- the inner evaluation and the stacks refuse to combine runs with different code-pin sha256;
- after the record exists, an arch-v2 code file changes only through a stamped entry that names the file, the new
  sha256 and the runs repeated. The entry may propose the line format; the lead rules on it with that entry.

### 2.2 The reviewer's findings

- **F1 (the guard can be passed without the lead).** `_stamp_rows` must also require `extra.written_by == "lead"`
  and the experiment to equal `hydromet_arch_v2/prereg_snapshot` exactly. The string is not proof of authorship, so
  the control of record is this: the id of every lead stamp row is written in
  `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`, and every non-smoke run record stores
  `deviations_registry_id` and `deviations_sha256`. A `M6-hydromet` row ending in `/prereg_snapshot` that names the
  deviations file and is not listed there is an integrity incident, and every run it released is void.
- **F2 (code moved after the entry).** Recorded in §1. Entry §9.8 and §10 are out of date (the D2 nets, the
  kill-rule smoke and 125 tests exist). They are corrected in entry 2, not by editing entry 1.
- **F3 (no per-file code hash in run records).** Decision 4.
- **F4 (`load_config` accepts any `--config` path).** A non-smoke run refuses a config path that is not a pinned
  file of `configs/hydromet_sota/v2/`. With a unit test, before the first non-smoke fit.
- **F5 (holes in the `PIN CHANGE` route).** Decision 2, conditions 2, 3 and 5.
- **F6 (D2 token matched as a substring).** The `D2 RELEASED` line is matched as a whole line. Same batch as F1.
- **F7 (registry row `afc1fbaebe0d`, written before this stamp).** It is a step-time measurement of the D2 network
  on training rows (`M6V2/decisions_v1`, metric `d2_kill_rule_s_per_step`). Rulings L6 allows the D2 smoke kill
  rule on train-only data as written, so the row is allowed as a train-only timing record; it releases nothing
  and decides nothing until the D2 entry is stamped. The D2 entry lists it. If `thermox/nets.py` changes before
  that entry is stamped, the measurement is repeated and both are reported.
- **F8 (sonde-tier site list).** The pool fingerprint is accepted as the pin of that list, under the condition of
  decision 3.
- **F9.** The docstring that says "Three conditions" is corrected in the same batch. The YAML files of
  `configs/hydromet_sota/v2/` stay as they are; `pins.json` holds their hashes.
- **F10 (N2 tables).** The reviewer reproduced them exactly. Decision 1.
- **F11, F12.** Entry §9 items 1 to 7 are accepted as written, with the reader named in the run record (decision 3).
- **Sources not re-read by the reviewer or the lead.** The constants table and the R-TD and R-TD-W transcriptions
  (entry §1 and §7.1) stand on the engineer's record and its stated limits (the status words of entry §1; the PDF
  check of §7.1). A transcription error found later is corrected by a stamped entry; it is a counted change, and
  every fit that used the wrong value is repeated.
- **Pin hash in run records.** Every non-smoke run record of a GBM, P or stack run stores `pins_sha256`, as the
  neural driver's record does. A non-smoke run record without it is not valid. Same batch as F1.

### 2.3 Rules that are restated, not changed

- No validation-cell number of any arch-v2 arm, reference or stack exists or is computed before the lead's go for
  that step (rulings §2.2, §4: "Never without the lead: a validation go/no-go, a freeze file, a look").
- References first, with parity (rulings L8). The outcome of G-D1a goes to the registry before any other model's
  variant is fitted (rulings Q2).
- Compute: the lock helper for every GPU job, at most 2 h per job (`--resume` for R-TD-W as entry §7.1 states),
  never a GPU that carries another user's process, at most 6 CPU threads per CPU job, 60 GB free disk floor.

## 3. Order of work from the stamp onward

1. **One batch of code edits, unpinned files only** (`gso_mt/runner.py`, the drivers, the stack and evaluation
   code, tests): F1, F4, F6, F9, the run-record fields of decision 2 condition 3 and decision 3, and the code-pin
   check of decision 4. No pinned file is touched. A fit launched before this batch is complete is void.
2. The arch-v2 tests run with 0 failures. The engineer writes the code-pin record (0444) and its registry row and
   sends the lead the sha256 of the record and of each file in it.
3. The pin check and the **reference** fits (G1, G2, P, R-TD, R-TD-W, both label forms) may start as soon as step 2
   is sent. They need no further word from the lead.
4. The lead checks the batch against §2 and records the code-pin sha256 in the LEAD_DECISIONS file. **The first D1
   grid fit, any selection, gate G-D1a and any stack wait for that confirmation.** If the check fails, the fits of
   step 3 made on the failing code are void and are repeated after the fix.
5. Then the D1 grid, selection, seeds, ablation (a), gate G-D1a (outcome to the registry), ablation (b), refits and
   the stacks, as entry 1 and the amendment describe. Every run has a registry row; inner numbers go to
   `M6V2/inner_v1`.

## 4. Later entries

The guard hashes the deviations file as it is on disk. Any append closes the gate for **every** arch-v2 launch until
a new lead row exists. Therefore:

- The engineer does **not** append to `hydromet_arch_v2_deviations.md` (it is read-only). Entry 2 is written as a
  draft in a separate file, `docs/preregistration/hydromet_arch_v2_entry2_draft.md`.
- Entry 2 holds: the D2 entry of rulings L4 (`thermox/nets.py` sha256, the kill-rule measurement and row
  `afc1fbaebe0d`, the implementation constants, the D2 grab share and config); the corrections of entry §9.8 and
  §10; the code-pin sha256 and the list of edits of §3 step 1; any `PIN CHANGE` line it needs, with the fix of
  decision 2 condition 2 if it carries one; the `D2 RELEASED` line.
- The lead has the draft reviewed independently, appends it to the deviations file byte for byte, and writes the
  new stamp row in the same step, so that the gate is closed for seconds only. No job is launched during that step;
  jobs already running are not affected.
- The DO stage and track T-E keep their own entries (rulings L4).

## 5. What the workstream may do from the stamp onward

- §3 steps 1 to 3 at once; step 5 after the lead's confirmation of step 4.
- Write the entry-2 draft at any time. Train-only smoke runs of D2 (`smoke: true`, never logged as inner rows) are
  allowed as before.
- Never without the lead: a D2 grid run, the DO stage, any England, Powell or Tier 2 row, a validation go/no-go, a
  freeze file, a look, a change to a pinned file, a change to the deviations file.
