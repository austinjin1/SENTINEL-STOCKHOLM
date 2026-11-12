# M6 SENTINEL-Hydro — architecture round v2: lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10. The binding stamp time is the timestamp of the registry row
(`M6-hydromet` / `hydromet_arch_v2/prereg_snapshot`) that carries the sha256 of this file.
**Stamps:** `docs/preregistration/hydromet_arch_v2_amendment.md`
- sha256 `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656`
- 448 lines, 41,665 bytes, last modified 2026-11-10 12:56:57 EST.

The amendment is stamped **as written, byte for byte**, together with the rulings and clarifications below. Where
this file and the amendment differ, **this file wins**. Both files are read-only (0444) from the stamp onward.

**The stamp is partial.** It releases the US development work of §12 (references, D1, D3, and D2 subject to §3 L6
below) once the first deviations entry of §3 L2 is stamped. It does **not** release: any work on England data
(track T-E), the D2 DO stage, any use of Powell rows, Tier 2, or any look.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash | equals the hash reviewed (`a65e6eb5…e656`), re-computed by the lead immediately before the stamp |
| Full re-read | all 448 lines re-read by the lead on 2026-11-10 before writing this file |
| Independent review | two reviewers (integrity, design) and an adversarial verifier on this exact hash (workflow run `wf_5a96bec9-d7b`): `M6_review_integrity.json` sha256 `010ead67…9501`, `M6_review_design.json` `8e0020e5…537f`, `M6_verify.json` `df709ac7…3740` in `docs/preregistration/reviews/arch_v2_20261110/`. Verifier: stamp with rulings; no finding needs the author to edit the file |
| A validation, test or England number on a new arm | none found. Registry: no row with an experiment starting `M6V2`, `M6EA` or `M6/test_b`; 2 lever rows `M6SOTA/inner_v1` (inner rows, lever engineer). `configs_tried.jsonl`: every arch-v2 row is a smoke row (`smoke: true`). The counts at the stamp are in the stamp row |
| Frozen M6 code | unchanged (46 files, code hash `5bd7d329…fd60b`, re-computed by the verifier) |
| A bar is lowered | no. §15.1 and campaign rules 1–8 are unchanged; the rulings below add comparators and conditions |
| Static input count | the frozen `ealstm_ws` reference has 26 dynamic and 188 static inputs (`train_info.json` of `ealstm_ws_v1_s0/g0_s0`, read by the lead); with the 3 VAA statics and 4 solar channels the v2 set is 30 dynamic and **191** static inputs |
| Stamp guard | `sentinel/hydromet_sota/gso_mt/runner.py::stamp_status` needs a row of workstream `M6-hydromet` whose experiment ends in `/prereg_snapshot` and whose `extra.sha256` maps the amendment path to its current sha256, and the file read-only |
| Living files this file relies on (sha256 at the stamp) | `hydromet_ea_protocol.md` `cc60763f…6304` (DRAFT, changing); `hydromet_sota_protocol.md` `d25d19d7…81ac`; `hydromet_protocol.md` `55ab7182…4401`; `M6_architecture_plan.md` `ffcbb381…b5f1`. "Unchanged" in the amendment header applies to `hydromet_protocol.md`, `FREEZE_hydromet.json` and the 46 hashed files only |

**Pre-stamp incident, recorded.** Before 14:32 EST on 2026-11-10 the hourly build loaded rows dated after the train
block into memory at 366 train-role sites (330 including WY2026 dates), against §2.4 rule 2. It computed nothing
from them and saved none; the saved table ends 2021-09-30. The build now filters dates inside the file scan and
raises on any row outside. The guarantee of §0 stands with this incident recorded. The counts of §0 are superseded
by the counts in the stamp row.

## 2. Rulings

### 2.1 Answers to §14

- **Q1 (counting) — accepted as a closed list only.** A variant is outside the cap of 12 only if it differs from
  its parent by one switch of this closed list: label set (with / without grab, logger and Oliver rows), static
  schema (US set / T-E common schema), fine-tune cell. It has identical hyperparameters and an identical config
  sha256 apart from that switch, and a `variant_of` registry row before it trains. All its inner numbers are
  reported, variants are tallied separately in every table, and no choice between a parent and its variant is made
  after a validation number exists. Anything else counts against the 12. Smoke rows stay uncounted and are refused
  by the inner-result logger.
- **Q2 (G-D1a for N2) — rejected.** G-D1a is **one** gate, on the 3-seed mean of J_sonde, with one outcome for all
  tracks. J_grab is reported and decides nothing in the gate. The outcome is written to the registry before any
  other model's variant is fitted.
- **Q3 (Tier 2) — not in this round.** The training pool is Tier 1 only. It is pinned before the first post-stamp
  fit. No source or station is added after any inner number exists. The clause "unless a stamped deviation adds it
  for all models before any model has a validation number" in §2.4 is struck.
- **Q4 (England split rule).** The eligible-only rule of `hydromet_ea_protocol.md` §4 is adopted with
  **S = 0.50** (chosen from catalogue metadata only: the larger test cell), the 15 % guard and the hash order as
  written. No per-group cap. Option (c) is not needed. The split is not frozen by this file; see §2.4.
- **Q5 — confirmed.** G ≥ 15 and G_eff ≥ 8 per endpoint are floors and are never lowered. The comparator set is
  the full set of §2.2 below, every seed.
- **Q6 — 6.** T-B family of 6 as written; T-E zero-shot 6, T-E fine-tuned 6, N2 2. No endpoint is removed after the
  stamp. An endpoint that fails the minimum stays in its family with p = 1.
- **Q7 (lever and v1 models in the joint look).** A lever model is a claim candidate only if it is named as one in
  the registry **before any arch-v2 inner number exists**; then its inner predictions are always supplied.
  Otherwise it is secondary. Lever protocol §8 is aligned through that protocol's own deviations log before any
  lever validation number (lever engineer). The v1 models are analysed in the joint look exactly as frozen in
  `hydromet_protocol.md` (same models, Holm family, comparators and every-seed rule) and are reported as secondary
  with no new label. The arch-v2 families are additional. The change from a separate v1 `test_b` look to the joint
  look is recorded as a dated entry in the deviations file of §3 L2.
- **Q8 — confirmed and tightened.** Both `hydromet_ea_split.json` and `sonde_ea_split.json` must exist with
  verified sha256 before any England fit or validation number. Order: M6 split freeze, then M1's draw, then the
  first M6 England fit. The loader enforces it and drops every M1-test group for all years. The words "or the lead
  has ruled on it" in §9.1 are struck. If M1 draws no group, M1's file still exists and records the empty set.
- **Q9 (pins; novelty check).** `configs/hydromet_sota/v2/pins.json` is accepted as the pin record with the
  content of §2.3 (PIN). The dated novelty search for the diel observation operator is commissioned by the lead;
  until it exists in `docs/novelty_checks/`, no novelty wording is used.

### 2.2 Comparators, selection and claim rules

- **R-TD recipe.** R-TD's hidden size and dropout are those of Willard et al. 2024, or of Rahmani et al. 2021 where
  that paper gives none, transcribed with page and table citation in a stamped deviations entry before the first
  R-TD fit. No other value is run. Every label states that R-TD had one configuration.
- **R-TD grab share.** Fixed at 0.5. It never depends on D1's selection; `inherit_from_d1` is removed before
  `rtd.yaml` is pinned.
- **R-TD published form.** R-TD is always fitted in both label sets (with and without grab rows, 3 seeds each; the
  second is logged as `variant_of`). Both forms are comparators in every family. The no-grab form is named the
  published form.
- **G1 and G2 (the two readings of §3 and §5).** The stronger-baseline reading is binding. Every reference,
  G1 included, gets the §3 additions (the 4 solar channels and the 3 VAA statics). G2 = G1 + equilibrium
  temperature and its trailing means + DO saturation at Te + the out-of-fold stacked predictions.
- **GBM selection.** For each GBM and primary target, the grid point and form (derived or direct) with the lowest
  standardised inner-sonde MSE is used (ties: the earlier grid point, derived before direct). The instantaneous
  model for N2 is chosen by that variable's term of J_grab. The choices are written to the registry before any
  validation number. Both forms are reported.
- **Selection criterion (disclosed change 1) — accepted.** J = 0.8 · J_sonde + 0.2 · J_grab with the 30
  sample-day cap, fixed. The same code and rows score every model, lever models included.
- **One US claim model.** There is one arch-v2 claim model for the joint US look, ordered by J_sonde, and it is
  tested in both the T-B and the N2 family. (This replaces "J_grab for N2" in §10 rule 1.)
- **Comparator set (IUT).** {R-TD with grab rows, R-TD published form, G1, G2, P}, plus the references-only stack
  when D3 is the claim model. Every seed.
- **Go/no-go G-D1b.** The 3-of-4 rule of §8 (N2: both endpoints) is applied against **every** member of the
  track's comparator set, P and both R-TD forms included. The heatwave endpoints are shown in the validation
  table. A no-go is final for the track.
- **Co-scored project models.** On T-B and N2 no label is issued for an endpoint where the claim model's point
  estimate is not better than that of every v1 frozen model and every lever model scored in the same look. This is
  an added necessary condition outside the Holm families. Tables show these models side by side.
- **N2 forms.** On N2 the claim model must beat each comparator in every form that comparator has (hour-resolved
  and daily mean) and is itself also scored with its daily mean. P is named a daily-mean baseline in every N2 table.
- **N2 new stations.** Before any WY2026 value of a station new to track N is read, its COMID and 500 km flow
  connection are checked against every pinned arch-v2 training location. A connected station is excluded from N2;
  exclusions are logged as counts.
- **Minimum groups on the US tracks.** Before any arch-v2 validation number, a stamped deviations entry gives the
  metadata G and G_eff of every T-B and N2 endpoint. An endpoint below the minimum is descriptive and keeps p = 1
  in its family. A non-rejection is reported as "not detectable at the stated MDE".
- **T-D timing.** T-D is scored once, with the models of the joint T-B/N2 freeze only. The clause "or for an
  earlier US-model freeze written by the lead" in §9.4 is struck.
- **Label wording.** A label reads "better than the re-run published top-down LSTM and the listed baselines on
  <track, endpoint>", names the comparators as re-implemented here, and says "future year at previously scored
  sites" for T-B and N2. "First", "novel" or unqualified "state of the art" are used only with a dated entry in
  `docs/novelty_checks/`.

### 2.3 Data, pins and guards

- **PIN.** `pins.json` holds the sha256 of every pool table, `lever_sites.parquet`, the FINAL split, the track-N
  role table, the sonde-tier site list, `sites.py`, both reserved modules, `gso_mt/grab.py`, `gso_mt/data.py`, the
  v2 build script and every file in `configs/hydromet_sota/v2/` (D2, stack, GBM and `schema_variant.yaml`
  included). It is written once, set 0444, and its sha256 entered in a registry row before the first post-stamp
  fit. Every non-smoke entry point (neural, GBM, P, stack, evaluation) calls `verify_pins` and `assert_stamped`.
  Values that exist only in the YAML files (D1's LSTM dropout, the `implementation_constants` block) are part of
  the configuration as pinned. Any later change to a pinned value is a counted configuration and needs a stamped
  deviation.
- **Date filter.** From the stamp on, every arch-v2 reader of a US label table applies its date filter inside the
  file scan. `hourly_train.npz` as rebuilt with that filter is the table that is pinned.
- **StreamPULSE.** M6's registration is the empty set: no arch-v2 code reads any StreamPULSE row in this round,
  the D2 DO stage included, and the pool loader asserts it.
- **Powell rows.** Powell priors and K600 rows are read only at FINAL train-role, non-inner sites that are neither
  in M1's 729-site list nor among the 48 new-site overlaps, for dates up to 2021-09-30, from a table in the pinned
  pool, and only after the shared Powell role table exists with M6's use entered in it. Until then D2 uses no
  Powell-derived prior and the DO stage has no K600 rows.
- **Physical constants.** No fit and no feature build that uses a physical constant (solar geometry, equilibrium
  temperature, DO saturation, reaeration, Schmidt number) runs before the constants sit in one shared module with a
  source per value and a unit test, recorded in a stamped deviations entry. This replaces the plan's "before the
  amendment is stamped". `sentinel/sim/physics.py` is not edited. The Raymond et al. 2012 coefficients and the
  Schmidt-number polynomial are each transcribed from the source with page and table citation before any DO-stage
  fit; nothing is fitted to Powell K600 rows before that.
- **Licences.** Before the first post-stamp fit, `docs/licenses.md` has a row for every source the round reads
  (WQP with M6 added, the EA sonde archive, HydroATLAS products, the UWWTR table, ERA5-Land / Open-Meteo, gridMET,
  StreamCat, NHDPlus VAA, Powell estimates) with licence, citation and redistribution status. A source without a
  row is not read.
- **Look guards.** Each look freeze records a sha256 over the import closure of all arch-v2 and lever files used
  at the look, checked at look time by new guard code that wraps and does not edit the frozen guard. T-E gets its
  own freeze file and access check before any England test-role row is downloaded. No arch-v2 file is placed under
  the frozen globs.
- **Counts.** The counts of the pinned files and of the stamped T-E entry are binding. The figures 12,771, 359, 39
  and the Tier 1 row counts in the amendment are superseded by them.

### 2.4 England (track T-E): stamped by reference, on hold

- T-E is stamped by reference only. **No England reading is QC-summarised, aggregated, fitted or scored** until a
  stamped entry in the deviations file records: the sha256 of `hydromet_ea_protocol.md` v1.0 and of
  `hydromet_ea_split.json`; S = 0.50; and per endpoint the metadata G, G_eff and claim / descriptive status. The
  "Metadata facts" of §9.1 are superseded by that entry. The split writer is run once, by the lead.
- **Before the split freeze:** the writer and every reader hash the same canonical core (salt, share_test and
  share_val included), shown by a test on a synthetic split; the split file pins the HydroRIVERS and HydroBASINS
  layers and the UWWTR table by sha256; the preview step's default S is 0.50; the hash check is never loosened to
  make a frozen file load; the `hybas_l7`-not-null condition and the group-id rule are written into the protocol.
- **All stations get a status.** The frozen split gives a status to all 1,964 catalogue stations and lists the
  HydroBASINS L7 units of each group. An ineligible station takes the role of the eligible station it is
  flow-connected to, otherwise that of its L7 unit's group, otherwise "reserved". Ineligible stations never supply
  M6 rows. A station with test or reserved status is closed to M1 and to every other workstream for all years.
- **Thresholds.** G ≥ 15 and G_eff ≥ 8 are applied per endpoint mechanically by frozen code, on metadata before
  the split freeze and on the realised paired cell inside the look before any score is computed. S, the thresholds
  and the endpoint list never change after the split freeze. The deviations entry records counts only.
- **Power.** The MDE values of §9.1 are best-case single-comparator values under an unverified variance
  assumption. They are not a power claim. The entry restates the MDE with the frozen G_eff.
- **Static set.** The T-E static set is `ea_data.COMMON_STATIC` as in the file pinned in `pins.json` (lat, lon and
  40 RiverATLAS attributes). The figure 39 in §3 is an error.
- **Zero-shot go/no-go cell.** It is scored once, before any fine-tuning, on all England groups that are neither
  M6 test-role nor M1-test, in the EA-val block. The fine-tuned go/no-go on val-role groups is declared
  low-information. G and G_eff of both cells are written in the stamped T-E entry. Gate G-D1b is otherwise as §2.2.

## 3. Lead clarifications (binding)

- **L1 — the stamp.** The stamp is: this file's and the amendment's sha256 in the registry row, both files 0444.
  The row is written by the lead only. The git commit is deferred while the commit hold is on and is owed before
  any freeze.
- **L2 — deviations file.** Deviations and the entries this file requires go in
  `docs/preregistration/hydromet_arch_v2_deviations.md` as dated, numbered, append-only entries. Each entry is
  checked and stamped by the lead with its own registry row (`hydromet_arch_v2/prereg_snapshot`, carrying the
  amendment sha256 unchanged and the deviations-file sha256) **before** the run it affects.
- **L3 — entry 1 (gates the first post-stamp fit of any model).** It holds: (a) the shared constants module path,
  sha256, source per value and unit-test result for every constant used by the US input set, G2's features and
  D1; (b) the sha256 of `pins.json` and the licence rows; (c) metadata G and G_eff of every T-B and N2 endpoint;
  (d) the joint-look deviation of Q7; (e) the count of lever inner rows at that time and the engineer's
  declaration whether any lever inner number was read before the v2 configs were pinned; (f) the corrections of
  L5. The R-TD transcription may be in entry 1 or in a later entry; it gates only the R-TD fits.
- **L4 — later entries.** D2 (constants of the energy balance; no Powell prior unless the role table exists);
  the DO stage (Raymond coefficients, Schmidt polynomial, K600 rule); T-E (§2.4). Each before its first fit.
- **L5 — corrections of the amendment text.** "[194 → …]" in §6.1 and §6.2 reads "[191 → …]"; the static set
  itself (§3) is unchanged. Non-smoke neural runs assert 30 dynamic and 191 static inputs. "39 RiverATLAS statics"
  in §3 reads "40" (see §2.4).
- **L6 — D2.** The D2 smoke kill rule runs on train-only data as written. D2 grid runs wait for the D2 entry of L4.
- **L7 — the upstream-forcing table of the lever track.** `upstream_forcing_sota.parquet` is read by arch-v2 code
  only after the lever engineer has stated in writing that the file is complete, and with its sha256 in
  `pins.json`. The arch-v2 engineer builds no copy of it.
- **L8 — order.** References first, with parity: G1, G2 and P on CPU and both R-TD forms are fitted before or
  together with D1 and never after a candidate's validation number exists.
- **L9 — compute.** GPUs 0–9 only through the lock helper, ≤ 2 h per process, one heavy CPU job at a time with
  ≤ 6 threads, stop below 60 GB free disk. Nothing is installed into the `sentinel2` environment.
- **L10 — the code guard after this stamp.** `assert_stamped` passing on this stamp alone releases no fit. Before
  entry 1 is sent, the engineer extends the guard so that every non-smoke entry point also requires (a) a
  lead-written `M6-hydromet` `…/prereg_snapshot` row whose `extra.sha256` carries the current sha256 of the
  deviations file, with that file read-only, and (b) `verify_pins` passing. Until that code exists and its tests
  pass, no non-smoke entry point is run.

## 4. What the workstream may do from the stamp onward

1. **Now (code, tests, builds, counts; no fit beyond train-only smoke):** remove `inherit_from_d1`; gate every
   non-smoke entry point with `verify_pins` and `assert_stamped`; write the shared constants module with sources
   and unit tests; finish the reference drivers, evaluation and gate scripts; write `pins.json`; add the licence
   rows; compute the metadata G and G_eff of the T-B and N2 endpoints; transcribe the R-TD recipe; write entry 1
   and send its path and sha256 to the lead.
2. **After the lead stamps entry 1:** pin check, then G1, G2, P and R-TD (both forms), then the D1 grid, seeds,
   ablation (a), gate G-D1a (outcome to the registry), ablation (b), refits; D3 and its control; every run with a
   registry row, inner numbers as `M6V2/inner_v1`.
3. **After its own stamped entry:** D2 grid and ablations; G-DO; the DO stage.
4. **Not before the T-E entry is stamped:** anything that reads an England reading.
5. **Never without the lead:** a validation go/no-go, a freeze file, a look.
