
---

## Entry 2 — 2026-11-10, 20:57 EST

Written by the M6 architecture-v2 engineer (AI-assisted). Status: **draft for the lead; not appended, not stamped**.
This text lives in `docs/preregistration/hydromet_arch_v2_entry2_draft.md` until the lead has had it reviewed, appends
it to this file byte for byte and writes the new stamp row (entry-1 lead rulings §4). The engineer does not touch
this file.

Files this entry refers to and does not change: the amendment (sha256
`a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656`); the lead rulings (sha256
`b144974a96c2140c933ccbe1e8d3247d9a1ffa0ad6663ee958237e33eb4179b0`); this file with entry 1 only (sha256
`65a1599aa604924aeb2dab5af841760983027c609f3454f24dccb9e40d02b048`, lead stamp row `e956ed69fd42`); the entry-1 lead
rulings `docs/preregistration/hydromet_arch_v2_entry1_lead_rulings.md` (sha256
`0c20e00354fa29b16c2f7798e335bf10203951a4e635b5ed02aada08f483fe9e`); `configs/hydromet_sota/v2/pins.json` (sha256
`84958df5acf2e468a8ec926a5f59d779901f50a72375188b63723c71bff1c5d9`, 38 files, row `4a3779d5f227`).

**What this entry asks for when stamped.**
1. The D2 entry of rulings L4 for the **temperature stage** of D2 ThermOx-T (§1). The line of §9 removes the code
   refusal of non-smoke D2 runs. It does not start anything: the first D2 grid run still waits for the lead's word
   (lead message of 2026-11-10: "Still not released: … the D2 grid"), and D2's no-grab variant waits for the outcome
   of gate G-D1a in the registry (rulings Q2).
2. A ruling on the line format for a later change of a code-pinned file (§4), on the rule for combining runs (§4.3)
   and on `gso_mt/synthetic.py` (§4.4).
3. The corrections of entry 1 §8.1, §9.5, §9.8 and §10 (§2).

**Not asked for and not released by this entry:** any validation-, test-block or England number; the DO stage of D2
and gate G-DO; track T-E (own entries, L4); any Powell-derived row or prior; Tier 2; a freeze or a look.

**This entry contains no `PIN CHANGE` line and no `CODE PIN CHANGE` line.** Every file of `pins.json` and every file
of the code-pin record has its recorded sha256. The only machine-read line of this entry is the `D2 RELEASED` line of
§9.

### 1. D2 ThermOx-T, temperature stage (rulings L4, L6)

**1.1 Files.**

| File | sha256 | Pinned by |
|---|---|---|
| `sentinel/hydromet_sota/thermox/nets.py` (the network) | `5e554c3129fba9edefa22b6e5d7cfda7bae6780a8329d675d23fe2f3c33785d5` | code-pin record (§3) |
| `sentinel/hydromet_sota/thermox/physics.py` (energy balance) | `50126467d85823a56c49d54edb0ffff77aca58c7a724a70d6b04d79b26648692` | `pins.json`, unchanged |
| `sentinel/hydromet_sota/gso_mt/constants.py` (every physical number) | `fc668307b97a5299373d47bf3649b7255d2a1d6e0d5bdd127e6089f7eb4712a7` | `pins.json`, unchanged |
| `configs/hydromet_sota/v2/d2.yaml` | `11690637125659aaed5a373df503eef764d2f1c21a896cc7da5da86925982d2e` | `pins.json`, unchanged |

`thermox/nets.py` has not changed since the kill-rule measurement of §1.6 (the record holds the same sha256), so
the measurement is not repeated (entry-1 lead rulings, F7). The trainer, the loss, the criterion and the stack code
are those of D1 (`gso_mt/trainer.py`, `criterion.py`, `nets.py`; hashes in §3).

**1.2 What the network computes** (amendment §6.2; details in the module docstring of `thermox/nets.py`).
1. Static MLP [191 → 128 → 64 → 32], input noise σ = 0.5 in training, → 13 bounded static parameters (§1.4). The
   last layer starts near zero with its bias at the prior median of each parameter.
2. GRU (hidden 64) on the 30 dynamic inputs and the static embedding → daily latents: log discharge (standardised;
   supervised by the auxiliary discharge label with weight λ_Q), snowmelt share m ∈ [0, 1), groundwater modifier
   ∈ [0.5, 2], residual r ∈ [−c, c] °C with c = `residual_cap` of the grid point (0 or 1 °C).
3. Daily physics over the whole 365-day window: equilibrium temperature Te and exchange coefficient K from the
   surface energy balance (4 Newton steps, Te clamped to [−5, 45] °C) with seasonal shade and wind sheltering, for
   the site cell and for the upstream-basin forcings; source mixing T* = Σ w_i T_i over atmosphere, groundwater
   (exponential kernel of air temperature plus an offset), reservoir (exponential kernel of Te) and upstream (upstream
   Te lagged by a travel time), then melt water at 0 °C, T* ← (1 − m) T*, T* not below 0 °C; exact exponential
   integrator with k = s_K · K · 86 400 / (ρ c_p D) per day and depth D = D_ref (Q/Q̄)^b.
4. Diel sub-model on the decoded days: hourly anomaly of the equilibrium temperature from the short-wave anomaly and
   the air-temperature anomaly, linearised around the daily state, integrated over 2 periodic cycles × 24 steps; the
   second cycle with its mean removed is the 24-point curve. Daily maximum, minimum, 7-DADMax and the grab operator
   come from the curve through the routines D1 uses.
5. **DO.** The DO mean and DO diel curve are plain statistical heads on the same GRU features. There is no oxygen
   mass balance, no gas exchange and no K600 in this stage. D2 as released here is a physics model of temperature
   only; any text that describes it says so. The DO stage keeps its own entry (L4) and gate G-DO is not written.
6. **No Powell-derived prior** (rulings §2.3). `powell_prior: false` in `d2.yaml`; `make_net` raises on `true`.

**1.3 Constants of the energy balance.** D2 calls `thermox.physics.surface_drivers`, `equilibrium_temperature`,
`heat_flux` and `pressure_kpa` (the frozen pressure routine, re-exported). It holds no second copy of a formula: the
sensitivity ∂H/∂Ta of the diel sub-model is a central difference of `heat_flux`. Every number comes from
`gso_mt/constants.py`; sources and status words are those of entry 1 §1 and are not repeated with new claims.

| Constant (name in `thermox/physics.py`) | Value | Status in entry 1 §1 |
|---|---|---|
| Stefan–Boltzmann σ (`SIGMA`) | 5.670374419·10⁻⁸ W m⁻² K⁻⁴ | verified |
| Emissivity of water (`EMIS_WATER`) | 0.97 | verified |
| Albedo of open water (`ALBEDO_WATER`) | 0.08 | verified |
| Penman wind function (`WIND_FN_A`, `WIND_FN_B`) | 74.4213 W m⁻² kPa⁻¹ (= 6.43 MJ m⁻² d⁻¹ kPa⁻¹ ÷ 86 400 s), 0.536 per m s⁻¹ of 2 m wind | verified |
| Saturation vapour pressure, slope, psychrometric coefficient, ε, wind profile 10 m → 2 m | as entry 1 §1 | verified |
| Clear-sky emissivity 1.24 (e/T)^(1/7); cloud fraction and all-sky emissivity | as entry 1 §1 | secondary (Brutsaert 1975 not read); verified |
| 1 atm (`KPA_PER_ATM`), seconds per day | 101.325 kPa, 86 400 | definition |
| ρ·c_p of water (`RHO_CP`) | 4.184·10⁶ J m⁻³ K⁻¹ | **nominal, no page checked** |
| Equilibrium temperature and exchange coefficient (Edinger et al. 1968) | formula | **secondary** (abstract only) |

Two limits stay as entry 1 states them and now concern a released arm:
- ρ·c_p is a rounded textbook value with no page checked. It enters D2 only in the relaxation rate
  k = s_K · K · 86 400 / (ρ c_p D), where the scale s_K ∈ [0.3, 3] and the depth D are learned per reach. A relative
  error of ρ·c_p therefore shifts those two learned parameters and not the fitted temperature. Consequence: the
  learned depth and exchange scale of D2 are **not** reported as physical estimates.
- The equilibrium-temperature concept is cited from a paper of which only the abstract was read. D2 builds it from
  the verified terms above by Newton iteration on `heat_flux`; no number is taken from that paper.

**1.4 Implementation constants** (no literature source; fixed before any D2 fit; all in `thermox/nets.py`, none in
YAML except where stated). Bounded static parameters, `PARAMS`:

| Parameter | Lower | Upper | Prior median | Scale |
|---|---|---|---|---|
| `depth_ref_m` (depth at the window's geometric-mean discharge) | 0.05 | 5.0 | 0.5 | log |
| `depth_exp` (b in D = D_ref (Q/Q̄)^b) | 0.0 | 0.6 | 0.3 | linear |
| `shade_max` | 0.0 | 0.95 | 0.3 | linear |
| `leaf_on_doy` | 60 | 180 | 120 | linear |
| `leaf_off_doy` | 240 | 330 | 290 | linear |
| `wind_shelter` | 0.05 | 1.0 | 0.6 | linear |
| `gw_odds` (groundwater weight, odds against the atmospheric term) | 0.0 | 2.0 | 0.10 | linear |
| `gw_tau_d` | 10 | 365 | 90 | log |
| `gw_offset_c` | −3.0 | 5.0 | 1.0 | linear |
| `res_odds` (reservoir weight) | 0.0 | 2.0 | 0.02 | linear |
| `res_tau_d` | 5 | 120 | 30 | log |
| `up_odds` (upstream weight) | 0.0 | 2.0 | 0.20 | linear |
| `exchange_scale` (s_K) | 0.3 | 3.0 | 1.0 | log |

Other constants of the module: `LEAF_OFF_FRACTION` 0.5 (shade without leaves as a fraction of the maximum);
`LEAF_WIDTH_D` 10 d (logistic leaf-on and leaf-off transitions); `AIR_PEAK_HOUR` 15 h local solar time;
`TRAVEL_SPEED_KM_D` 43.2 km d⁻¹ (0.5 m s⁻¹; travel time of the upstream term = √(watershed area) ÷ speed);
`MAX_LAG_D` 5 d; `DEFAULT_LAG_D` 1 d (watershed-area input absent); `GW_MOD_LOG_RANGE` ln 2 (groundwater modifier
in [0.5, 2]); `Q_EXP_CLAMP` 3.0 (|ln(Q/Q̄)| in the depth relation); `DTA` 0.5 K (half-width of the central
difference); a 365-day calendar for the leaf dates. From `thermox/physics.py` (pinned): `TE_RANGE` (−5, 45) °C,
`N_NEWTON` 4, `PRIOR` {shade 0.3, wind sheltering 0.6}. The prior medians of shade and wind sheltering equal those of
the G2 equilibrium-temperature input (`gbm.yaml` `te_prior`); `make_net` raises if `d2.yaml` gives other values or
another `te_range`.

None of these numbers was chosen with a validation-, test-block or England value, and none was tuned on inner rows:
they were written before any D2 fit other than the train-only smoke of §1.6, whose J is not recorded in this file
and was not used to set anything. A later change to any of them is a change of `thermox/nets.py` (a counted
configuration, §4).

**1.5 Config** (`d2.yaml`, pinned, unchanged since `pins.json`).
- Grab share **0.5** at every grid point (not stated in the amendment; fixed before any fit, as R-TD's; never tuned).
- Net: `static_hidden [128, 64]`, `static_emb 32`, `static_noise 0.5`, `gru_hidden 64`, `n_newton 4`,
  `te_range [-5, 45]`, diel `{enabled, cycles 2, steps 24, days: rule}`, `mixing full`, `warmup_epochs 3` (residual
  held at 0), `powell_prior false`, `prior {shade 0.3, wind_shelter 0.6}`.
- Grid at seed 0: g1 `{residual_cap 0.0, lambda_q 1.0}` (always reported), g2 `{1.0, 1.0}`, g3 `{1.0, 0.3}`.
- Ablations on the selected point: `abl_nodiel`, `abl_nomix` (atmosphere only). Label-set variant `nograb` (closed
  list, Q1; not counted).
- Counted configurations of D2: 5. `do_stage: {enabled: false}`.
- Training settings are those of `common.yaml` shared with D1 (window 365, batch 256, 200 steps per epoch, at most
  60 epochs, patience 6). Non-smoke runs assert 30 dynamic and 191 static inputs (L5).

**1.6 Kill-rule measurement** (amendment §8; rulings L6; entry-1 lead rulings F7). Record
`sota_v2/d2/kill_rule.json`; registry row `afc1fbaebe0d` (`M6V2/decisions_v1`, metric `d2_kill_rule_s_per_step`,
written 2026-11-11T00:41:38Z, before the entry-1 stamp; it stands as a train-only timing record and releases
nothing).

| Item | Value |
|---|---|
| Run | train-only smoke `d2_g1_s0` (grid point g1), diel sub-model on every loss day, GPU 0 |
| Step time | median 0.1484 s per step (epochs: 0.1609, 0.1484) against the limit of 0.5 s |
| GPU memory | 1767.8 MB allocated at most, 2712.0 MB reserved at most |
| Size | 61 973 parameters; batch 256; window 365; 2 epochs; 50 train-role locations (the log reports 20 dense and 20 grab fit locations) |
| Decision | `all_days` — step 2 (June–September only) was not needed and was not run |
| `nets_sha256` in the record | `5e554c3129fba9edefa22b6e5d7cfda7bae6780a8329d675d23fe2f3c33785d5` (= §1.1) |

Stated as they are:
- The smoke run was made **before** `pins.json` existed (its record holds no pin hash) and with the smoke loader
  call that leaves out the lever pool's new USGS fit sites. It therefore had **183 static inputs, not 191**. The
  released network has 8 × 128 = 1024 more weights in the first layer of the static MLP (62 997 parameters). I did
  not re-measure the step time with 191 inputs. The static MLP runs once per sequence and the margin to the limit is
  a factor 3.4; I expect no change of the decision but this is an expectation, not a measurement (§8 item 4).
- The smoke run raises on a non-finite loss and on a non-finite gradient norm at every step (`assert_finite`); it
  did not raise.
- The driver reads the decision from the record for every non-smoke D2 run and refuses to run if the record was
  measured on another state of `thermox/nets.py`.
- A second smoke directory exists (`smoke/d2_g2_s0`: grid point g2 on the **synthetic** pool of 50 locations, a code
  check of the residual head; it read no real row). Neither smoke run is an inner row; no number of either is in
  the registry other than the timing row.

### 2. Corrections of entry 1 (entry-1 lead rulings, F2). Entry 1 is not edited; these paragraphs replace the named text.

**2.1 Entry 1 §9.8 now reads:** *Not written yet:* gate G-DO and the DO stage of D2; the validation-cell scoring
script (never run without the lead); the co-scoring of lever models. *Written since entry 1:* the D2 network
(`thermox/nets.py`, 19:12–19:39 EST) and the D2 kill-rule smoke (§1.6).

**2.2 Entry 1 §10 now reads:**
- Entry 1 was written with 105 tests collected (104 passed, 1 skipped, 19:08 EST). The lead ran 125 tests, all
  passed, at 19:51–19:52 EST before the stamp. After the batch of §3: `tests/test_hydromet_sota_v2_*.py` and
  `tests/test_reserved_loaders.py`, **134 passed, 0 failed, 0 skipped**, 2026-11-10 20:40–20:42 EST, CPU, with
  `pins.json` and the code-pin record in place and the real registry.
- Frozen M6 scoring code: `scoring_code_sha256()` = `5bd7d3292a29aa9beb46cd1da5e480ca58b3856eeb909dd7fd43b0e3561fd60b`
  (unchanged).
- Non-smoke entry points have run since 2026-11-10 20:44:36 EST, references only (§6). No validation-, test-block
  or England number exists for any arch-v2 model.

**2.3 Entry 1 §8.1 (guard) is superseded by the code of §3.** `assert_released` now requires five conditions:
(i) `assert_stamped`; (ii) a lead row — workstream `M6-hydromet`, experiment **exactly**
`hydromet_arch_v2/prereg_snapshot`, `extra.written_by == "lead"`, written by a script whose name does not contain
`sota_hydromet_v2` — that carries the amendment sha256 unchanged and the current sha256 of this file, with this file
read-only (the last such row is the one recorded); (iii) `verify_pins`; (iv) the sha256 of `pins.json` in the stamped
text of this file; (v) `verify_code_pins` (§3). The marker `written_by` is not proof of authorship; the control of
record is the lead's list of stamp-row ids (entry-1 lead rulings, F1), and every non-smoke run record stores the id.

**2.4 Entry 1 §9.5 is made exact:** D2's physical constants are in `gso_mt/constants.py` and `thermox/physics.py`;
its bounded parameters and implementation constants are in `thermox/nets.py` (§1.4). `d2.yaml` repeats `n_newton`,
`te_range` and the two prior medians; the code raises if `te_range` or the prior medians differ from
`thermox/physics.py`.

### 3. Code-pin record and the batch of edits (entry-1 lead rulings §3 steps 1 and 2; decision 4)

**3.1 Record.** `docs/preregistration/hydromet_arch_v2_code_pins.json`, read-only (0444), written once at
2026-11-11T01:40:12Z, sha256 **`574307a10dd417c03997890bf4022f6ac346be01de6e2a906acbfc1e350c0182`**, 18 files;
registry row `7744d6502add` (`M6V2/pins_v1`, model `code_pins`, metric `n_code_pinned_files`; the existing pin row
`4a3779d5f227` has metric `n_pinned_files`). The hashes were sent to the lead at about 20:42 EST, before any
fit was launched (first launch 20:44:36 EST). The lead re-hashed the record and its 18 files at 20:57 EST (hash check
only; the check of the batch against the entry-1 lead rulings §2 is a separate step).

**3.2 Files of the record.** "Batch" = edited in the one batch of step 1 (after the entry-1 stamp, before the
record); the other files are as they were at the entry-1 stamp.

| File | sha256 | Batch |
|---|---|---|
| `sentinel/hydromet_sota/gso_mt/__init__.py` | `a5eca79f9ea0a2685afcfb13f6eb999af8fbd3d80cd78f48e269b1516d88d7cb` | no |
| `sentinel/hydromet_sota/gso_mt/criterion.py` | `f5e3d1fd6124c739fb63e8be6233dc4e35e370e7390f2cace5dfc52995c15a74` | no |
| `sentinel/hydromet_sota/gso_mt/gates.py` | `80931f9c9f7a7a105eb8ed91864826f146ba241357af36ddefa575934bb4a0b3` | no |
| `sentinel/hydromet_sota/gso_mt/gbm_ref.py` | `fe6ab323b27d1949f70c35097150bb420cae4f0d4d6c50c8684a347acf2b7e9a` | no |
| `sentinel/hydromet_sota/gso_mt/nets.py` | `2978a534829cdb6e65078c9f8f3cdc4f53c17feaf67cee2eabbe3d541cf1e599` | no |
| `sentinel/hydromet_sota/gso_mt/runner.py` | `69f16a2ec57cac614c6c4f0c2a0023df9b351bbbb772f4b93d07189f47bdf705` | yes |
| `sentinel/hydromet_sota/gso_mt/trainer.py` | `f7569366d81ec466642d75e9bf65402a3a9096ccb456493b6745866e53f93fab` | no |
| `sentinel/hydromet_sota/stack/__init__.py` | `32aff60c075be3263f87a60a4f1f2b1d17c711e9dc3663787b2c5cdb03703290` | no |
| `sentinel/hydromet_sota/stack/build.py` | `8a6bf15422ac24b16d455f490a29255184774d974c2c7ba0caa0f4b53d83d9ee` | yes |
| `sentinel/hydromet_sota/stack/convex.py` | `baf78635105954a496c7fc1f162f19dee4abdfc150ed25ba3af348dd0007c20a` | no |
| `sentinel/hydromet_sota/thermox/__init__.py` | `1cb0b698e385559b4667a0aebf363b0fcddc534182c2a3ed76f60f100687a6f8` | no |
| `sentinel/hydromet_sota/thermox/nets.py` | `5e554c3129fba9edefa22b6e5d7cfda7bae6780a8329d675d23fe2f3c33785d5` | no |
| `scripts/train/sota_hydromet_v2_gbm.py` | `11dbee4b82bd547902cd9b94bf9ad7b602c18e1373c4386b18929b35142992d7` | yes |
| `scripts/train/sota_hydromet_v2_nn.py` | `ac25f97aae571d1a4ca67d070b801a159bf9b3627d17e25f575bdd8d47ceac99` | yes |
| `scripts/train/sota_hydromet_v2_physics.py` | `8a058f909d8cdfc0571cfeea57df417aaf9cedfe7281a40b46d278e51b9c207e` | yes |
| `scripts/train/sota_hydromet_v2_stack.py` | `6aebbe4e6e23289f1dbb0e1ba1b94fd6b141897188cc384f594156cfa61aa6fe` | yes |
| `scripts/eval/sota_hydromet_v2_inner.py` | `25475442de0593e2e0f6219aeaaa5a799a793403fcac946b2de1f0d9749d4f99` | yes |
| `scripts/data/sota_hydromet_v2_pins.py` | `4b49fa08348238f0a36491ea1dd0a9b90e37ae9fc59023c25f3297b23e333d47` | yes |

Tests edited in the same batch (not code-pinned; hashes stored in the record as information):
`tests/test_hydromet_sota_v2_guard.py` `dc8016fabbbc6c18d1a138c27813319d87306e840369dcfcfef831e7593cfb6f`,
`tests/test_hydromet_sota_v2_stack.py` `41de414f9758a20c301ff3f87a336be52ba2be490a789cc088e5fe9bd62e4ed7`,
`tests/test_hydromet_sota_v2_gso.py` `039e9a2ca8cf4ab9f2e3aa5638f1c515753fbf886adc138e354d9056fb501b1f`.
No pinned file, no config and no lever or frozen file was touched.

**3.3 What the batch changed.**
- **F1.** Stamp rows are accepted only with `extra.written_by == "lead"` and the experiment exactly
  `hydromet_arch_v2/prereg_snapshot` (`gso_mt/runner.py`).
- **F4.** A non-smoke run refuses a `--config` path that is not a pinned file of `configs/hydromet_sota/v2/`
  (`pinned_config_path`; unit test).
- **F6.** The `D2 RELEASED` line is matched as a whole line.
- **F9.** The docstring of the guard states the conditions as they are.
- **Release record (F1; decision 2 condition 3; decision 3; the pin hash in every run record).** Every non-smoke
  run directory holds `release.json` and every non-smoke registry row holds: the amendment and deviations sha256 with
  the ids of the lead rows that released the run, `pins_sha256`, `changed_under_stamp` as `verify_pins` returned it,
  `code_pins_sha256`, `code_changed_under_stamp`, and the name of the reader of the pool. GBM, P and stack runs store
  it as the neural runs do. A run that reads the pool must have read it through
  `load_v2_train(require_pins=True)` (`assert_pool_read`).
- **Code pin (decision 4).** `verify_code_pins` in every non-smoke entry point; the pin script writes the record
  once (`--code-pins`) and passes the entry conditions (i)–(iv) only, since the record cannot verify itself before
  it exists. The inner evaluation (`select`, `gate_d1a`, `table`) and the stacks refuse to combine runs without a
  release record or made under another state (`assert_combinable`, §4.3).
- **Resume.** A run resumed across processes must be in the state of its first job (pin record, pinned files
  changed under a stamp, code pin, code files changed under a stamp); otherwise it is refused and is repeated in a
  new run directory. The deviations sha256 and stamp-row id of each job are listed in `release.json` (`jobs`), so a
  run that spans the stamp of a later entry is visible as such (§7).

### 4. Later change of a code-pinned file — proposed line, for the lead to rule on (decision 4)

**4.1 Line.** The code accepts a code-pinned file whose sha256 differs from the record **only if** this file, in the
state the lead has stamped, contains a whole line of the form `CODE PIN CHANGE: <label> sha256 <new sha256>`
(`<label>` = the repository path as in §3.2; nothing else on the line). One state per label: from the first such
line for a label onward only the sha256 of the **last** line for that label is valid; the recorded hash and every
earlier line are void. This is enforced in code (`stamped_code_states`) and tested. The line is not read as a
`PIN CHANGE` line and the reverse (both readers compare whole lines).

**4.2 Scope.** The code-pin targets are the 18 files of §3.2 plus every file that matches
`sentinel/hydromet_sota/stack/*.py`, `scripts/train/sota_hydromet_v2_*.py` or `scripts/eval/sota_hydromet_v2_*.py`
at run time. A new driver, evaluation script or stack module therefore blocks every non-smoke entry point until a
stamped line names it; this is the only route to add one (for example the validation-cell scoring script, which
does not exist). The entry that carries a line names the file, the new sha256 and each run that is repeated because
of it. Every line is a counted change.

**4.3 Combining runs.** `assert_combinable` requires every combined run directory to hold a release record with
the same `pins_sha256`, the same `changed_under_stamp` and the same `code_pins_sha256` as the combining process.
`code_changed_under_stamp` (the map of `CODE PIN CHANGE` states) is **recorded** for every run and listed in the
audit record of each combination, but it is **not compared**. Reason: a stamped line that changes, say, the stack
script would otherwise void every reference fit although the fit code is unchanged, and the entry that carries the
line already names the runs repeated. If the lead prefers the strict rule (any difference of
`code_changed_under_stamp` refuses the combination), it is one comparison key in `gso_mt/runner.py`, to be made
under the first `CODE PIN CHANGE` line (§8 item 2). Until then no `CODE PIN CHANGE` line exists and the two rules
give the same answer.

**4.4 `gso_mt/synthetic.py` is not code-pinned** (sha256
`b138ea5ca8f63e673a37e66b3295184648dd8f652e7c12f2f578c4e6fbd0daf0`, stored in the record as information). It is
imported only under `--synthetic`, which every driver refuses without `--smoke`; no non-smoke path imports it
(decision 4: "if any non-smoke path imports it"). A static test checks the two import sites.

### 5. `PIN CHANGE` (decision 2)

This entry needs no `PIN CHANGE` line: `d2.yaml` and every other file of `pins.json` equal their pins
(`verify_pins` passes with an empty `changed_under_stamp` in every run record so far). The fix of decision 2
condition 2 (last line of a label wins in `gso_mt/data.py`) is therefore **not** made here; it stays owed to the
first entry that carries a `PIN CHANGE` line, together with the `PIN CHANGE` line for `gso_mt/data.py` itself and
the lead's reading of that diff (condition 5). Until then the route is inert.

### 6. State at the time of writing (2026-11-10 20:57 EST)

- Entry-1 lead rulings §3: steps 1 and 2 done and sent; step 3 running since 20:44:36 EST; steps 4 and 5 wait for
  the lead's confirmation of the code-pin sha256. No D1 fit, selection, gate or stack has been started.
- Running under stamp row `e956ed69fd42`, pin record `84958df5…`, code pin `574307a1…`, both change maps empty:
  R-TD seed 0 (grab form), R-TD-W seed 0 in both label forms, G1 seed 0. Queued behind them: the other R-TD and
  R-TD-W seeds and forms, G2, P, the GBM seeds and no-grab forms. GPUs through the lock helper only, never GPU 2;
  at most 6 CPU threads per job.
- Registry rows of this round at 20:57 EST: 0 `M6V2/inner_v1` (no fit has finished); 1 `M6V2/decisions_v1` (the
  timing row of §1.6); 2 `M6V2/pins_v1` (`4a3779d5f227`, `7744d6502add`). No row of an experiment starting `M6EA`
  or `M6/test_b` was written by this workstream.
- Not written: gate G-DO and the DO stage; the validation-cell scoring script; the co-scoring of lever models.

### 7. Note for the append of this entry

- The guard hashes this file as it is on disk. Between the append and the new stamp row every arch-v2 launch is
  refused. A job that is already running is not affected. A job (or the next 100-minute process of a resumed fit)
  that **starts** in that interval stops at the guard before it reads any data; it writes no registry row and no
  result, and its queue worker stops. I restart such a worker by hand; a word from the lead before the append lets
  me hold the queues instead.
- A fit resumed after the new stamp continues in the same run directory: the pin record, both change maps and the
  code pin are unchanged by this entry, and `release.json` lists the deviations sha256 and stamp-row id of every
  process of the run. Runs made before and after this entry's stamp are combinable under §4.3.
- This entry changes no code. If the lead's review asks for a code change before the stamp, that change needs a
  `CODE PIN CHANGE` line in this entry and the reference fits made on the changed file are named and repeated.

### 8. For the lead to decide with this entry

1. The line format and scope of §4.1 and §4.2.
2. §4.3: keep "recorded, not compared" for `code_changed_under_stamp`, or make it a comparison key.
3. §4.4: `gso_mt/synthetic.py` outside the code pin.
4. §1.6: accept the kill-rule measurement made with 183 static inputs, or have the train-only smoke repeated with
   191 before the first D2 grid run (about 10 minutes on one GPU; both measurements would be reported and the
   record `kill_rule.json` and a second timing row written).
5. §1.3: D2 is released with ρ·c_p as a nominal value and the equilibrium-temperature concept on secondary status,
   with the stated consequence that D2's learned depth and exchange scale are not reported as physical estimates.
6. When the D2 grid may start, relative to gate G-D1a (the line of §9 removes the code refusal only).

### 9. Machine-read line of this entry

D2 RELEASED: sentinel/hydromet_sota/thermox/nets.py sha256 5e554c3129fba9edefa22b6e5d7cfda7bae6780a8329d675d23fe2f3c33785d5
