# M6 SENTINEL-Hydro — architecture v2 handoff

**Owner:** M6 architecture-v2 engineer. **Updated:** 2026-11-11 21:01 EST. AI-assisted.

## Where things stand
- **Amendment stamped (partial):** `docs/preregistration/hydromet_arch_v2_amendment.md` sha256 `a65e6eb5…e656`;
  rulings `hydromet_arch_v2_lead_rulings.md` sha256 `b144974a…79b0` (the rulings win).
- **Deviations entry 1 STAMPED:** `docs/preregistration/hydromet_arch_v2_deviations.md` sha256
  `65a1599aa604924aeb2dab5af841760983027c609f3454f24dccb9e40d02b048`, lead stamp row `e956ed69fd42`. Binding
  entry-1 rulings: `docs/preregistration/hydromet_arch_v2_entry1_lead_rulings.md` sha256 `0c20e003…3fe9e` (four
  decisions, findings F1–F12, order of work §3). Review: `docs/preregistration/reviews/arch_v2_20261110/M6e1_review.*`.
  **Never append to the deviations file** (any append closes the gate); never chmod it.
- **Pins written once:** `configs/hydromet_sota/v2/pins.json` sha256 `84958df5…bff1c5d9` (0444, 38 files), row
  `M6V2/pins_v1` `4a3779d5f227`. Pinned files must not be edited: `gso_mt/{data,grab,solar,constants}.py`,
  `thermox/physics.py`, `scripts/data/sota_hydromet_v2_build.py`, every file in `configs/hydromet_sota/v2/` (no new
  file there), lever `sites.py`, both reserved modules, data tables. Change route (decision 2): whole line
  `PIN CHANGE: <label> sha256 <sha>` in a stamped deviations entry; the first such entry also fixes last-line-wins in
  `gso_mt/data.py` and carries a line for `data.py` itself. No `PIN CHANGE` line exists.
- **Code pinned (decision 4):** `docs/preregistration/hydromet_arch_v2_code_pins.json` (0444, written once) sha256
  `574307a10dd417c03997890bf4022f6ac346be01de6e2a906acbfc1e350c0182`, 18 files, row `M6V2/pins_v1` `7744d6502add`.
  The 18 files (`gso_mt/{__init__,criterion,gates,gbm_ref,nets,runner,trainer}.py`, `stack/{__init__,build,convex}.py`,
  `thermox/{__init__,nets}.py`, the four train drivers, `scripts/eval/sota_hydromet_v2_inner.py`,
  `scripts/data/sota_hydromet_v2_pins.py`) **must not be edited**; proposed route: whole line
  `CODE PIN CHANGE: <label> sha256 <sha>` in a stamped entry (last line per label wins; also the only way to add a
  driver, evaluation or stack module). The lead confirmed the hashes (20:57 EST, 2026-11-10); the check of the
  step-1 batch against the entry-1 rulings §2 is with an independent reviewer. Tests and `gso_mt/synthetic.py` are
  not code-pinned.
- **Guard:** every non-smoke entry point calls `gso_mt.runner.assert_released` (five conditions: stamp; lead row with
  `extra.written_by == "lead"`, experiment exactly `hydromet_arch_v2/prereg_snapshot`, amendment sha unchanged and
  the current deviations sha, file 0444; `verify_pins`; pins sha in the stamped text; `verify_code_pins`). Every
  non-smoke run writes `release.json` and its registry rows carry the release fields; `assert_combinable` refuses
  runs of another pin state or code-pin sha. F4: only pinned config paths; F6: `D2 RELEASED` whole line.
- HOLD_VAL_SCORING stays. No validation-, test-block or England number on any arch-v2 model exists. Lever run
  directories, logs and `M6SOTA/inner_v1` values are not to be opened (none read).
- Frozen M6 code hash `5bd7d3292a29aa9beb46cd1da5e480ca58b3856eeb909dd7fd43b0e3561fd60b` unchanged (12:58 EST).

## Lead files of 2026-11-11 (both 0444; where they and a message differ, the file wins)
- `docs/preregistration/hydromet_arch_v2_batch_confirmation_20261111.md` sha256 `e2914a8d…8d60a`, row
  `68544edc55b5`: **B-1 batch confirmed, D1 grid / selection / G-D1a / stacks released**; B-2 four closure files
  bind like code pins from now (`sentinel/hydromet_sota/{gbm,data,runner,__init__}.py`, hashes in the file; add to
  `code_pin_targets()` and the record with entry 2; re-derive the closure then); B-3 `CODE PIN CHANGE` form accepted
  (examples use the literal `<sha256>`; malformed fails closed; a line for `gso_mt/runner.py` only after the lead
  read the diff); B-4 `code_changed_under_stamp` compared by file class — fit-code class label list sent to the lead
  13:19 EST for approval, implement with the first `CODE PIN CHANGE` entry; B-5 `synthetic.py` stays unpinned; B-6
  five dedicated tests — DONE 13:20 EST (guard test file 29 passed; full arch-v2 set 139 passed, 0 failed, 13:22 EST).
- `docs/preregistration/hydromet_arch_v2_cpu_lead_ruling_20261111.md` (see "What is running").

## Order of work (entry-1 rulings §3) — where we are
1. Batch of edits (F1, F4, F6, F9, release record, code-pin check) — DONE 2026-11-10 ~20:30 EST.
2. Tests 134 passed, 0 failed; code-pin record + row — DONE, sent 20:42 EST.
3. References (G1, G2, P, R-TD, R-TD-W both forms; inner rows only) — RUNNING since 20:44:36 EST, mostly done.
4. Lead confirmed the batch in writing — DONE 13:17 EST (B-1).
5. D1 grid → `select --arm d1` → seeds 1–2 → `abl_nograb` → `gate_d1a` (outcome to the registry first) → ablation
   (b) → refits → D3, `ref_stack`, `ref_stack_w` → `table --rescore`. RELEASED; D1 grid launcher started 13:18 EST
   (`launch/d1_launch.sh d1 q_d1a.txt`; relaunched 13:30:14 EST, launcher PID 2743974, after a void take of GPU 5 at
   13:26 — Lite's lock held the card before its process allocated; the helper waited, nothing started, killed by PID):
   **took GPU 1 at 13:50:33 EST**: `d1_g1_s0` lock-helper PID 2774147, driver PID 2774155, training loop since
   13:59:23 (350,183 params, 4.7 GB); the lead was told. The launcher waits for a card among GPUs 0, 1, 5 that
   carries no process AND is not held in the lock helper, then runs `launch/worker_gpu.sh` pinned to that card (`SENTINEL_GPUS=<k>`), queue `q_d1a.txt` = d1_g1_s0,
   d1_g2_s0, d1_g3_s0. Then by hand: `scripts/eval/sota_hydromet_v2_inner.py select --arm d1`, queue seeds 1–2 of the
   selected point (`--point <sel> --seed 1|2`), `abl_nograb` s0–2 (`--point abl_nograb --base <sel>`), `gate_d1a`,
   `abl_indep` s0–2, refits (`--refit`), stacks (`--stack d3|ref_stack|ref_stack_w`), `table --rescore`. Report the
   first D1 fit (job, GPU, PID, start) to the lead at once.
Not released: any validation/test/England number; the D2 grid; the DO stage; Powell; Tier 2; freeze; look.

## D1 progress (2026-11-11)
- Seed-0 grid on GPU 1: `d1_g1_s0` 13:50–14:14, `d1_g2_s0` 14:14–14:33, `d1_g3_s0` 14:33–15:01 EST (one process
  each, rc 0; ~0.16 s/step for hidden 192, 0.074 for hidden 128). 9 inner rows.
- **Selection written once 15:02 EST: `selected = g1`** (lowest seed-0 J; `sota_v2/d1/selection.json`, decisions
  row `ac5c0fb24f78`, combined under one state).
- Phase 2 queue `launch/q_d1b.txt` launched 15:03 EST through `d1_launch.sh d1b` (lock-aware; GPU 1 was free):
  `d1_g1_s1` 15:03–15:22, `d1_g1_s2` 15:22–15:42, `d1_abl_nograb_s{0,1,2}` 15:42–16:29 EST, all rc 0.
- **Gate G-D1a written once 16:30 EST: FAILED** (relative reduction 0.0205 < 0.03 on the 3-seed mean J_sonde, g1 vs
  abl_nograb; `sota_v2/gates/g_d1a.json`, decisions row `29ec9b38f6f1`). Claim configuration of every model = its
  no-grab variant: D1 `abl_nograb`, R-TD/R-TD-W `published`, G1/G2 `nograb`, P unchanged; stack `label_form` nograb.
- Phase 3 DONE 20:25 EST on GPU 1: `d1_abl_indep_s{0,1,2}` (16:31–18:27; ablation (b) as defined over g1) and
  `d1_abl_nograb_s{0,1,2}_refit` (18:27–20:25; claim-form refits), all rc 0. **D1 complete: 14 runs registered.** g1 grab-form refits not queued
  (not needed by a released step; lead asked). Then: stacks `d3`, `ref_stack`, `ref_stack_w` (need g1/g2 nograb
  s0–2 and P), `table --rescore`.

## Reference results as registered (M6V2/inner_v1, 36 rows at 12:58 EST 2026-11-11)
- R-TD fixed s0–s2 and published s0–s2: done (one process each, 14–21 min). `sota_v2/rtd/`.
- R-TD-W fixed s0–s2 and published s0–s2: done (one process each, 23–68 min, 9,637,604 params; the published recipe
  early-stops at epochs 0/3/4). `sota_v2/rtd_w/`.
- G1 and G2 seed 0 with selection (`sota_v2/g{1,2}/selection.json`, decisions_v1 rows `g1.sel`, `g2.sel`); inner
  rows J, J_sonde, J_grab. `sota_v2/g{1,2}/g{1,2}_sel_s0/`.
- P (`sota_v2/p/p_fixed`): DONE 20:57:51 EST, rc 0 (4 processes: 05:55–09:31, 09:31–12:56, 12:56–17:03,
  17:03–20:58; wall 15 h 02 min; CPU 56,345 + 53,940 s measured on processes 3–4, ≈ 91,000 s estimated for 1–2;
  ~3.6–3.8 cores average). The driver checks its 100-min deadline only before a member; a member = frozen
  `Air2StreamRegional` calibration (300 iterations, 4 threads, 3.3–4 h). Lead ruling C-1: the 2 h limit binds GPU
  processes only, so no deviation; figures sent to the lead 21:00 EST for entry 2. Rows: decisions `p.fixed`
  selection + inner `p.fixed_s0` J/J_sonde/J_grab.
- Pending: G1/G2 seeds 1–2 and no-grab forms (queued behind P in `launch/q_cpu.txt`); both reference stacks (after
  G-D1a). No failed run, no missing registry row.

## What is running (kill by PID only)
- First CPU worker `cpu` (PID 2491873) finished its queue with P at 20:57:51 EST and exited; the sampler
  (`p_cpu_sampler.sh`) exited with it (`launch/p_fixed_cpu.log` holds the CPU samples). Lead decision (a), 13:25 EST: P continues
  over the 2 h cap as an authorised deviation (recorded by the lead in entry 2 and the lead file); conditions: no
  edit of the driver/baseline, 300 iterations and 4 threads untouched, wall time of every process and the arm's CPU
  time reported. Read-only sampler `launch/p_cpu_sampler.sh` PID 2651516 writes CPU seconds of every worker child
  once a minute to `launch/p_fixed_cpu.log`. Process 1 05:55:19–09:31:25, process 2 09:31:25–12:56:36 (CPU time of
  those two not recorded; calibration wall times 12599.7 s and 11901.1 s).
- **Lead CPU ruling file** `docs/preregistration/hydromet_arch_v2_cpu_lead_ruling_20261111.md` (0444, sha256
  `0d21f6dfe13107e7b12df428648c7c0d5e14d620c3854ff9fe2bf5a810b37051`, row `a15c846c1d5f`): C-1 the 2 h limit binds
  GPU processes only (P's 3.5 h CPU processes exceed no stamped limit; logging conditions); C-2 authorised deviation:
  at most two M6 CPU jobs at once, ≤ 6 threads each, second worker only while load1 ≤ 48 (stops itself and its job
  by PID above), start/end/rc/CPU seconds logged, no entry twice, released work only (no D1, no selection, no stack).
- Second CPU worker `cpu2`: `launch/worker_load.sh cpu2 q_cpu2.txt`. First start 13:06:43 EST (worker PID 2668296,
  job PID 2668303); the load guard fired at 13:11:43 (load1 51.04 > 48): job killed by PID in its data load (rc 143,
  554 CPU s, nothing fitted, no registry row), worker stopped. Restarted by hand 13:21:03 EST at load1 37.6: worker
  PID 2729045, job `g1_sel_s1` python PID 2729052. Second guard stop 13:51:04 (load1 50.70): job killed after its
  features were built (5,965 CPU s lost — GBM members are cached only when a member completes; `members/` empty).
  Third start 13:59:54 EST, worker PID 2787718; **third guard stop 14:14:04** (job killed at the features stage
  again, 1,354 CPU s lost). Not restarted: three kills, ~8,000 CPU s lost, no GBM seed finished; the lead was asked
  (15:04 EST) whether to hold cpu2 until P ends or to allow a stricter start threshold. After the gate failed the
  no-grab GBM forms became the critical path (stack members), so cpu2 was restarted 16:31 EST at load1 35.7 (worker
  PID 2869325) within C-2 with `q_cpu2.txt` reordered: g1_nograb_s0–2, g2_nograb_s0–2, then g1/g2_sel_s1–2.
  **Fourth guard kill 16:36:25** (load1 48.06). Restarted 20:26:18 EST at load1 26.6, worker PID 2920899. No GBM no-grab seed exists yet; each job ~1.5–2 h. Note for the lead: the host's load oscillates 38–51 from other
  workstreams, so a stricter START threshold (e.g. ≤ 42) at the next restart would waste less; stop stays at 48. Queue: G1/G2 seeds 1–2 and no-grab forms (10 jobs). Log
  `launch/cpu2.log` (with `cpu_s` per process). If it stops on load (rc 6 in the log), restart it by hand when load1
  ≤ 48; it never runs a finished job twice (drivers return "already done").
- The three GPU workers finished their queues (rtd 22:25, rtdw_published 22:57, rtdw_fixed 00:15 EST). Nothing of
  mine on a GPU.
- Launcher: `sentinel_runs/hydromet/sota_v2/launch/worker.sh <worker> <queue>`; queue lines `gpu <job> <nn args>`
  (lock helper, `SENTINEL_GPUS=0,1,3,4,5,6,7,8,9`, `--min-free-mb 10000`) or `cpu <job> <script> <args>`; rc 75
  reruns; other rc stops the worker; stops under 65 GB free. Logs `launch/<worker>.log`, `launch/<job>.out`.
- **GPU state 2026-11-11:** another user holds GPUs 2, 3, 4, 6, 7, 8, 9; only 0, 1, 5 free. Check `nvidia-smi`
  immediately before every launch and take only an idle GPU (set `SENTINEL_GPUS` to the idle ones).

## Entry-2 draft (written, not appended)
`docs/preregistration/hydromet_arch_v2_entry2_draft.md`, sha256 `22f463e843edd3962504fbdd011eeff08e1bdbea71ca94cb0d1987f1eb1e8f27`
(2026-11-10 20:59 EST). Holds: the D2 temperature-stage entry (nets sha `5e554c31…85d5`, physics/constants/d2.yaml
shas, kill-rule record and row `afc1fbaebe0d`, bounded parameters and implementation constants, grab share 0.5),
corrections of entry 1 §8.1/§9.5/§9.8/§10, the code-pin record and the batch, the proposed `CODE PIN CHANGE` line,
the open points (compare `code_changed_under_stamp` or not; `synthetic.py` unpinned; kill-rule smoke ran with 183
static inputs), no `PIN CHANGE` line, and the one machine-read line `D2 RELEASED: … 5e554c31…85d5` at the end.
**Frozen at that byte state (lead, 13:25 EST): do not edit it; the lead commissions its independent review and adds
the two compute deviations (P over the 2 h cap; two CPU workers) himself. Any needed change: tell the lead first.**
The lead appends it byte for byte and stamps in the same step; the lead's expected ruling is that
`code_changed_under_stamp` is compared too — that is one key in `STATE_KEYS` of `gso_mt/runner.py`, to be changed
only under a stamped `CODE PIN CHANGE` line.

## Not written
Gate G-DO and the DO stage; the validation-cell scoring script (never run without the lead); co-scoring of lever
models. A new script under `scripts/{train,eval}/sota_hydromet_v2_*.py` fails `verify_code_pins` until stamped.

## Rules that bind every step
No git commit; kill by PID only; nothing installed into `sentinel2`; GPUs only through the lock helper, never GPU 2
or any GPU with another user's process; ≤ 2 h per process (`--resume`); one heavy CPU job at a time, ≤ 6 threads;
disk floor 60 GB (`df -h /` before every build); no England reading; no Powell row; Tier 1 only; reserved blocks
never read; frozen M6 code hash must not change; run `date` before writing any time; smoke J never tabulated.
