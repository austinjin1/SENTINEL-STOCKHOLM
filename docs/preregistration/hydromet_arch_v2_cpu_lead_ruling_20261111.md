# SENTINEL-Hydro (M6) — architecture round v2: lead ruling on the CPU execution limits

**Written by the lead:** Wed 11 Nov 2026 1:05 PM EST.

**Status.** Read-only (0444). This is **not a stamp**: it releases no arm, no track and no number, and it changes
no recipe, constant or bound. It rules how two stamped sentences about CPU execution are read and authorises one
deviation from them. Its text is written byte for byte into entry 2 of `hydromet_arch_v2_deviations.md` when the
lead appends that entry; until then this file is the record. It is never edited; a correction is a new, dated lead
file with its own registry row.

## 0. Files the lead read before writing this, with their sha256 at writing

| File | sha256 |
|---|---|
| `docs/preregistration/hydromet_arch_v2_amendment.md` | `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656` |
| `docs/preregistration/hydromet_arch_v2_entry1_lead_rulings.md` | `0c20e00354fa29b16c2f7798e335bf10203951a4e635b5ed02aada08f483fe9e` |
| `docs/preregistration/hydromet_arch_v2_deviations.md` | `65a1599aa604924aeb2dab5af841760983027c609f3454f24dccb9e40d02b048` |
| `docs/preregistration/hydromet_arch_v2_code_pins.json` | `574307a10dd417c03997890bf4022f6ac346be01de6e2a906acbfc1e350c0182` |

## 1. The stamped text at issue

Amendment §12, first bullet, byte for byte:

```text
- GPUs 0–9 only through `python -m sentinel.core.gpu --name <job> --min-free-mb 10000 -- <cmd>`; ≤ 2 h per process
  (jobs stop at 110 min and resume); never a GPU another process occupies. CPU: one heavy job at a time, ≤ 6 threads.
```

Entry-1 lead rulings §2, compute bullet, byte for byte:

```text
- Compute: the lock helper for every GPU job, at most 2 h per job (`--resume` for R-TD-W as entry §7.1 states),
  never a GPU that carries another user's process, at most 6 CPU threads per CPU job, 60 GB free disk floor.
```

Rule 5 of `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` adds, for the whole campaign: "CPU ≤ 6
threads per job; load is ~40 of 48 cores, so prefer GPU work and keep CPU jobs sequential per workstream."

## 2. The facts, as the engineer reported them on 2026-11-11

- The physics arm P (`scripts/train/sota_hydromet_v2_physics.py`, configuration `p_fixed`) fits four members:
  inner and refit, each for `site_cell` and `basin_mean`. Its driver checks a 100-minute deadline **before** a
  member starts, never inside one, and exits with code 75 so that the next process resumes from the cached
  members.
- One member is the frozen `Air2StreamRegional` calibration: 300 Adam iterations over roughly 1,050-1,240 donor
  sites, with `threads: 4` pinned in `physics.yaml`. It takes about 3 h 20-30 min on CPU, so each process of the
  chain runs about 3.5 hours, longer than the member-start deadline can prevent.
- Measured wall clock (EST): process 1 05:55:19-09:31:25 (3 h 36 min, member inner/site_cell, calibration 12,599.7
  s, 1,056 of 1,064 donors); process 2 09:31:25-12:56:36 (3 h 25 min, member refit/site_cell, 11,901.1 s, 1,227 of
  1,239 donors); process 3 started 12:56:36 on inner/basin_mean. Two of four members are cached; nothing is lost
  if a process is killed.
- The arm uses about 1.4 of the machine's 48 logical cores. The one-minute load average at 12:45 EST was 29.2, and
  7 of the 10 GPUs carry another user's jobs, so GPU work cannot absorb the queue.
- The engineer's second CPU worker (`launch/worker_load.sh` with queue `launch/q_cpu2.txt`, 10 jobs: the G1 and G2
  seeds 1-2 and the no-grab forms) is prepared and was not started. The first worker's queue was cut after
  `p_fixed` at the byte it had consumed, so no job can run twice.

## 3. Rulings

### C-1 — the two-hour limit binds GPU processes, not the CPU physics chain (a reading, not a deviation)

The "≤ 2 h per process (jobs stop at 110 min and resume)" of amendment §12 stands inside the bullet's GPU clause,
between the lock-helper command and "never a GPU another process occupies"; the CPU clause of the same bullet sets
"one heavy job at a time, ≤ 6 threads" and no time limit. The entry-1 compute bullet reads the same way: "at most
2 h per job" is qualified by "the lock helper for every GPU job", and its CPU clause is "at most 6 CPU threads per
CPU job". **Ruling:** the two-hour per-process limit applies to GPU processes. The physics arm is CPU-only with 4
threads, so its 3.5-hour processes exceed no stamped limit, and no deviation arises. P continues; it is not killed
and not held.

The lead's message to the engineer at 13:25 EST called this a deviation from a stamped resource limit. That
characterisation was wrong and is corrected here: it is a reading of the stamped text, and this file is the
binding form. The decision itself — P continues untouched — is unchanged.

Conditions, which do not depend on the reading above: the code-pinned physics driver and the frozen baseline are
  not edited; the 300 iterations, the 4 threads and every pinned constant stay as they are; the wall-clock start
  and end and the CPU seconds of every process of the chain are logged, estimates are marked as estimates, and the
  arm's totals are written into entry 2.

### C-2 — a second CPU worker: an authorised deviation, with a ceiling

A second CPU worker running beside P does break stamped text: amendment §12 says "CPU: one heavy job at a time",
and rule 5 of the campaign decisions says "keep CPU jobs sequential per workstream". This is a deviation and is
recorded as one; it is not a reading.

**Ruling:** the lead authorises, for the architecture-v2 round of M6 only, at most **two** CPU jobs of this
workstream running at once, under all of the following, every one of which is a condition of the authorisation:

- Each job uses at most 6 threads, so at most 12 threads in all for M6.
- The second worker starts a job only while the one-minute load average is at most 48, and it stops itself and the
  job it started — by PID, never by name — when the load average goes above 48.
- The start time, end time, exit code and CPU seconds of every process of both workers are logged, and the figures
  go into entry 2.
- No queue entry is run twice: the first worker's queue is cut at the byte it had consumed, as the engineer
  reports, and the second worker's queue holds only the G1 and G2 seeds 1-2 and the no-grab forms.
- The second worker runs released work only. It runs no D1 grid fit, no selection on D1 and no stack: entry-1
  rulings §3 step 4 is unchanged — "The first D1 grid fit, any selection, gate G-D1a and any stack wait for that
  confirmation."
- No recipe, seed, constant, bound, iteration count or thread count changes because of this ruling, and no pinned
  file is touched.

Reason for authorising it rather than refusing it: the machine has real headroom (load 29.2 of 48 cores, the
physics arm using about 1.4 of them), 7 of the 10 GPUs carry another user's work so the queue cannot be moved to a
GPU, and the alternative is about seven hours of idle waiting for work that is already released. The purpose of
the stamped sentence is to keep a shared machine usable, not to cap throughput, and the load guard serves that
purpose directly. Nothing about the scientific content of any arm changes.

### C-3 — everything else is unchanged

- GPU rules stand in full: the lock helper for every GPU job, at most 2 h per GPU process, never a GPU that
  carries another user's process. At the time of writing another user holds GPUs 2, 3, 4, 6, 7, 8 and 9, so only
  0, 1 and 5 may be taken, and only after `nvidia-smi` is checked immediately before the launch.
- The 60 GB free-disk floor stands; `df -h /` before every build or download.
- The release gate is untouched. No D1 grid fit, no selection on it, no gate G-D1a and no stack before the lead's
  written confirmation of the code batch.
- This ruling expires with the architecture-v2 round of M6 and extends to no other workstream. Another workstream
  that wants a second concurrent CPU job asks the lead for its own ruling.

## 4. How this file is read

- C-1 is a reading of stamped text and adds nothing to the record of deviations. C-2 is a deviation from stamped
  text, authorised in advance and in writing, and it is written into entry 2 as a deviation.
- Where this file and a lead message differ, this file wins.
- The id of this file's registry row is recorded in the campaign decisions file, and entry 2 cites this file by
  name and sha256.
