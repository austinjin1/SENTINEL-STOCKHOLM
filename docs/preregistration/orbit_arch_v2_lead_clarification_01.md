# M2 SENTINEL-Orbit — architecture round v2: lead clarification 01 (three questions of M2, 2026-11-10)

**Written by:** the lead (AI-assisted), 2026-11-10. The binding time is the timestamp of the registry row that
records this file's sha256 (workstream `M2-orbit`, experiment `orbit_arch_v2/lead_clarification`).

**Status.** This file answers three questions M2 put to the lead in its step-5 report (20:02 EST 2026-11-10),
before any dev_v5m artefact, any value-reading build and any validation number of a new arm or comparator existed
(M2's statement; the lead found no `sentinel_runs/orbit/v2` folder at the time of writing). It changes no gate,
threshold, hypothesis, comparator list or family of the stamped amendment
(`docs/preregistration/orbit_arch_v2_amendment.md`, sha256
`5001373da42652926c6ea4a84028ac46628aa0e6d8b6e99380244417aeb8df8a`) or of the stamped lead rulings
(`docs/preregistration/orbit_arch_v2_lead_rulings.md`, sha256
`0a554541e465cc5dd5e778049a8a2d74d7bff1fa8bc14d0e90d88f6457b81761`). Where a stamped text can be read in more
than one way, this file takes the strictest reading. It is not a `prereg_snapshot` stamp and releases nothing.

## K-1. Small-lake family: the count used for the status-setting MDE and for Holm

The stamped texts give three pieces: amendment §1.7 ("4 targets × the 3 H1 comparators, Holm within the family,
per arm"), ruling R-10 (the MDN_B + GBM average is a fourth claim comparator in the small-lake family) and ruling
R-15 ("Holm runs over both arms' hypotheses together and a label for a target needs every comparator hypothesis
of that target rejected"). Read together they give 4 targets × 4 claim comparators × 2 arms = **32** elements.
The other readings M2 named (12, 24) are smaller.

- The status-setting MDE80 of the small-lake stratum (R-16, `scripts/train/orbitv2_power.py`) is computed at the
  smallest Holm α of a family of **32**: α = 0.05 / 32. No reading of the stamped texts gives a larger count, so
  this is the largest MDE any reading gives, and the stratum can become confirmatory under no reading under which
  it would be descriptive here.
- If the stratum is confirmatory, Holm at the look runs over the same 32 elements (target × claim comparator ×
  arm), step-down with monotone adjusted p-values (R-17a); a label for a target and arm needs all four of its
  elements rejected, every point difference in the arm's favour, and the per-seed-unit rule (R-15, R-17).
- The divisors 6 (H2-S) and 3 (Track N) of R-16 are untouched.
- `orbitv2_power.py` is registered with its sha256 before the first validation number of any new arm (R-16). M2
  reports the sha256 it registers; the value at the time of this file was
  `abb8a2b12b81e8808d58e1d372b4e6ca8a859d437e5e1d8d6f600f6473005949`.

## K-2. Comparator fit scheme before the event of R-19

C-10 orders the comparator families first; R-19 fixes the arms' fit scheme (`base` or `recent`) only when the
regenerated v1 freeze candidate has passed its review. The comparators must be fitted on the arms' fit cell.

- The comparator script takes the fit scheme as an argument and writes it into every comparator run record and
  registry row. The value passed is the scheme of the latest registered v1 selection at the time the first
  comparator job starts, named with that selection's registry row id.
- The finalize step refuses to seal, select or report any comparator family whose fit scheme differs from the
  arms' value fixed under R-19. If the values differ, **every** comparator family is refitted in full under the
  arms' value before the first validation number of any arm (R-14a still holds). The rows of the first fit are
  kept, marked superseded and reported; no member choice, hyperparameter, early-stopping round or stacking
  weight of the first fit is carried into the second.
- The v1 selection under protocol §6, §6a and §11.6 is made by the protocol's rule on dev_v5. Its record cites no
  dev_v5m comparator number, and no dev_v5m comparator number is a reason for anything in the v1 candidate.

## K-3. Comparator compute

- The comparator families run as ruled (R-10 to R-14b); the member count is not reduced to save time, and no
  member is dropped because it is slow.
- Each job is resumable per member, uses at most 6 threads, and runs under LEAD rule 5 and R-25 (disk floor
  60 GB free on `/`). At most two such CPU jobs of M2 run at once while the machine's load is above its core
  count. A member interrupted by a time limit resumes with the same seed and configuration; a resumed member's
  record names both parts.
- CPU-hours are logged per job in the registry. There is no CPU-hour ceiling on comparators: a ceiling could
  only remove a reference.
