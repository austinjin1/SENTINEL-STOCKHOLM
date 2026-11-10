# EA England continuous water-quality data — allocation note (M6 new-geography block)

**Written 2026-11-09 by M6, before any EA data are downloaded or inspected** (lead approval of the same day).
Source: Environment Agency Hydrology API (`https://environment.data.gov.uk/hydrology/`), water-quality sonde
measures (temperature, dissolved oxygen; also pH, conductivity, turbidity, ammonium, chlorophyll), 15-min, 2012 →
present, **Open Government Licence v3.0** ("Contains Environment Agency information © Environment Agency and/or
database right"). Shared processed dataset: `processed/ea_wq_sonde` (manifest, datasheet, licence, content pins),
built by M6; other workstreams (e.g. M1 Sonde) may read it under the rules below.

## Reserved for M6's held-out test block (no other workstream may train, tune or select on these)
1. **Time reservation (effective now, all sites):** every EA sonde value dated **on or after 2023-10-01**
   (hydrological years 2024–2026, i.e. Oct 2023 onward) at **every** EA site, all variables. These are the candidate
   unseen block; they are not used by M6 either until its EA protocol and split are frozen.
2. **Site reservation (effective when the M6 EA split is frozen):** **all years** at the sites of the M6-EA
   **test-role basin groups**. Groups = locally delineated catchments merged along flow connections (HydroBASINS /
   CAMELS-GB / EA catchment boundaries); roles are drawn by a seeded hash over group IDs, written to
   `docs/preregistration/hydromet_ea_split.json` with its sha256 before any fit. The list of reserved site IDs is
   published in that file; until then rule 1 alone applies.
3. **Everything else** (pre-2023-10-01 data at non-test-role sites) may be used for training/tuning by any
   workstream, including M6's own EA train/val blocks (zero-shot evaluation needs no EA training; the fine-tuned mode
   trains only on that cell, with the comparators fine-tuned identically).

## Notes
- The EA sonde network skews toward impacted waters (deployments near sewage works / discharges, pollution
  campaigns). M6 will disclose this and, if derivable a priori from public data (e.g. distance to permitted
  discharge points / EA "consented discharges" register), pre-declare a stratum "near discharge vs not".
- Daily statistics (mean, min, max; 7-DADMax) are derived from approved/qualified 15-min values only; flags kept.
- Any change to this allocation is dated here and announced to the lead before it takes effect.

## Proposal — M1 Sonde / EW-v3 (2026-11-09, written by M1; PENDING agreement of M6 and the lead; not in effect)
M1 has downloaded and inspected **no** EA data and will not download it separately. It would read only
`processed/ea_wq_sonde` once M6 publishes it, and only within an agreed allocation. **Until M6's EA split is frozen,
M1 does not train on EA data at all** (at most station/measure counts and metadata checks). M1 accepts M6's
reservations (a) and (b) as they stand; nothing below overlaps them.

Purpose: (i) more training data for M1 forecasting / EW event heads (the rule-3 cell only: pre-2023-10-01 data at
non-test-role sites, after M6's split freeze); (ii) an **unseen, M1-owned EA test block** for an EW-v3 / M1 GB
transfer track. Under (a) + (b) no post-2023 EA data are available to M1, so an M1 test block can only be carved out of
**pre-2023-10-01 data at sites outside M6's test-role groups**.

**Proposed M1 block (non-overlapping):** in the same seeded hash over basin-group IDs that writes
`docs/preregistration/hydromet_ea_split.json`, drawn once and before any fit, a further role **M1-test** is assigned
to ≤ 20 % of the groups that are **not** M6 test-role groups. At M1-test groups, **data dated before 2023-10-01** are
reserved for M1's single test look: no workstream (M6 included) trains, tunes or selects on them. Data from 2023-10-01
at those groups remain under M6's reservation (a), and M1 never uses them. Every other group keeps rule 3 unchanged.
The M1-test site list and sha256 are published in that split file. If M6 prefers not to add a role to its draw, the
alternative is that M1 draws its own M1-test groups **only from M6's published non-test-role groups**, after M6's
freeze, with a seeded hash, before any fit, written to `docs/preregistration/sonde_ea_split.json`.

M1 rules: power calculations use only metadata and rule-3 data from non-M1-test groups. No M1 code reads M1-test
values before an M1 freeze for that track. The track gets its own protocol section, split pin, configuration cap and
hardened single-look guard. M1 never reports on M6's block. Comparator parity: every M1 baseline gets the same EA
training data.

### Decision (lead, 2026-11-09) — M1 allocation APPROVED (recorded by M1)
- **Approved variant.** M1 draws its **M1-test** groups **only from M6's published non-test-role basin groups,
  after M6's EA split is frozen.** This is the second variant above; there is no change to M6's draw.
- **Share.** At most **20 %** of the non-M6-test groups, drawn by a seeded hash over group IDs. The seed and the hash
  rule are fixed in M1's EA protocol section before the draw.
- **Data.** At M1-test groups M1 uses **pre-2023-10-01 data only**. Data from 2023-10-01 stay under M6's reservation
  (a) and are never used by M1.
- **Record.** The list goes to `docs/preregistration/sonde_ea_split.json` with its sha256, before any M1 fit on EA
  data. No workstream trains, tunes or selects on M1-test groups.
- **Until M6's split is frozen,** M1 does nothing with EA data except metadata/count checks.
