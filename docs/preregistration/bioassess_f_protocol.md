# M5-F — SENTINEL-Biota France block (pre-registration)

**Status:** DRAFT **v1 (2026-11-10T19:26Z)**, written by the M5 engineer (AI-assisted) for the lead's re-read and
stamp. **Not yet stamped.** v1 revises draft v0 (sha256 `b920816c205805176f931a17b48114f0bb8d4891364283079c430c1c8704e821`)
in place, after the independent pre-stamp review (`docs/preregistration/reviews/bioassess_f_prestamp_review_20261110.md`)
and the lead's rulings on it and on the engineer's implementation choices (a)–(o). §15 maps every review item and
every ruling to the section that changed, and §15.3 lists what is not done, with the reason.

**Stamp procedure.** The lead re-reads this file, records its sha256, writes the rulings in a separate file
(`docs/preregistration/bioassess_f_lead_rulings.md`), sets both files to mode 0444 and logs a registry row
(workstream `bioassess`, experiment `bioassess_f/prereg_snapshot`). The git commit is deferred; the commit and the
public push must exist before any freeze. After the stamp this file is not edited: a later change is a dated entry
in a new stamped file, made before the data it affects are seen.

**Names.** The display name in prose is SENTINEL-Biota (formerly SENTINEL-BioAssess). Code, registry keys and file
names keep `bioassess`. Arms are named second, for example "SENTINEL-Biota (arm BioSet-MA)".

**State when this was written.**
- No French biology or physico-chemistry observation file is on disk.
- No French label value, taxon list or station table has been read by anyone in this workstream.
- Documentation only: 60 reference files (SANDRE reference records, Hub'eau data dictionaries, Naïades
  documentation pages, the official status-assessment guide) were fetched on 2026-11-10 under the lead's ruling on
  review question 6. Each is logged with URL, time, bytes and sha256 in
  `docs/preregistration/bioassess_f_doc_fetch_log.md`. No endpoint that returns observation rows was called and no
  station-level query was made.
- BioSet-MA and the TabPFN hybrid have only been run in code and shape tests (synthetic data and allowed UK pool
  rows). No validation number exists for either.
- No French row count is used in this file. The counts in the architecture plan (API-page metadata) are
  UNVERIFIED; real counts come at gates G2–G4.
- The code is not yet aligned with every rule of this file (list in §15.3). It will be before any validation
  number is produced.

**Governing documents.** `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md` (rules 1–6 and the M5
section), `docs/plans/architecture_research/M5_architecture_plan.md`, `docs/plans/architecture_research/SYNTHESIS.md`
(§3.3 France row, C4, X5), `docs/plans/sota_campaign.md` (rules 1–8; M5-F reservation of 2026-11-10),
RESEARCH_PLAN §14.2, §15.1 and §15.5. Where the plan and the decisions differ, the decisions govern. Nothing here
lowers a bar.

**Relation to M5 and M5-X.** M5-F never replaces the frozen M5 result or the M5-X result. No M5-X file is
edited: M5-F code lives in new modules, so the M5-X scoring-code hash is unaffected. Every M5-F claim names its
track ("M5-F, F-A", ...). M5-X adds no configuration because of this protocol.

**Before the stamp** (decision rule 2): code, unit tests on synthetic data and on allowed UK pool rows (shape and
code tests only), the documentation fetches listed above, and this draft. **No French download and no validation
number on a new arm before the stamp.**

---

## 1. Why this block

M5 and M5-X showed that on identical community inputs every learner sits near the same information ceiling
(plan §1). The M5-X candidate is expected to be SOTA on 2 of 12 targets. The user directive of 2026-11-10 asks
for a harder task with more data and more predictive targets. France is the answer chosen by the lead:

- an **unseen country and network** (SANDRE taxonomy, I2M2-era field protocol), never touched by any workstream;
- the largest open EU archive pairing river biology with chemistry (Naïades / Hub'eau);
- a **second assemblage** (benthic diatoms) and **time-integrated chemistry**, which are new information rather
  than new capacity.

## 2. Data, licence, caps, lock-on-download

### 2.1 Sources and download route
| Source | Content | Licence (recorded per file) |
|---|---|---|
| Hub'eau API "Hydrobiologie" v1 (`/v1/hydrobio/stations_hydrobio`, `/indices`, `/taxons`), which serves Naïades | Stations, taxon lists and biological index results of macroinvertebrate (support 13) and benthic-diatom (support 10) operations. Fish and macrophytes are not requested | Licence Ouverte Etalab. Hub'eau general conditions: reuse "régie par la licence ouverte Etalab", free reuse with citation of the author. Naïades footer: "tous les contenus de ce site sont sous licence etalab-2.0". The licence string and attribution text are copied into the manifest for **each downloaded file**, with URL, retrieval time (UTC), bytes and sha256 |
| Hub'eau API "Qualité physico-chimique des cours d'eau" v2 (`/v2/qualite_rivieres/analyse_pc`; `station_pc` once) | Physico-chemical analyses at the biology stations for the parameters of §6.3; pesticides and metals for F-TOX only if the cap allows (below) | as above |
| UK EA ecology (`processed/uk_ea_ecology`, on disk) | Invertebrates, diatoms, matched chemistry (`pm30d`, `prior365d`), `wq_observations` | OGL v3.0 (existing manifest) |
| NRSA (`processed/nars_streams`, on disk) | US stream invertebrates and chemistry | existing manifest |

Overseas departments are excluded a priori (different biogeography; the national indices do not apply). Only
running waters ("cours d'eau") are used.

**Route (fixed now).** The route is a filtered pull through the Hub'eau API, not the national bulk export. The
size of the national export is unverified (the Naïades access pages are script-driven and show no file sizes) and
is probably far above the cap. The Hub'eau conditions ask users to filter requests to the subset they need.

1. Requests use the `.csv` endpoints with a `fields` whitelist where the endpoint honours it. Each response is
   written gzip-compressed, directly into the locked store (§2.3 item 1).
2. **Biology (step B):** `stations_hydrobio` by department; `indices` and `taxons` by department (the 96
   metropolitan departments of §4) × support code (13, 10) × calendar year, for sampling dates from 2007-01-01 to
   the day of download.
3. **Chemistry (step C), only after the lock list (gate G2):** `analyse_pc` by station batch (at most 50 station
   codes; the stations with at least one eligible invertebrate operation, §3) × the parameter codes of §6.3 ×
   calendar year, for sampling dates from 2006-01-01 to the day of download. Requests for dates ≤ 2021-12-31 and
   ≥ 2022-01-01 are separate requests (filters `date_debut_prelevement`, `date_fin_prelevement`).
4. The API serves at most 20,000 records per query (page × size; documented limit for both APIs). Before each
   block a count request (`size=1`) is sent. A block above 20,000 records is split by month, then by smaller
   station batches. Count responses land in the locked store like any other response.
5. **Never requested:** `condition_environnementale_pc`, `operation_pc`, any status-class product, and any
   parameter outside §6.3 and §6.5. `station_pc` is requested once, for station altitude and for the fallback
   pairing of §3; its `durete` field (mean water hardness, a chemistry-derived station attribute) is never
   extracted.
6. **Size check before each step.** Projected bytes = (sum of the block counts) × (compressed bytes per record
   measured on the first completed block of that endpoint). A step starts only if 1.5 × the projection fits in
   what remains of the cap. Otherwise work stops and is reported. Nothing is reduced silently.
7. F-TOX pesticide and metal analyses are pulled last, and only if at least 1 GB of the cap remains after steps B
   and C. Otherwise F-TOX is "not done, with reason".
8. One request at a time, at least 0.5 s between requests, a descriptive user agent. Every request URL, UTC time,
   HTTP status, byte count and sha256 goes to the manifest.

**Codes before the first chemistry request.** The parameter, fraction, unit, support, remark, qualification,
in-situ and status codes used here were checked against SANDRE reference records fetched on 2026-11-10 (§6.3;
fetch log). Before the first chemistry request they are written, with the source URL and sha256 of each record, to
`docs/preregistration/bioassess_f_codes.json`; its sha256 is reported to the lead and logged in the registry.

### 2.2 Caps and disk
- New data for M5-F: **≤ 5 GB** in total, measured with `du` over raw, processed and temporary files. Transient
  archives and count responses count toward the cap while they exist; the peak is logged.
- `df -h /home/bcheng` is run and logged before every download or build step. Work stops and is reported if free
  space would fall below **60 GB**.
- Intermediate caches are deleted when no longer needed.
- Compute rules are in gate G10 (§11).

### 2.3 Lock-on-download (the reserved unseen block)
The lead reserved the block on 2026-11-10 (`sota_campaign.md`): French biology operations dated **2023–2025**
and their paired physico-chemistry. The lock is implemented as follows, in this order.

1. **Everything lands locked first; extraction is by whitelist and date.** Every raw French API response or
   archive (biology, chemistry, station tables, count responses) is written first to `raw/naiades/locked/`. The
   fetcher writes French data nowhere else. One audited extractor copies out of the locked store only:
   - *biology input columns* (whitelist below): rows dated ≤ 2021-12-31 go to the unlocked biology table; rows
     dated ≥ 2022-01-01 go to the "locked-operation inputs" table, whose uses are limited to item 6;
   - *chemistry columns* (whitelist below) of rows dated ≤ 2021-12-31, to the unlocked chemistry table. Chemistry
     rows dated ≥ 2022-01-01 never leave the locked store.

   **Routing field:** `date_prelevement` in both APIs (hydrobiology: date of the biological sampling;
   physico-chemistry: start date of the water sampling, format AAAA-MM-JJ). The extractor reads it only as a
   date. A row whose `date_prelevement` is missing, cannot be parsed, is before 1990-01-01 or is after the
   download date stays in the locked store and is never used; such rows are counted.

   **Biology whitelist:** `code_station_hydrobio`, `date_prelevement`, `code_support`,
   `code_operation_prelevement`, `code_prelevement`, `code_banque_reference`, `code_appel_taxon`,
   `libelle_appel_taxon`, `codes_taxons_parents`, `libelles_taxons_parents`, `code_type_resultat`,
   `resultat_taxon`, `code_qualification`, `code_methode`, `libelle_methode`, `code_lot`,
   `codes_indices_operation`, `code_indice`, `libelle_indice`, `resultat_indice`, `unite_indice`, `longitude`,
   `latitude`, `code_departement`, `code_commune`, `code_cours_eau`, `code_masse_eau`. From `stations_hydrobio`
   records only the station columns of this list are extracted (`code_station_hydrobio`, `longitude`, `latitude`,
   `code_departement`, `code_commune`, `code_cours_eau`, `code_masse_eau`).

   **Station whitelist** (`station_pc`): `code_station`, `longitude`, `latitude`, `code_departement`,
   `code_commune`, `code_cours_eau`, `code_masse_deau`, `altitude_point_caracteristique`.

   **Chemistry whitelist** (`analyse_pc`): `code_station`, `date_prelevement`, `heure_prelevement`,
   `code_parametre`, `resultat`, `code_unite`, `code_fraction`, `code_support`, `code_remarque`, `code_insitu`,
   `limite_detection`, `limite_quantification`, `limite_saturation`, `code_statut`, `code_qualification`,
   `code_operation`, `code_prelevement`, `code_analyse`, `code_banque_reference`.

   **Never extracted:** every other column. In particular `durete`, any chemistry-derived status class or station
   attribute, free-text comment fields, and the water depth, water width and prospected length recorded with a
   biological operation (they do not exist on UK rows).
2. **Lock list by key, before any chemistry request.** From the two biology tables only, write
   `docs/preregistration/bioassess_f_lock.json`:
   - one key per biology operation dated ≥ 2022-01-01: `FR:<station code>:<YYYY-MM-DD>:<support code>` (plus
     `code_operation_prelevement` where present);
   - role `test` for 2023-01-01..2025-12-31, `buffer` for 2022, `future` for anything later;
   - counts per role, year and support, and the sha256 of the sorted key list.
   The file is made read-only and its sha256 is reported to the lead and logged to the registry. **No French
   physico-chemistry request is sent before this file exists.**
3. **Chemistry is split by date at download and again at extraction.** Requests for ≤ 2021-12-31 and
   ≥ 2022-01-01 are separate (§2.1). Both land in the locked store, and the extractor of item 1 routes every row
   by `date_prelevement` alone, without parsing, logging or summarising result values.
   The boundary is 2022-01-01 rather than 2023-01-01 because the prior-365-day window of a January 2023 operation
   reaches back to January 2022.
4. **Locked values are scorer-only.** Before the single look, exactly one audited function reads chemistry in the
   locked store: `ftrack.locked_availability()`, which returns per (operation key, parameter, window) the **number
   of analyses** and the number of distinct sampling dates, and nothing else (decision rule 2: counts only). Label
   values of locked rows are read only by the one-time scorer under an open single-look claim (§12). Every other
   loader refuses them, and a static test asserts that no development module opens the locked directory.
5. **Unlocked labels never reach across the boundary.** Windows of training and development operations are
   truncated at 2021-12-31, so a December 2021 operation's ±30-day window uses no January 2022 analysis.
6. **Inputs of locked operations: closed list of permitted uses.** The inputs of operations dated ≥ 2022-01-01
   (taxon lists, index results, station metadata, context columns) may be used only as follows. Every use is
   label-free and is the same for candidates and comparators.
   - (a) building the lock list of item 2;
   - (b) prediction at scoring time (the inputs of the rows being predicted, including their attached diatom
     operation);
   - (c) region-context inputs at prediction time, for configurations #3, #7 and #8 and for the comparator
     controls that have a region-context variant (§8 item 9), on the rows defined by the mirror rule of §9.1;
   - (d) the prediction-time k-means grouping of the hybrid's query rows (§9.2), also used by its two controls.

   **Nothing else.** No fit, pretraining, vocabulary, scaler, feature selection, taxon list or PCA basis uses
   these inputs. The PCA-20 basis of the region descriptors is fitted on training pseudo-regions only (§8 item 9).
   Every label for a model that uses (c) or (d) says "unlabelled test-region inputs used at prediction time".
7. **What the guard is.** Directory mode 0700 and file mode 0400 mark the locked store; they are not access
   control, because every agent runs as the same OS user. The protection is procedural plus code: this rule, the
   single extractor and the single counting reader, `ftrack` refusing locked paths and lock keys in every fit, the
   static test of item 4, and the content-hash pins checked at the look (§12).
8. **Any breach** (a locked value read outside the scorer, or a locked-operation input used outside item 6) is an
   incident: work stops, the lead is told, and the affected rows are flagged in the results.

## 3. Rows, pairing, pools

**Row unit.** One macroinvertebrate operation (station × date). Abundances are summed over the operation's
elementary samples or phases, and over several operations at the same station and date. Diatom operations enter
as a second modality (§9) and as extra training rows; they are not scored units of their own.

**Row eligibility (fixed now).**
- Support code 13 (macroinvertebrates) for row units; support code 10 (benthic diatoms) for the second modality.
- Taxon records with `code_type_resultat` = 1 (count). Other result types are not used.
- Taxon records with `code_qualification` = 2 ("Incorrecte") are dropped, for every model.
- Operation date from **2007-01-01**. The date is fixed a priori, not from counts (reason: the first year of the
  Water Framework Directive surveillance network, by the engineer's recollection, not checked). Earlier operations
  are not requested.
- Field protocol: every standard macroinvertebrate stream protocol of that period is eligible (the IBGN-era
  protocol; the I2M2-era NF T90-333 method named in the SANDRE record of index 5910; the large-river protocol).
  The method code is an input token (the M5-X `analysis_method` rule). The list of eligible `code_methode` values
  is fixed at G4 from the SANDRE method reference and from the number of operations per code (counts only).
  Sensitivity, no refit: test rows for which an I2M2 result (index 7613) is delivered.
- At least 3 taxa after harmonisation.
- A metropolitan department (§4) and usable station coordinates.

**Biology–chemistry pairing.** Same SANDRE station code. The fallback is the nearest chemistry station on the same
watercourse code within 500 m. The choice is made once, from counts only, before the split freeze, and recorded:
the same-station pairing is kept if at least 4 of the six listed F-A primary targets of §6.1 (before any
replacement) have ≥ 650 test location clusters with it; otherwise the fallback is added for all rows, and gates G3
and the replacement rule are evaluated with it.

**Diatom attachment.** The diatom operation nearest in time at the same station within ±90 days is attached to
an invertebrate operation, with the offset in days as a modality feature. Ties: the nearer date, then the earlier
date. **No diatom operation dated after the end date of the row's pool is attached**: test-design training
2021-12-31; development training 2017-12-31; development pseudo-test 2021-12-31; UK allowed rows 2020-12-01; UK
development training 2015-12-31. So an unlocked operation attaches only unlocked diatom operations and never
carries a lock key. Test rows attach diatom operations of any locked role (use (b) of §2.3 item 6). The same rule
is used on UK rows. Sensitivity, no refit: F-MA results on test rows with |offset| ≤ 30 days.

**Location clusters.** The M5 1-km rule (`data.location_clusters`) on station coordinates. Every bootstrap
resamples clusters.

**Taxonomy.** SANDRE taxon names are matched to the existing `taxonomy_harmonised` table, with lineage from the
same backbone:
- (a) by the exact name (`libelle_appel_taxon`) after the existing name normalisation;
- (b) failing that, by the lowest-rank name in `libelles_taxons_parents` that matches; the count is carried to
  that parent;
- (c) failing that, the record is dropped, for every model alike.

The matched share of records and of individuals is reported per source and fold. If less than 90 % of the
individuals of the unlocked pool match (counts only), the lead is told before G4. Vocabularies, `min_name_count`
filters, feature selection and scalers are fitted per fold on that fold's training rows only (M5 rule W1).

**Pools.**

| Pool | Rows | Used by |
|---|---|---|
| France test | Operations with role `test` in the lock file, in fold region *g* | Scorer only |
| France training (test design) | Operations dated 2007-01-01..2021-12-31, outside region *g*, with no station within 1 km of a region-*g* test station | F-A, F-T |
| France development pseudo-test | Operations dated 2019-01-01..2021-12-31 in region *g* | Development scoring for all tracks |
| France training (development design) | Operations dated 2007-01-01..2017-12-31 outside region *g*, 1-km rule; 2018 is the development buffer year | F-A, F-T development |
| UK allowed rows | See below | All tracks |
| NRSA allowed rows | The NRSA rows of the M5-X training pools (item-7 exclusions through `xtrack.frozen_test_keys()`) | F-Z comparators only (§8) |

**UK allowed rows (decision of 2026-11-10).**
- Invertebrate samples: `uk.uk_site_roles` role `pool_fit` or `pool_val`, sample date **≤ 2020-12-01**, and not a
  key in `xtrack.frozen_test_keys()`.
- Diatom samples: light-microscopy analyses only (`analysis_type` = `LABP`); sample date **≤ 2020-12-01**; the
  site is not a `reserved_test` site; no `reserved_test` invertebrate site lies within 1 km; its paired
  `wq_point` is not paired to any `reserved_test` site; and its 50-km block (the M5 reserved-block grid) holds no
  `reserved_test` site. Diatom samples with labels and no invertebrate sample are "diatom-only" training rows.
- The sample-date cut is 2020-12-01, not 2020-12-31, so that no ±30-day window of an allowed row reaches 2021
  (lead ruling (b)). Labels are the on-disk `pm30d` and `prior365d` summaries, used as built. Any statistic
  computed anew for M5-F uses only `wq_observations` dated ≤ 2020-12-31.
- Counts (sample and site metadata only, 2026-11-10; fixed before any fit): 199,234 allowed invertebrate samples
  and 14,030 allowed diatom samples, at 5,395 eligible diatom sites; 7,038 `reserved_test` sites in 16 reserved
  50-km blocks are excluded. Moving the cut from 2020-12-31 to 2020-12-01 drops 0 allowed invertebrate samples and
  0 allowed diatom samples: the latest diatom sample before 2021 is dated 2020-11-30, and the one invertebrate
  sample dated 2020-12-02..2020-12-31 is not an allowed row.
- **Never used, for any purpose:** UK 2021–26 `pool_fit` rows, any `reserved_test` row, NRSA 2018-19 and
  2023-24, NLA 2022, and every other workstream's frozen test rows, including new targets on those rows.

**Fit-id guard.** Every M5-F fit calls `ftrack.assert_fit_ids` on its fit and validation ids. It refuses lock-file
keys of any role, development pseudo-test rows of the fold, disallowed UK rows and frozen test keys. Every
encoder or checkpoint load goes through a guarded loader that checks a training-id sidecar against the fold's
training pool. No M5 or M5-X checkpoint is reused.

## 4. Folds (fixed now)

Leave-region-out × time-out over **8 fixed groups of metropolitan departments**. A station's group is given by
its department code (from the station record; else the first two characters of its commune INSEE code; else the
department of the nearest station with a known code). No label and no French file was used to draw the groups.
They are administrative groups of whole departments drawn by hand to approximate the large river basins. They are
**not** the water-agency districts: G1 joins two agencies, and G3 and G5 cross agency boundaries.

| Fold | Name | Departments |
|---|---|---|
| G1 | Nord and Nord-Est (Artois-Picardie, Rhin-Meuse) | 02, 08, 54, 55, 57, 59, 62, 67, 68, 80, 88 |
| G2 | Seine amont and Île-de-France | 10, 51, 52, 60, 75, 77, 78, 89, 91, 92, 93, 94, 95 |
| G3 | Normandie and Bretagne | 14, 22, 27, 29, 35, 50, 56, 61, 76 |
| G4 | Loire moyenne et aval, Centre-Ouest | 18, 28, 36, 37, 41, 44, 45, 49, 53, 58, 72, 79, 85, 86 |
| G5 | Massif central | 03, 15, 19, 23, 42, 43, 48, 63, 87 |
| G6 | Adour-Garonne | 09, 12, 16, 17, 24, 31, 32, 33, 40, 46, 47, 64, 65, 81, 82 |
| G7 | Rhône, Saône, Alpes du Nord, Jura | 01, 21, 25, 38, 39, 69, 70, 71, 73, 74, 90 |
| G8 | Méditerranée and Corse | 04, 05, 06, 07, 11, 13, 26, 2A, 2B, 30, 34, 66, 83, 84 |

- **Known weakness (disclosed).** The groups split river systems: by the reviewer's reading the Seine (G2/G3),
  the Loire (G5/G4), the Rhône (G7/G8) and the Dordogne–Lot (G5/G6); this was not checked against a network file.
  RESEARCH_PLAN §14.2 asks for flow-connected stations to stay together, so three reports are added (all without
  a refit):
  - a **nearest-training-station distance report**: for every test station the distance in km to the nearest
    training station of its fold, saved with the split file;
  - a **≥ 50 km sensitivity stratum**: all confirmatory comparisons repeated on test stations whose nearest
    training station is ≥ 50 km away;
  - a **same-watercourse stratum**: comparisons repeated without the test stations whose watercourse code
    (`code_cours_eau`) also has a training station.
- **HER-1 sensitivity.** If a level-1 hydro-ecoregion code per station can be obtained from an open layer under a
  recorded licence before G4, the per-region table and the region-unit CI are repeated with HER-1 units (no
  refit). Otherwise it is "not done, with reason". The department groups stay primary.
- **Merge rule (counts only; full order fixed now).** A unit is "small" if it has fewer than 100 test location
  clusters carrying at least one listed F-A primary target. While a small unit exists: take the smallest (ties:
  the lower group number) and merge it with its partner at the lowest level not yet merged.
  - Level 1: (G1, G2), (G3, G4), (G5, G6), (G7, G8).
  - Level 2: (G1+G2, G3+G4), (G5+G6, G7+G8).
  - Level 3: all groups.

  Merging stops when every unit has ≥ 100 such clusters. Merges are fixed at the split freeze and apply to every
  track. **If fewer than 6 region units remain**, the region-unit condition of §10 is "not evaluable" and row 1
  of the decision table cannot be met by any track.
- **Region** for `r2_anomaly`, for the region-unit CI and for region context is the fold unit after merges.
- The frozen split table `docs/preregistration/bioassess_f_split.json` (keys, roles, folds, clusters, target
  availability masks, pairing choice, merges, distance report, sha256) is written after the lock list and before
  any fit on a French row. It needs the lead's stamp (G4).

## 5. Tracks and their a-priori practical reasons

Written on 2026-11-10, before any French row exists on disk and before any result on any M5-F arm.

| Track | Status | Practical reason (a priori) | Test rows | Training |
|---|---|---|---|---|
| **F-A** new national network, leave-region-out × time-out | Confirmatory | Agencies want chemistry where only biology is sampled. A different taxonomic system and field protocol test whether the method carries to a network it was not built on, in a basin it has not seen, in years it has not seen | France test rows, ±30-day targets | France training (test design) + UK allowed rows |
| **F-Z** zero-shot new country | Confirmatory | Many countries run biomonitoring with sparse chemistry. This is the deployment case with no local labels at all | The same rows and targets as F-A | No French label in any fit, early stopping, scaler, calibration or candidate selection (§8–§9) |
| **F-T** time-integrated statistics | Confirmatory | Benthic fauna integrate months of exposure, and status classification uses annual statistics. On allowed UK rows the ±30-day median correlates only 0.58–0.76 with the prior-365-day mean for DO, BOD, NH4 and pH (plan §1.2) | France test rows with ≥ 6 analyses of the target's parameter in the prior 365 days (§6.3 rule 10) | As F-A |
| F-E extremes and exceedances | Secondary | Exceedances drive management, and the transfer-function literature warns that ML fails when extrapolating (Sun, Holden & Birks 2024) | F-A test rows beyond the P90 of the fold's F-A training-pool labels (French training rows plus UK allowed rows), per target; P10 for the low side of pH and for DO saturation. Event set of §6.4 | As F-A |
| F-MA multi-assemblage stratum | Secondary | Agencies co-sample invertebrates and diatoms; diatoms are the classical nutrient and pH indicators | F-A test rows with an attached diatom operation | As F-A |
| F-TOX toxic pressure | Exploratory | Pesticide monitoring is sparse and costly; SPEAR is the published invertebrate indicator | F-A test rows with a pesticide panel in the prior 365 days | As F-A |

**One look, one family.** F-A, F-Z and F-T are scored in a single look under one frozen split and form one Holm
family (§10). F-E and F-MA are secondary: unadjusted and labelled as such. F-TOX is exploratory: descriptive
only, never in a family, never a SOTA label. Every track is reported, wins and losses.

**F-Z label wording.** Every F-Z label carries: "zero French training rows; candidate selected without French
labels; comparators selected on French development labels; unlabelled French inputs used at prediction time".

## 6. Targets

### 6.1 Primary targets (F-A and F-Z; ±30-day median, window [d−30, d+30])
pH; log10 conductivity at 25 °C; log10 chloride; log10 nitrate-N; log10 ammonium-N; log10 orthophosphate-P.
These match the M5-X X-U targets so the UK rows carry the same labels.

### 6.2 Primary targets (F-T; window [d−365, d])
Arithmetic mean over the window, then the transform: pH; log10 orthophosphate-P; log10 ammonium-N;
log10 nitrate-N; log10 BOD5; DO saturation (%). A label exists only with **≥ 6 analyses of that parameter** in
the window, counted as observations (§6.3 rule 10).
Secondary F-T statistics (not in the family): the prior-365-day means of conductivity and chloride. The
within-window P90 and P10 statistics of draft v0 (P90 of ammonium-N and BOD5, P10 of DO saturation) are deferred:
they are built only if a stamped deviation adds them before any number on them exists (lead ruling (n)).

### 6.3 Label construction (fixed now)

**Verified codes** (SANDRE reference records fetched 2026-11-10, all with status "Validé"; URL and sha256 of each
record in the fetch log).

| Label | SANDRE parameter | Fractions kept | Units kept → label unit |
|---|---|---|---|
| pH | 1302 Potentiel en Hydrogène (pH) | 23 | 264 (unité pH) |
| Conductivity at 25 °C | 1303 Conductivité à 25°C | 23 | 147 (µS/cm) |
| Chloride | 1337 Chlorures | 3, 23 | 164 (mg(Cl)/L); 162 (mg/L) |
| Nitrate-N | 1340 Nitrates | 3, 23 | 173 (mg(NO3)/L) × 0.2259; 168 (mg(N)/L) × 1 |
| Ammonium-N | 1335 Ammonium | 3, 23 | 169 (mg(NH4)/L) × 0.7765; 168 (mg(N)/L) × 1 |
| Orthophosphate-P | 1433 Orthophosphates (PO4) | 3, 23 | 176 (mg(PO4)/L) × 0.3261; 177 (mg(P)/L) × 1 |
| Total P | 1350 Phosphore total | 23 | 177 (mg(P)/L) |
| Suspended solids | 1305 Matières en suspension | 23 | 162 (mg/L) |
| Dissolved O2 | 1311 Oxygène dissous | 23 | 175 (mg(O2)/L) |
| O2 saturation | 1312 Taux de saturation en oxygène | 23 | 243 (%) |
| BOD5 | 1313 Demande Biochimique en oxygène en 5 jours (D.B.O.5) | 23 | 175 (mg(O2)/L) |
| Nitrite-N | 1339 Nitrites | 3, 23 | 171 (mg(NO2)/L) × 0.3045; 168 (mg(N)/L) × 1 |
| Kjeldahl N | 1319 Azote Kjeldahl | 23 | 168 (mg(N)/L) |
| Water temperature | 1301 Température de l'Eau | 23 | 27 (°C) |

- Fractions: 23 = "Eau brute"; 3 = "Phase aqueuse de l'eau (filtrée, centrifugée...)". Support: 3 = "Eau".
- Biology: supports 13 "Macroinvertébrés aquatiques" and 10 "Diatomées benthiques". Index codes: 7613 I2M2;
  1000 IBGN; 5910 the "équivalent" index (phases A+B of the NF T90-333 method); 6951 large-river index;
  5856 IBD; 1022 IPS.
- Conductivity at 20 °C (1304) is a different parameter. It is never mixed in or converted.
- Molar factors: NO3 → N 0.2259; NH4 → N 0.7765; PO4 → P 0.3261; NO2 → N 0.3045. Floors are those of
  `uk.UK_TARGETS`.

**Rules.** Each rule names its UK counterpart in `sentinel/data/bio_uk_ea.py` / `fdata.py`.

| # | Rule for French analyses | UK counterpart |
|---|---|---|
| 1 | Only analyses of water (`code_support` = 3) at the paired station | Same idea: only routine samples of the river water itself (material whitelist) |
| 2 | Only the parameter × fraction × unit combinations of the table above. Any other fraction or unit code is dropped and counted. No unit is converted by guess | Identical: summaries are per determinand × unit; units are never mixed |
| 3 | Qualification (nomenclature 414): keep 1 "Correcte", 4 "Non qualifié", 0 "Qualification non définissable"; drop 2 "Incorrecte" and 3 "Incertaine". Sensitivity: code 1 only | None (the EA archive has no such code) |
| 4 | Status (nomenclature 446): no filter; the code is stored and its counts reported | None |
| 5 | Remark code (nomenclature 155): 1 → the reported result. 2, 7, 10 → censored below; the analysis enters at the reported result, which by the SANDRE definition carries the detection or quantification limit. 3 → censored above; enters at the reported result (the saturation limit). 0, or an empty result → dropped. Any other code → dropped and counted. The censored fraction is stored | Identical: censored results enter at their reported limit and the censored fraction is stored |
| 6 | In situ versus laboratory (nomenclature 156): no preference. Every kept analysis of the parameter enters, whatever `code_insitu` | Identical: every observation of the determinand enters |
| 7 | Several analyses on one day all enter as separate observations. Before that, exact duplicate records (same station, date, time, parameter, fraction, unit, result, remark code and in-situ code) are reduced to one | Identical: all observations enter; duplicates are removed on the observation identifiers |
| 8 | Non-finite or non-numeric results are dropped. Zero or negative values of a log target go to the floor through the transform | Identical |
| 9 | Plausibility bounds on the window statistic: pH 3–11; water temperature −1–40 °C; DO saturation 0–300 %. Outside → the label is missing. No other bound. The bounds apply identically to every role, model and comparator. The number of labels set to missing is reported per role; for locked rows it is produced inside the look and reported as a count only | Identical (`fdata.PLAUSIBLE`) |
| 10 | Windows: `pm30d` = [d−30, d+30], median, ≥ 1 analysis. `prior365d` = [d−365, d], arithmetic mean, **≥ 6 analyses of that parameter, counted as observations after rule 7, not as distinct dates**. The number of distinct dates is stored. Sensitivity, no refit: F-T on test rows with ≥ 6 distinct dates | Identical (`n_obs` counts observations) |
| 11 | Windows of unlocked rows are truncated at 2021-12-31 (§2.3 item 5) | UK: see §3 |

- The official French status rule replaces a result below the quantification limit by half the limit (guide cited
  in §6.4). It is **not** used here: rule 5 keeps the French labels identical in construction to the UK labels.
- A sensitivity analysis drops rows whose window is more than half censored, as in v0.
- **Frozen at G4, from documentation and counts only.** Before any French chemistry value is read, the number of
  unlocked analyses per parameter × fraction × unit × remark × qualification × in-situ × status code is tabulated
  (code columns only; `resultat` is not read) and added to `bioassess_f_codes.json` together with the eligible
  method codes of §3. If more than 5 % of the unlocked analyses of a primary-target parameter fall outside the
  whitelist of rule 2, the lead is told. A whitelist may be changed only at this point, from documentation and
  counts, by a dated entry in that file. The lead stamps the file at G4.

**Replacement rule (availability masks only).** A primary target must have ≥ 650 test location clusters (§10). If
one fails, it is replaced by the first that passes from: F-A/F-Z — log10 total P, log10 suspended solids,
DO saturation, log10 BOD5; F-T — prior-365-day mean of log10 conductivity, log10 chloride, log10 total P,
log10 suspended solids. F-A and F-Z always share one target list. If fewer than six pass, the family shrinks.
- "Listed primary targets" means the lists of §6.1–§6.2. "Primary targets" without qualifier means the final
  lists after this rule, fixed at G4.
- Every F-A/F-Z replacement target has a UK determinand (total P, suspended solids, DO saturation, BOD5), so F-Z
  can be trained and its candidate selected (§9.5) whatever the replacement.

### 6.4 Auxiliary targets (multi-task supervision) and events
- Auxiliary, both windows where labels exist: total P, suspended solids, DO saturation, DO (mg/L), BOD5,
  nitrite-N, total oxidised N (UK) or Kjeldahl N (France), water temperature, alkalinity (UK), and the F-T
  secondary means.
- **F-E events.** `sentinel/eval/events.py` v1.0 is imported, never copied or edited (sha256 pinned). The French
  events are `EventDef` objects in a new module, `sentinel/models/bioassess/fevents.py`, passed through the v1.0
  API. Thresholds are in label units. Every event is evaluated on the **±30-day median label** of the F-A test
  rows.

| Code | Event (label units) | Origin of the threshold |
|---|---|---|
| `E2a_lab` (v1.0, reused) | chloride > 230 mg/L | M5-X v8 event set |
| `E5b` (v1.0, reused) | total P > 0.1 mg P/L | M5-X v8 event set; kept for comparability with M5-X. It is not the French limit |
| `E6_snap` (v1.0, reused) | pH < 6.5 or > 9.0 | M5-X v8 event set; kept for comparability with M5-X. It is not the French limit |
| `F_NO3` | nitrate-N > 11.295 mg N/L (50 mg NO3/L) | French good/moderate limit |
| `F_PO4` | orthophosphate-P > 0.16305 mg P/L (0.5 mg PO4/L) | French good/moderate limit |
| `F_NH4` | ammonium-N > 0.38825 mg N/L (0.5 mg NH4/L) | French good/moderate limit |
| `F_TP` | total P > 0.2 mg P/L | French good/moderate limit |
| `F_PH` | pH < 6.0 or > 9.0 | French good/moderate limits (pH minimum 6, pH maximum 9) |
| `F_DOSAT` | DO saturation < 70 % | French good/moderate limit |
| `F_BOD` | BOD5 > 6 mg O2/L | French good/moderate limit |

- **Source of the French limits (verified).** "Guide technique relatif à l'évaluation de l'état des eaux de
  surface continentales (cours d'eau, canaux, plans d'eau)", Ministère de la Transition écologique, December
  2023, Annex 6 "État écologique des cours d'eau – Paramètres physico-chimiques généraux", general table, column
  "Bon / Moyen" (https://www.eaufrance.fr/sites/default/files/2023-12/guide_reee_decembre_2023.pdf; sha256 in the
  fetch log). The guide applies the arrêté of 25 January 2010 as amended, last by the arrêté of 9 October 2023.
  It sets no limit for conductivity or chloride.
- **These events are not the regulatory status class.** The guide classifies the P90 (P10 for oxygen and pH
  minimum) of a series of analyses; an M5-F event is an exceedance of the same limit by the ±30-day median.
- An event is evaluated only if it has ≥ 30 positives and ≥ 10 clusters in the development pseudo-test
  (`events.power_flag`); the test-block count is reported from masks at scoring time. Derived probabilities come
  from the predictive distribution and are recalibrated on development rows only. A direct classifier is added
  only for powered events. Comparators: threshold-on-baseline and prevalence/climatology.

### 6.5 F-TOX (exploratory)
Target: log10 of the maximum toxic unit over insecticide analyses in the prior 365 days, and dissolved Cu and Zn
where measured. Toxicity values are **measured ECOTOX values only**, from a source fixed and licence-checked
before the split freeze; never an M4 model prediction (SYNTHESIS C4). Comparator: SPEAR (pesticides), only if
its taxon classification is obtained under a licence recorded before the freeze; otherwise "not compared, with
reason". The licence is a human decision that the lead raises; nothing is fetched before it. Results are
descriptive whatever the row count.

## 7. Metrics and intervals

- **Co-primary, fixed now: pooled R² and within-region `r2_anomaly`.** Reason, a priori: deployment in a new
  network needs absolute levels (pooled R², M5's original endpoint) as well as ranking within a region
  (`r2_anomaly`, M5-X's endpoint). The M5-X diagnostics show that region-level bias is the only large model-side
  error component and that `r2_anomaly` cannot see it. The choice is not revisited after French results.
- Implementation: the M5 functions in `sentinel/models/bioassess/evaluate.py`, unchanged.
- **Secondary:** median per-region R², RMSE, bias, Spearman; negative log-likelihood and 90 % interval coverage
  for models with predictive distributions.
- **CIs:** percentile location-cluster bootstrap, B = 2,000 marginal and B = 10,000 paired. Sensitivity:
  50-km-block and region-unit paired CIs, with per-region sign counts (8 units are coarse).
- **Seeds:** 0, 1 and 2 for every learned model on both sides.
- **Statistic entering the bootstrap.** On each side the predictions of seeds 0, 1 and 2 are averaged per row;
  the metric is computed on this seed-mean prediction, and Δ = metric(candidate) − metric(comparator). A
  deterministic model is its own seed mean. An ensemble's seed *s* is the mean of its members at seed *s*.
- **Per-seed pairing.** Δ_s = metric(candidate seed *s*) − metric(comparator seed *s*) for *s* = 0, 1, 2. A
  deterministic comparator is repeated for each *s*.
- Both definitions are those of the M5-X scorer (`scripts/eval/bioassess_x_final.py`).
- **Missing predictions.** A model that returns no prediction for a scored row gets its fold's training-pool mean
  of the target for that row, so that every paired comparison uses all scored rows. The number of such rows is
  reported per model.

## 8. Comparators and parity (references first)

Comparators are built before or with the candidates (decision rule 3). No candidate validation number on French
development rows is produced before the comparator table for that track exists.

1. **Simple:** training-pool mean; month climatology; geography-only RF (lat, lon, altitude); nearest training
   station.
2. **Published transfer functions:** WA-PLS2; kNN-MAT; and **MEMLM** (Sun, Holden & Birks 2024, Clim. Past
   20:2373–2398, doi:10.5194/cp-20-2373-2024), our reimplementation of the paper's stacked ensemble of random
   forest, extremely randomised trees and LightGBM (member list as summarised in the M5 plan from the paper's
   abstract) on the community features, within the RF row budget. The stacking rule is taken from the paper
   before the first fit and recorded in the comparator table. It is labelled "MEMLM
   (reimplemented)"; no fidelity check against the authors' results is possible. If the paper's description is
   not sufficient to implement it, MEMLM is "not compared, with reason".
3. **bio.infer** 1.3.6, run itself through its own OTU pipeline (the M5-X fidelity check stands; ITIS coverage of
   French taxa is reported). It stays eligible whatever its rank; weak results are "compared, weak".
4. **GJAM is not a comparator: "not compared, with reason".** In M5-X its inverse-prediction routine failed the
   pre-declared fidelity check (r ≥ 0.90 against gjam's own in-sample predictions) on fold 0 of each block, at
   100 taxa: in-sample r 0.73–0.88 on X-L and 0.83–0.92 on X-U (M5-X amendment v6; result files
   `sentinel_runs/bioassess/x_track/dev/r_out/gjam/X-L/fold00_info.txt` and `.../X-U/fold00_info.txt`,
   `fidelity_pass` FALSE in both; recorded gjam fit times 99.3 min and 114 min). It is not re-attempted.
5. **Index regressions (practitioner baseline).** F-A/F-T: the national indices delivered with the operation —
   I2M2 (7613) and, where delivered, IBGN (1000), the "équivalent" index (5910) and the large-river index
   (6951); IBD (5856) and IPS (1022) where a diatom operation is attached; I2M2 component metrics only if they
   are delivered as index results — as linear inverse calibration and as RF, each with and without geography.
   F-Z: indices computed identically by our code on both countries from harmonised taxa (richness, EPT richness,
   EPT share, Shannon), fitted the same way. A published score table (for example BMWP) is added only if obtained
   under a licence recorded before the freeze (a human decision that the lead raises); measured values only.
6. **ML baselines:** RF, LightGBM, multi-output RF (Kocev et al. 2013) and TabPFN v2 (installed 2.2.1), each on
   [community] and [community + context].
7. **Baseline ensembles:** RF[c+ctx] + TabPFN; RF[c+ctx] + LightGBM + TabPFN; the mean of the three strongest
   single baselines on development.
8. **Hybrid controls:** TabPFN v2 with PCA(32) of the Hellinger features in place of the learned embedding and a
   random context; the same with a kNN-retrieved context. Both use the query grouping of §9.2.
9. **Region-context controls:** region-descriptor features for every ML baseline of item 6; and the strongest
   baseline ensemble plus a region-offset model (ridge regression of pseudo-region mean residuals on the
   descriptors, fitted on training pseudo-regions). The descriptor of a row is the PCA-20 projection of the mean
   Hellinger profile of its region-context rows (the mirror rule of §9.1, the same rows as the candidates). The
   Hellinger taxon list and the PCA basis are fitted on training pseudo-regions only.

**F-Z comparator inputs (lead ruling (a)).** The F-Z candidates take no coordinates (§9.1). The F-Z comparator
list is not reduced because of that. In F-Z the following are named comparators in their own right, next to the
[community] versions: geography-only RF (lat, lon, altitude); nearest training station; and, for every ML baseline
of item 6 and every ensemble of item 7, both a coordinate-using input set [community + lat, lon, altitude, day of
year] and a coordinate-free one [community + altitude, day of year]. Neither replaces the other, and the candidate
must beat every one of them ("Every listed comparator", below).

**Parity, item by item.** Whatever a SENTINEL candidate receives, every comparator may receive:

| Candidate receives | Comparators receive |
|---|---|
| A second assemblage | The same diatom Hellinger features (taxa in ≥ 10 training samples), missing where no diatom operation is attached |
| Diatom-only training rows | The ML baselines of item 6 may receive them too, with the invertebrate features missing (missing where the learner accepts it, else 0 plus an indicator). WA-PLS2, kNN-MAT, MEMLM, bio.infer and the index regressions are invertebrate methods and do not |
| Auxiliary targets | Multi-output RF on the same target set; per-target learners may add out-of-fold predictions of the auxiliary targets as features |
| Context | The same context columns, from the closed list of §9.1. Chemistry-derived station attributes are excluded for everyone |
| Pool prevalence (§9.1) | Every comparator that takes context may receive the same values, built by the same code: one extra column per feature taxon (the same grid, radius, leave-one-out and missing rule), or out-of-pool taxa set to missing (0 plus an indicator where the learner does not accept missing values). The variant is chosen per target on development |
| Unlabelled test-region inputs | Item 9, on the same region-context rows |
| Training rows | The same pools and fold rules. A comparator may choose per target, on development, between France-only and France + UK (F-A, F-T), and between UK-only and UK + NRSA (F-Z) |
| Seeds and averaging | 3-seed bagging or averaging; ensembles with the same number of members |
| Predictive distribution for events | Threshold-on-baseline with the residual spread estimated on development rows |

**Row budgets.** TabPFN v2 ≤ 6,000 context rows and RF/LightGBM ≤ 60,000 training rows, by a seeded subsample
stratified by source × region (the M5-X budgets). The hybrid has the same TabPFN budget. The budgets may be
raised for both sides together after a VRAM test on UK rows, before any French validation number.

**TabPFN version policy.** The comparator and the hybrid's backbone are the same model: TabPFN v2 as installed
(package `tabpfn` 2.2.1, v2 weights). **TabPFN-2.5 and any later release are not used**: their licence needs a
human decision, which the lead raises. If a human accepts a licence before the first fit on French development
rows, a dated entry in a stamped file adds that release as a comparator. No version changes after French
development starts. Every M5-F SOTA label carries the sentence "TabPFN-2.5 and later releases were public and
were not compared (licence)".

**Comparator variants.** For each named comparator and target, the variant (training pool, diatom block, region
descriptors, pool-prevalence columns, auxiliary out-of-fold features) is chosen on French development, seed 0.
Coordinate use is not a variant in F-Z: both input sets are named comparators (above). At test each named
comparator is fitted once per target in its chosen variant, with seeds 0, 1, 2 where it is stochastic. The list of
named comparators and chosen variants is written in the freeze candidate.

**Per-target comparator (enters the Holm family).** For each track, target and co-primary metric: the strongest
comparator on development for that metric. A comparator that is strongest anywhere is never dropped.

**Every listed comparator (enters the label).** At test the candidate is also compared, by a paired 95 % CI on the
same metric, with **every** comparator of items 1–3 and 5–9 that applies to that track and target. This covers the
strongest published method and the strongest simple baseline of RESEARCH_PLAN §15.1. Row 1 of the decision table
(§10) needs every one of these lower bounds above 0. **No comparator and no variant is chosen at test.** A
comparator "applies" when it can be fitted for that track and target under the pools of §3; one that cannot is
listed "not compared, with reason" in the freeze candidate, before the look.

## 9. SENTINEL candidates, development, selection

### 9.1 BioSet-MA (new module; BioSet is not edited)
A set transformer over one or two assemblages with multi-target predictive distributions. The recipe is fixed
here and is not tuned on French rows.

- **Tokens:** flat taxon embedding (d = 96) + lineage-sum embedding + modality embedding + the 4 BioSet abundance
  features + pool prevalence with its missing flag. Caps: 150 invertebrate and 100 diatom tokens, most abundant
  kept. Lineage embedding tables are initialised with standard deviation 0.1, the taxon-name table with 1.0.
- **Pool prevalence (lead ruling (f)).** Built from taxon lists and coordinates of the fold's training rows only;
  never from a label, never from a locked-operation input.
  - Grid of 50-km cells. The pool of a sample is every training sample in a cell whose centre lies within 200 km
    of the centre of the sample's own cell. Prevalence of a taxon = the share of pool samples that list it.
  - A training row is left out of its own pool (leave-one-out). A pool of fewer than 10 samples gives a
    "missing" flag instead of a value.
  - At prediction the pool of a row is still made of training samples only. **No unlabelled French test input
    enters this feature.** In F-Z the pool holds UK samples only, so a French row whose cell centre is more than
    200 km from every UK training cell gets the missing flag.
  - Source-token dropout also sets the prevalence to missing and the protocol tokens to unknown.
  - The same values are offered to every comparator that takes context (§8 parity).
- **Encoder:** 2 pre-LN self-attention blocks, 4 heads, weights shared across modalities; PMA(1) ⊕
  abundance-weighted mean per modality.
- **Fusion:** 1 transformer layer over [modality vectors, context token, optional region token]; modality
  dropout 0.3, always keeping one. Source-token dropout 0.1 so that an unseen source (F-Z) maps to a learned
  "unknown" token.
- **Context columns (closed list, the same for every model and comparator):**
  1. day of year of the operation, as sine and cosine;
  2. latitude and longitude of the station (WGS84);
  3. station altitude in metres (France: `altitude_point_caracteristique` of the station record with the same
     SANDRE code; UK: `site_altitude_m`), with a missing flag;
  4. source token (France, UK, NRSA);
  5. field-protocol token per modality (France: `code_methode`; UK: `analysis_method`);
  6. offset in days of the attached diatom operation.

  Nothing else is an input. In particular: no hardness (`durete`) or other chemistry-derived attribute, no status
  class, no catchment area or distance from source, no water depth or width at sampling, no typology or
  hydro-ecoregion code, no year, and no water-body, watercourse or network code (these codes serve only pairing,
  folds and the sensitivity strata of §4).

  **F-Z candidates take no latitude or longitude (lead ruling (a)).** For #7, #8 and the two tabular members of
  #8 the context is items 1 and 3 and the tokens 4–6: coordinates learned on UK rows are out of range in France.
  Coordinates are still used, for every model, to build pools, pseudo-regions, inner-validation blocks and
  sampling strata; there they are not model inputs. The F-Z comparators keep their coordinate-using versions
  (§8).
- **Trunk and heads:** 2 × 256 GELU, dropout 0.1; per target a Student-t (ν = 4) location and log-scale with a
  per-source scale offset; masked negative log-likelihood over the targets of §6 in both windows. The log-scale is
  clamped to [−6, 4]; the scale head starts at zero.
- **Training:** AdamW, lr 1e-3, weight decay 1e-2, batch 256, token dropout 0.1, fp16 autocast.
- **Inner validation (lead rulings (g), (m)).** Inner-validation rows are training-role rows only. Whole 50-km
  blocks are held out, per country, until 15 % of that country's location clusters are in validation. `ma_inv`
  drops diatom-only rows.
- **Early stopping.** At most 100 epochs, patience 10, on the column-balanced NLL (the mean over columns of the
  column-mean NLL) of the primary target columns of the tracks the fit serves, on the inner-validation rows: the
  F-A and F-T primary columns together for #1–#4, the F-Z primary columns for #7.
- **Resumable fits (lead ruling (o)).** Batches are assembled on the GPU, and a long fit runs as chunks of at most
  100 minutes that resume from a saved state. A chunk is a fixed number of epochs, set once per configuration from
  a timing run on UK rows before any validation number. The seeds and the chunk schedule are the same for every
  fold and seed. Resume points are logged in the registry. A resumed fit is not guaranteed to equal an
  uninterrupted one bit for bit; this is disclosed. The checkpoint named in the freeze file is hash-pinned, and it
  is what is scored.
- **Region-context module (episodic; lead ruling (k)).** A region token is attention-pooled from the community
  vectors of at most 256 other samples of the same region.
  - Training episodes: each epoch, a grid of random side 50–150 km and random origin cuts the training rows into
    pseudo-regions. Chunks of ≤ 64 rows of one pseudo-region are packed into batches of ≤ 256 rows (about 4
    chunks per batch). The chunk is the context group.
  - Query = the row's own community vector plus a learned seed. Keys = the other rows of the group only: never
    the row itself, never a row of another group. A row with no other row in its group gets no region token (the
    slot is masked).
  - Inner-validation regions: a fixed 100-km grid over the inner-validation rows.
  - **No context row supplies a label, in training or at prediction.** The region token is pooled from community
    inputs only. (In TabPFN and in the hybrid, context rows do carry labels: they are training-role rows only,
    never a locked or test operation, and in F-Z never a French row.)
  - **Prediction-time context rows are unlabelled test-region inputs.** This is transductive (use (c) of §2.3
    item 6). It is disclosed in every label and report. It is the same for all models: the comparator controls
    of §8 item 9 use the same rows.
  - **No-region-context ablation, always reported.** F-A and F-T: #2 `ma_full`, the same recipe trained without
    the module. F-Z: #7 and #8 predicted again with the region slot masked (the same weights, no region-context
    row; an inference-time ablation, not a new configuration). Caveat: the F-Z weights were trained with a region
    token in nearly every batch, so this ablation may understate a model trained without the module.
- **Region-context rows: one mirror rule, used in development and at test** (use (c) of §2.3 item 6).
  - Candidates for a row *r*: the eligible invertebrate operations (§3) in the same region unit (§4, after
    merges) and the same scored period as *r*. Scored period: role `test` (2023-01-01..2025-12-31) in the test
    design; 2019-01-01..2021-12-31 in the French development design; the pseudo-test period of the held-out area
    in the UK development design.
  - *r* itself is excluded. Buffer-year and training rows are never region-context rows of a scored row.
  - If a unit has more than 256 candidates: they are sorted by row key (byte order) and one sample of 256 is
    drawn without replacement with `numpy.random.default_rng([20261110, k, s])`, where *k* is the index of the
    region unit in the split file and *s* the model seed. Every row of the unit uses that sample; a row that is
    in the sample uses its other 255 rows.
  - Only inputs are used (taxon lists and context columns). The comparator region descriptors of §8 item 9 use
    the same rows.
- **Physics-coherence head (lead rulings (e), (l)).** Hinge penalties on the predicted locations in original
  units. Penalty = the sum over constraints of the row-mean hinge; weight λ = 0.1. Constraints, per label window:
  - total P ≥ orthophosphate-P, on all rows;
  - total oxidised N ≥ nitrate-N, on UK rows only; Kjeldahl N ≥ ammonium-N, on French rows only. All of these
    labels are in mg N/L or mg P/L; the unit consistency is unit-tested;
  - DO consistency: |log10 DO (mg/L) − log10(saturation / 100 × solubility)| ≤ 0.06, with the predicted
    temperature clamped to 0–35 °C and the predicted saturation clamped to 1–300 %.

  Solubility is the fresh-water equation of Benson & Krause (1984), through the shared routine
  `sentinel/models/hydromet/physics.py::do_sat_mgl` (called, or unit-tested equal to it), at pressure
  P = (1 − 2.25577e-5 · h)^5.25588 atm for station altitude *h* in metres (standard atmosphere). By the lead's
  check, without this factor the solubility error is about 0.03 log10 near 600 m and reaches the whole tolerance
  near 1,150 m. Where altitude is missing no correction is made (P = 1); such rows are flagged and counted. A unit
  test checks the Benson–Krause constants against the cited source. No ion latent is inferred.

### 9.2 TabPFN hybrid (new module)
TabPFN v2 (the comparator's installed model and call) on [top-K Hellinger taxa per modality, the 32-dimensional
BioSet-MA sample embedding of the same fold, context]. The constants below are fixed (lead ruling (h)).
- **Taxa.** K = 200 invertebrate and 100 diatom taxa (`mixed_otu` unit): the most prevalent among the fold's
  training samples, each in at least 10 of them. Values are Hellinger-transformed, and missing where the row has
  no sample of the modality.
- **Embedding.** The 32 dimensions are the first 32 principal components of the 256-wide trunk output of the
  fold's `ma_full` model at the same seed. The PCA (mean, components, fixed sign convention) is fitted on the
  fold's training rows only. The embedding model is trained on the fold's training rows only and loaded through
  the guarded loader.
- **Query grouping** (use (d) of §2.3 item 6). The rows to predict in a fold are grouped by k-means on their
  embeddings, standardised with training-row statistics: k = ⌈n / 500⌉, 4 initialisations, seed = model seed, any
  group above 500 rows split in two until none is.
- **Context per group and target:** half the row budget from the labelled training rows nearest to the group
  centroid, half a seeded random sample of the rest stratified by source × region. Context rows are labelled
  training-role rows only.

### 9.3 The eight configurations (cap 8, all counted)
A configuration is one training recipe. One fit of a configuration predicts both windows, so F-A and F-T share
fits. F-Z needs different training rows and is counted separately.

| # | Name | Recipe | Tracks |
|---|---|---|---|
| 1 | `ma_inv` | BioSet-MA, invertebrate tokens only | F-A, F-T |
| 2 | `ma_full` | BioSet-MA, invertebrates + diatoms | F-A, F-T |
| 3 | `ma_full_rc` | #2 + region-context module | F-A, F-T |
| 4 | `ma_full_coh` | #2 + physics-coherence head | F-A, F-T |
| 5 | `pfn_hybrid` | TabPFN hybrid on the #2 embedding | F-A, F-T |
| 6 | `ens_f` | mean(one BioSet-MA member, RF[c+ctx], TabPFN v2), the form that won M5 | F-A, F-T |
| 7 | `ma_zs` | #3 trained on UK allowed rows only, without latitude or longitude (same-continent, fixed a priori from the M5-X result that same-continent pools beat multi-source pools) | F-Z |
| 8 | `ens_zs` | mean(#7, RF[c+ctx], TabPFN v2), context without latitude or longitude | F-Z |

- **`ens_f` member.** One BioSet-MA member for both tracks: the configuration among #1–#4 with the highest mean
  of the F-A and F-T selection scores (§9.5); ties within 0.002 go to the lower number. #3 can be the member only
  if it passes G7. Gate G6 does not remove a configuration from membership.
- **`ens_zs` member pools.** All three members are trained on the **UK allowed rows only** (the rows of #7, the
  same budgets). No NRSA row and no French label enters any member.
- The list is closed. A configuration may be replaced, by a dated entry in a stamped file, only before it has
  produced any validation number on UK or French development rows. Once it has, it counts even if abandoned.
- Bug fixes that leave the recipe unchanged are logged and are not new configurations.
- Ablations run seed 0. The confirmatory candidate of each track and its members run seeds 0, 1, 2 before the
  freeze candidate is written.

### 9.4 Development
- **Before the stamp:** unit tests on synthetic data and on allowed UK pool rows (shapes, masking, guards,
  gradient flow, VRAM). No validation number.
- **After the stamp, UK development design:** the M5-X X-U development design (UK ≤ 2015 training, UK 2016–20
  `pool_fit` pseudo-test, leave-one-area-out), restricted to allowed rows. It is **mandatory for #7 and #8**,
  which are selected on it (§9.5), and optional for #1–#6, where it measures time and memory and confirms that
  the recipes train. Its numbers are reported and count toward the cap. It cannot select the confirmatory
  candidate of F-A or F-T.
- **French development:** the development design of §3 on the folds of §4. Comparators first.
- **"Zero-shot" (F-Z)** means no French label in any fit, early stopping, scaler, calibration or candidate
  selection. French development labels are used in F-Z only to choose the comparators and their variants. This
  asymmetry favours the comparators and is stated in every F-Z label (§5). #7 and #8 are also scored on the
  French development pseudo-test, for the report and for the comparator table, after C-FZ is fixed.

### 9.5 Selection rule
- **Selection score (F-A, F-T)** = mean over the track's primary targets of (pooled R² + `r2_anomaly`) / 2 on the
  French development pseudo-test, seed 0.
- **Selection score (F-Z)** = the same formula on the UK development pseudo-test of §9.4, over the F-Z primary
  targets that have UK labels, seed 0. **No French label enters the choice of the F-Z candidate.** The score is
  computed after G4, over the final target list, from stored seed-0 predictions.
- **One confirmatory candidate per track.** C-FA and C-FT: the eligible configuration among #1–#6 with the
  highest selection score for that track. C-FZ: the higher of #7 and #8. Scores within 0.002 are a tie, won by
  the lower-numbered configuration.
- Up to two further configurations per track are secondary. All others are reported from development only.

## 10. Confirmatory family, decision table, power

**Family.** Confirmatory candidate vs the per-target comparator, on pooled R² and on `r2_anomaly`, for F-A, F-Z
and F-T: 3 tracks × 6 targets × 2 metrics = **36 tests**, Holm at α = 0.05 over the whole family. One-sided rule
as in M5-X: p = two-sided paired location-cluster bootstrap p (B = 10,000) if Δ > 0, else 1. Δ is the seed-mean
statistic of §7.

**Decision table** per track × target × metric (first matching row):

| # | Condition | Label |
|---|---|---|
| 1 | Holm-rejected ∧ per-seed Δ > 0 for seeds 0, 1, 2 ∧ region-unit paired CI > 0 ∧ **paired 95 % CI lower bound > 0 against every listed comparator that applies** (§8) | **SOTA (M5-F, track, metric)** |
| 2 | Holm-rejected, row 1 not met | **Above, not robust** (each failed condition is stated; every comparator not beaten is named) |
| 3 | 95 % paired CI upper bound < 0 (against the per-target comparator) | **Below** |
| 4 | CI includes 0 ∧ realised Holm-worst MDE > 0.02 | **Inconclusive (underpowered)** |
| 5 | otherwise | **Competitive** |

- The CIs against the listed comparators are paired location-cluster bootstrap CIs (B = 10,000, the same
  resampling draws for every comparator), unadjusted, on the row's metric and the seed-mean statistic of §7.
- If fewer than 6 region units remain after the merges of §4, the region-unit condition is "not evaluable" and
  row 1 cannot be met.
- **Multiplier.** For a family of *m* tests the Holm-worst multiplier at 80 % power is
  z(1 − 0.05 / (2*m*)) + z(0.80): 4.039 for *m* = 36, 3.833 for *m* = 18, 3.707 for *m* = 12. Realised Holm-worst
  MDE = multiplier × (paired CI width / 3.92), with *m* the realised family size. The smallest effect of interest
  stays 0.02.
- **Target-level label.** "SOTA on target *t* in track *k*" needs row 1 on at least one co-primary metric and not
  "Below" on the other. The metric is always named.
- **F-Z transfer condition.** A SOTA label in F-Z also needs the candidate's own 95 % CI for that metric to lie
  above 0. If no model meets this, the target is reported as "no transfer".
- Secondary and exploratory analyses are unadjusted and labelled as such.

**Power (availability counts only).** The proxy is the M5-X one: the Holm-worst MDE of 0.015–0.024 at 2,149
clusters and α/12, rescaled to α/36 (multiplier 4.039 instead of 3.707) and by √(2,149 / n). Projected Holm-worst
MDE at 80 % power:

| Test location clusters with the target | 650 | 1,000 | 1,500 | 2,000 | 3,000 |
|---|---|---|---|---|---|
| MDE range | 0.030–0.048 | 0.024–0.038 | 0.020–0.031 | 0.017–0.027 | 0.014–0.022 |

- **Power gate.** A target enters the family only with ≥ 650 test location clusters, the count at which the low
  end of the range equals 0.03 (about the largest robust gain seen in M5). Below that it is descriptive.
- **Underpowered a priori.** The smallest effect of interest (0.02) is reached at about 1,435 clusters at the low
  end of the range and about 3,673 at the high end. A target that enters the family with fewer clusters than that
  is underpowered a priori for 0.02. A null result there is reported as "not shown", never as "no difference".
- **Assumption.** The scaling assumes that the per-cluster variance of the paired difference seen in M5-X (NRSA
  and UK rows, `r2_anomaly`) transfers to France. It may not: samples per cluster and label noise can differ.
  The realised CI widths feed row 4.
- The table is recomputed in `power_f_v1.json` from the availability masks at the split freeze, with the pooled-R²
  proxy from frozen M5 Test A added. No French label value enters it.
- **Honest expectation** (plan §9, judgement): P(SOTA on ≥ 1 target) about 0.35–0.45 for the F-A ensemble, about
  0.10 for a BioSet-MA single model, about 0.20 for F-Z. The every-comparator condition of row 1 was added after
  these figures were written and can only lower them.

## 11. Gates and kill rules (all fixed now)

| Gate | Rule | If it fails |
|---|---|---|
| G0 stamp | No French download, and no validation number on a new arm, before the lead's stamp of this file | — |
| G1 disk and cap | `df -h /home/bcheng` before each step; ≥ 60 GB free; size check of §2.1 before each pull; M5-F new data ≤ 5 GB including transient files | Stop and report |
| G2 lock and codes | `bioassess_f_lock.json` exists, is read-only and is logged; `bioassess_f_codes.json` holds the verified request codes with URL and sha256 | No chemistry request |
| G3 feasibility | ≥ 650 paired test location clusters for at least 4 targets of the final F-A list (after the pairing choice of §3 and the replacement rule of §6.3; masks only) | M5-F is reported as not feasible; nothing is fitted on French rows |
| G4 split freeze | Split file (with pairing choice, merges, distance report), `bioassess_f_codes.json` (label rules, code counts, method codes), event thresholds and `power_f_v1.json`, stamped by the lead | No fit on French rows; no French chemistry value read |
| G5 references first | Comparator development table exists for a track | No candidate validation number on that track |
| G6 BioSet-MA kill rule | Best BioSet-MA single model (#1–#4) is more than 0.02 below the best single comparator on the selection score components for ≥ 4 of 6 F-A primary targets | #1–#4 cannot be a confirmatory candidate; they remain ensemble members |
| G7 region-context gate | `ma_full_rc` ≥ `ma_full` and ≥ the region-offset baseline ensemble on mean development pooled R² | #3 is not eligible for F-A/F-T. (#7 and #8 are judged on the UK development design only) |
| G8 hybrid gate | `pfn_hybrid` ≥ both hybrid controls on the selection score | #5 is not eligible; reported as negative |
| G9 events | `events.power_flag` (≥ 30 positives, ≥ 10 clusters) | Event is descriptive |
| G10 compute | Jobs ≤ 2 h, GPUs 0–9 through the lock helper with `--min-free-mb`, never a GPU with another user's process; CPU ≤ 6 threads, one heavy CPU job; about 30 GPU-h planned for development; long fits run as resumable chunks of ≤ 100 minutes (§9.1). The one-time look refits every listed comparator; its budget is written in the freeze candidate from measured development timings | Stop and report at 45 GPU-h of development, or at 1.5 × the look budget |
| G11 freeze | Freeze candidate → independent pre-test review → git commit and public push exist → lead-written FREEZE → one look | No test read |

Any change to a rule above is a dated entry in a stamped file, made before the data it affects are seen.

## 12. Freeze and single look

- New guard module `ftestlook.py` with the campaign rule-7 hardening, as in M5-X amendment v7–v7b: only the
  lead-written FREEZE is accepted; scoring-code hash over the full import closure (including `events.py` and
  `fevents.py`); content-hash pins of every data file including the locked store; all artifacts verified before
  the claim; `O_EXCL` claim tied to pid/host; access closed at sealing; sealed registry row; rescore refused once
  a scoring row exists; BaseException and SIGTERM recorded as aborts; one resume.
- GPU work in the look runs in short label-free worker processes, as in M5-X.
- At the look every listed comparator of §8 is fitted in its frozen variant, and the scorer writes one paired CI
  per comparator, track, target and metric.
- Locked labels are read only inside the look. The write-up goes to `bioassess_f_test_results.md`, never into a
  frozen file.
- Registry (workstream `bioassess`): `bioassess_f/prereg_snapshot` (the stamp), `bioassess_f_dev`,
  `bioassess_f_events_dev`, `bioassess_f_test`. Every configuration is logged there and in `TRAINING_LOG.md`.

## 13. Items that are UNVERIFIED today

Verified on 2026-11-10 from documentation, and no longer on this list: the SANDRE parameter, fraction, unit,
support and index codes of §6.3; the remark, qualification, in-situ and status code lists; the routing date field;
the API paging limit; the licence strings; the French good-status limits of §6.4.

1. French pairing rate, parameter coverage, co-sampling rate of diatoms, and the pesticide panel (gate G3).
2. Whether biology and chemistry stations share SANDRE station codes in practice (§3 fallback).
3. Which fractions, units and method codes are actually used, and how often (counts at G4, §6.3).
4. Whether the `fields` parameter is honoured by every endpoint, and the bytes per record (size check, §2.1).
5. The size of the national bulk export (not used; §2.1).
6. Whether I2M2 component metrics are delivered as index results, and their codes (§8 item 5).
7. Availability of a level-1 hydro-ecoregion layer and its licence (§4).
8. Whether UK and French diatom data are comparable enough for a shared diatom modality. UK diatom rows are
   light-microscopy analyses only (§3), but taxonomy and counting conventions may still differ. The `ma_inv`
   versus `ma_full` contrast on French development shows whether the diatom modality helps.
9. The river-system splits listed in §4 (reviewer's reading, not checked against a network file).
10. Availability and licence of SPEAR classifications and of any published score table (human decision).
11. TabPFN v2 memory use at French feature counts on 11 GB cards.
12. A dedicated novelty search for BioSet-MA and the hybrid is still owed before any "novel" wording.

## 14. Cross-workstream rules and licence

- **One pull, one manifest, one licence record.** The Naïades / Hub'eau pull of §2.1 is the only French pull of
  the project. It is owned by M5 (SYNTHESIS §3.3 France row, X5).
- **Role partition before download.** The lock file and the split file give every French row exactly one role
  (training, development, buffer, test, future). They are written before any chemistry is requested.
- **M7 P3 is deferred.** If it ever runs, it may use these rows for evaluation only, under the same roles, never
  for fitting.
- **Fuse never combines M5-F and M7-P3 outputs on overlapping rows.**
- **F-TOX uses measured ECOTOX values only**, never M4 predictions (C4).
- No other workstream reads the locked store.
- **Licence register.** `docs/licenses.md` carries the Naïades / Hub'eau entry (Licence Ouverte Etalab 2.0,
  attribution, planned, not downloaded). The per-file manifest of §2.1 is added at download.

## 15. Change log: draft v0 → v1 (2026-11-10, before the stamp and before any French data)

### 15.1 Pre-stamp review items

| Review item | Change | Section |
|---|---|---|
| Must-fix 1, comparator test | 36-test Holm family kept against the per-target comparator. Row 1 also needs a paired 95 % CI lower bound > 0 against every listed comparator that applies; row 2 names each comparator not beaten; no comparator chosen at test | §8, §10, §12 |
| Must-fix 2, locked store | Every raw response lands in the locked store first; extraction by column whitelist and date; routing field `date_prelevement`; missing or unparseable dates stay locked; chemistry-derived attributes (`durete`) and status classes never extracted | §2.3 items 1, 3 |
| Must-fix 3, label construction | Eleven rules written, each with its UK counterpart; "≥ 6 analyses" is per parameter and counts observations; code-dependent lists frozen at G4 from documentation and counts in a lead-stamped file | §6.2, §6.3, G4 |
| Must-fix 4, route and codes | Hub'eau API route by department or station batch × parameter × year; size check before each step; transient files count toward the cap; verified codes with URL and sha256 before the first chemistry request | §2.1, §2.2, G1, G2 |
| Must-fix 5, transductive uses | Closed list (a)–(d); nothing else; PCA basis on training pseudo-regions only; one mirror rule for region-context rows with seed and order; label wording | §2.3 item 6, §8 item 9, §9.1, §9.2 |
| Must-fix 6, seeds | Statistic on the 3-seed mean prediction on both sides; per-seed Δ pairs seed *s* with seed *s*; the M5-X definition | §7, §10 |
| F-Z selection ruling | C-FZ chosen between #7 and #8 on the UK development design, without French labels; comparators still chosen on French development labels (disclosed); label wording | §5, §9.4, §9.5, G7 |
| Should-fix 1 | Closed list of context columns; chemistry-derived attributes excluded | §9.1, §8 parity |
| Should-fix 2 | F-Z label wording; `ens_zs` member pools | §5, §9.3 |
| Should-fix 3 | Diatom attachment restricted by the pool's end date; tie rule | §3 |
| Should-fix 4 | Which comparators receive diatom-only rows | §8 parity |
| Should-fix 5 | Underpowered-a-priori statement; variance-transfer assumption | §10 |
| Should-fix 6 | Multiplier formula | §10 |
| Should-fix 7 | Full merge order; floor of 6 region units | §4, §10 |
| Should-fix 8 | Pairing-fallback criterion; "listed" versus final primary targets; UK labels of replacement targets | §3, §6.3, G3 |
| Should-fix 9 | Row eligibility: supports, result type, date floor 2007-01-01, field protocols, unmatched taxa | §3 |
| Should-fix 10 | French events as `EventDef` objects in a new module; reused codes named; thresholds in label units; window and P90 pool stated | §5 (F-E row), §6.4 |
| Should-fix 11 | MEMLM added as a comparator, or "not compared, with reason" | §8 item 2 |
| Should-fix 12 | Distance report, ≥ 50 km stratum, same-watercourse stratum; wording on water agencies corrected | §4 |
| Should-fix 13 | Derivation of the 32-dimensional embedding; `ens_f` member fixed across tracks | §9.2, §9.3 |
| Should-fix 14 | Registry key `bioassess_f/prereg_snapshot`; stamp wording | header, §12 |
| Should-fix 15 | Cross-workstream rules; measured ECOTOX values only | §6.5, §14 |
| Should-fix 16 | `docs/licenses.md` entry; the guard is procedural plus `ftrack` | §2.3 item 7, §14 |
| Should-fix 17 | Resolved by lead ruling (b): UK sample dates ≤ 2020-12-01, so no ±30-day window reaches 2021 | §3 |
| Should-fix 18 | "TabPFN-2.5 and any later release"; hybrid backbone version | §8 |
| Questions 1–12 | Family 36; department groups primary with HER-1 and ≥ 50 km sensitivities; lock boundary 2022-01-01; cap accounting unchanged; F-Z pool asymmetry disclosed; documentation fetches logged; French events in a new module; GJAM wording with result files; TabPFN wording; SPEAR and score tables licence-gated, measured values only; diatom window ±90 days with tie rule and \|offset\| ≤ 30 sensitivity; file name kept | §4, §2.3, §8, §6.4, §6.5, §3 |
| Good-status limits | Verified from the December 2023 guide, Annex 6; French limits added as events; 0.1 mg/L total P and pH 6.5–9 kept as M5-X-comparable events and labelled as such | §6.4 |
| Display name | SENTINEL-Biota in prose | header |

### 15.2 Lead rulings on the implementation choices (a)–(o) applied

| Ruling | What v1 says | Section |
|---|---|---|
| (a) | F-Z candidates take day of year and altitude, no latitude or longitude. The F-Z comparator list keeps coordinate-using and coordinate-free versions as separate named comparators; the candidate must beat every one | §9.1 context columns, §9.3, §8 "F-Z comparator inputs" |
| (b) | UK biological sample dates ≤ 2020-12-01 (invertebrates and diatoms); `wq_observations` ≤ 2020-12-31 unchanged; dropped counts reported (0 and 0) | §3 |
| (c) | UK diatom modality: light-microscopy (`LABP`) analyses only; counts stated and fixed before any fit | §3, §13 item 8 |
| (d) | UK diatom sites in a 50-km block that holds a `reserved_test` site are excluded | §3 |
| (e) | DO coherence: tolerance 0.06 log10; clamps 0–35 °C and 1–300 %; altitude pressure factor (1 − 2.25577e-5·h)^5.25588, rows with missing altitude flagged and counted; constants unit-tested against the source; shared routine `physics.do_sat_mgl` called or tested equal | §9.1 coherence head |
| (f) | Pool prevalence: taxon lists of training rows only, leave-one-out, missing flag below 10 samples; no unlabelled French input enters it; offered to every context-taking comparator | §9.1 pool prevalence, §8 parity |
| (g) | Early stopping on the column-balanced NLL of the primary columns; ≤ 100 epochs, patience 10; log-scale clamp; scale head starts at zero; inner-validation rows are training-role rows only | §9.1 |
| (h) | Hybrid constants: K = 200 and 100 taxa, prevalence ≥ 10, PCA(32) embedding, context half nearest and half stratified, k-means with 4 initialisations, groups ≤ 500 rows | §9.2 |
| (i) | Lineage tables initialised at standard deviation 0.1, name table at 1.0 | §9.1 tokens |
| (j) | A diatom operation attached to a training row is itself unlocked and not later than the pool's end date | §3 diatom attachment |
| (k) | Region context: episodes, query and keys, inner-validation regions. No context row supplies a label. Prediction-time context rows are unlabelled test-region inputs: transductive, disclosed, the same for all models. No-region-context ablation reported | §9.1 region-context module, §2.3 item 6, §8 item 9 |
| (l) | Coherence constraints by source; penalty = sum of row-mean hinges, weight 0.1; N and P unit consistency unit-tested | §9.1 coherence head |
| (m) | Inner validation: whole 50-km blocks, 15 %, per country; `ma_inv` drops diatom-only rows | §9.1 |
| (n) | Plausibility ranges identical for every role, model and comparator; counts per role (locked rows: counts only); nitrite factor 0.3045; P90 and P10 window statistics only through a stamped deviation before any number | §6.3 rule 9, §6.3 codes, §6.2 |
| (o) | Resumable chunks of ≤ 100 minutes, batches assembled on the GPU, same seeds and chunk schedule, resume points in the registry, the hash-pinned checkpoint is what is scored | §9.1, G10 |

Ruling A (M5-X scorer: every listed comparator fitted at test) concerns `bioassess_x_protocol.md` and is recorded
there. The same rule is in §8 and §10 of this file.

**Choices the engineer made while applying the rulings** (for the lead to accept or change at the stamp):
- (a): in F-Z the coordinate-using and the coordinate-free input sets are both named comparators for every ML
  baseline and baseline ensemble. They are not development-chosen variants.
- (g): one fit serves F-A and F-T, so its early-stopping columns are the F-A and F-T primary columns together.
- (k): the F-Z no-region-context ablation is an inference-time ablation, because the cap of 8 configurations is
  full.
- (o): a chunk is a fixed number of epochs per configuration, set from a timing run on UK rows.
- Must-fix 2: the biology input columns of locked operations are extracted to a separate "locked-operation inputs"
  table, because uses (a)–(d) of §2.3 item 6 need them. Chemistry of locked rows never leaves the locked store.
- Must-fix 3: "≥ 6 analyses" counts observations, as the UK code does; a distinct-dates sensitivity is reported.
- Should-fix 11: MEMLM is a reimplementation with no fidelity check.
- G10: the look budget is written in the freeze candidate from measured timings, because the look now refits
  every listed comparator.

### 15.3 Not done in v1, with the reason

| Item | Reason | Where it is handled |
|---|---|---|
| Size of the national bulk export | The Naïades access pages are script-driven and show no file size. The route of §2.1 does not use the export | §2.1; §13 item 5 |
| Fold river-system splits checked against a network file | No network layer was fetched: fetches before the stamp were limited to documentation | §4 (distance report and two strata); §13 item 9 |
| HER-1 layer: availability and licence | Needs a data-layer download, which is not allowed before the stamp | §4; §13 item 7 |
| Source for the reason behind the 2007-01-01 floor | No fetched document states it; the reason is a recollection. The date itself is fixed | §3 |
| MEMLM member list and stacking rule checked against the paper | The paper was not fetched; the description is the M5 plan's summary. It is checked against the paper before the first fit | §8 item 2 |
| Whether `fields` is honoured by each endpoint; bytes per record | Needs a call to an endpoint that returns observation rows, which is forbidden before the stamp | §2.1 step 6; §13 item 4 |
| French pairing rate, parameter coverage, shared station codes, codes in use | Need French data | G2–G4; §13 items 1–3, 6 |
| TabPFN-2.5 licence; SPEAR and published score-table licences | Human decisions that the lead raises. Not acted on | §8; §6.5 |
| Novelty search for BioSet-MA and the hybrid | Not needed for the stamp; owed before any "novel" wording | §13 item 12 |
| TabPFN v2 memory at French feature counts | Needs the French feature set; a VRAM test on UK rows comes first | §8 row budgets; §13 item 11 |
| Code aligned with v1 | The modules were written before the rulings. Still to change, before any validation number: `ftrack.UK_ALLOWED_TO` (2020-12-31 in code; the allowed rows are the same under both cuts); latitude and longitude in the F-Z recipes; the pressure factor and the shared solubility routine in the DO-coherence term; the unit tests of rulings (e) and (l); the mirror-rule seed; resumable fits. Not yet written: `fevents.py`, the locked-store fetcher and extractor, `ftestlook.py`. No validation number has been produced with this code | §9, §12 |

## 16. Amendments

None. After the stamp, amendments are dated entries in a new stamped file, made before the affected data are
seen.
