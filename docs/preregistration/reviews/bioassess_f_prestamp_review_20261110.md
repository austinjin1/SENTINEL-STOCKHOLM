# M5-F protocol — independent pre-stamp review (verbatim)

- **Reviewed file:** `docs/preregistration/bioassess_f_protocol.md`, draft v0, sha256
  `b920816c205805176f931a17b48114f0bb8d4891364283079c430c1c8704e821` (454 lines).
- **Reviewer:** an independent AI review agent that did not write the draft; it changed no file and downloaded nothing.
- **Saved by:** the lead (AI-assisted), 2026-11-10 17:56 UTC. The text below is the reviewer's report, unedited.
- **Status:** the draft is not stamped. The lead's rulings on this review are sent to the M5 engineer; the revised
  draft is re-read and stamped separately (`bioassess_f_lead_rulings.md`).

---

**Verdict: STAMP AFTER FIXES**

Reviewed `/home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM/docs/preregistration/bioassess_f_protocol.md`, 454 lines. Its sha256 is `b920816c…c8704e821`, matching the expected value at the start and end of the review. I changed no file and downloaded nothing.

## Must-fix before stamp

1. **§8 L294-296 and §10 L376-384: the label is tested against one comparator, not both.** The decision table tests only the per-target comparator chosen on development; the strongest published method and strongest simple baseline are only "reported next to it". RESEARCH_PLAN §15.1 and `sota_campaign.md` L21 require a paired CI excluding 0 against both. Fix: name the two comparator classes and require the paired 95% CI > 0 against each at test for the SOTA row, or add a clause that any comparator beating the candidate at test blocks the label. M5-X has the same construction, so this is also a consistency ruling for the lead.

2. **§2.3 L60-63 and L72-77: raw archives holding locked values sit outside the locked store.** The biology archive contains "conditions environnementales" ("moved unopened"), and the chemistry archive exists whole before the splitter runs. Fix: every raw French archive or API response lands first in `raw/naiades/locked/`, and only whitelisted columns and rows dated ≤ 2021-12-31 are extracted out. Name the sampling-date field used for routing, and send rows with missing or unparseable dates to the locked store.

3. **§6.3 L193-203: label construction leaves open choices.** The draft fixes censoring and molar factors only. It has no rule for:
   - qualification, validation or remark codes;
   - in-situ versus laboratory duplicates;
   - several analyses on one day;
   - plausibility bounds;
   - rows with unexpected unit or fraction codes;
   - whether "≥ 6 analyses" means per parameter and distinct dates (the UK code `bio_uk_ea.py` counts observations, not dates).

   Fix: write each rule now, identical to the UK rule where one exists. Alternatively, state that all are frozen in the split file at G4 from documentation and counts only, before any French value is read.

4. **§2.1 L47, §2.2 L50-54, L199-203: the download route can break the 5 GB cap, and the codes are frozen too late.** "Bulk export is preferred", but the national physico-chemistry export is probably far above 5 GB (my estimate, unverified). The M5 plan §6 assumed a station × parameter subset of 0.2–0.8 GB. A filtered pull depends on SANDRE codes the draft marks UNVERIFIED, and they are only frozen "before any value is read". Fix:
   - pre-declare the route (API by station list × parameter list × date range, or stream-extract-delete);
   - require a size check before each pull, and count transient archives toward the cap;
   - record verified parameter, fraction and unit codes (with sha) before the first chemistry request.

5. **§2.3 L85-87 contradicts L267-269, L317-320 and L327.** Item 6 says inputs of locked operations are "never used to fit, pretrain, select features, build vocabularies or scale anything". The region-context controls, the region-context module (PCA-20 descriptors, ≤ 256 context rows) and the k-means over test-row embeddings all compute statistics on those inputs. Fix: list the permitted label-free transductive uses as explicit exceptions, give comparators the same ones, and define:
   - which rows form the region context at prediction time (test period only, 2022 buffer, or ≤ 2021);
   - the seeded sampling rule;
   - where the PCA basis is fitted.

6. **§7 L242 and §10 L372-384: the test statistic over seeds is undefined.** The draft does not say whether the Holm test uses seed-averaged predictions or the mean of per-seed metrics. It also does not say how "per-seed Δ" pairs candidate seeds with a deterministic comparator. Fix: one sentence defining the statistic entering the bootstrap, and one defining the per-seed pairing.

## Should-fix

1. **L277, L309-312:** the French context-column allow-list is not enumerated; M5-X enumerated its list. List the columns, and exclude any station attribute derived from chemistry. I recall the station record may carry mean hardness, but I am unsure.
2. **L171, L280, L343-344, L360-361:** F-Z selection and comparator choice use French 2019–21 development labels. Every F-Z label should say "zero French training rows, French development selection". State the member pools of `ens_zs`.
3. **L102-103 vs L133-134:** a December 2021 training operation can attach a 2022 diatom operation, which carries a lock key that `assert_fit_ids` refuses. Restrict unlocked operations to unlocked diatom operations, and add a tie rule (nearest date, then earlier).
4. **L95-96, L275:** state whether comparators receive the diatom-only training rows.
5. **L401-402:** the 650-cluster gate gives an MDE of 0.030–0.048. Reaching 0.02 at the low end needs about 1,433 clusters, and about 3,670 at the high end. State that targets between the gate and that size are underpowered a priori, and that a null there is "not shown". The scaling also assumes M5-X per-cluster variance transfers to France; say so.
6. **L208, L386:** the 4.04 multiplier is hard-coded. Give the formula z(1 − 0.05/(2m)) + z(0.80) for the case where the family shrinks.
7. **L157-158:** there is no rule if a merged pair is still below 100 clusters, and no floor on the number of region units.
8. **L98-100, L205-208, G3 L417:** the pairing fallback "fewer than 650" does not say for which target. "Primary targets" is ambiguous before versus after replacement. A replacement target may have no UK labels for F-Z.
9. **L93-96, L108-110:** row eligibility is not fixed. Missing: support codes, field protocol (IBGN-era versus I2M2-era), the lower date bound of the training pool, and the rule for unmatched taxon names.
10. **L214-223:** the five French events are not in `events.py` v1.0. Fix these:
    - name the reused codes (`E2a_lab`, `E5b`, `E6_snap`);
    - state thresholds in label units (50 mg NO3/L = 11.3 mg N/L; 0.5 mg PO4/L = 0.163 mg P/L; 0.5 mg NH4/L = 0.388 mg N/L);
    - state which window each event uses, and which pool defines the F-E P90.
11. **L249-266:** MEMLM (Sun, Holden & Birks 2024), cited in the M5 plan as a published method, is neither a comparator nor listed "not compared, with reason".
12. **L140-155:** by my reading of the department lists, the folds split four river systems: the Seine (G2/G3), Loire (G5/G4), Rhône (G7/G8) and Dordogne–Lot (G5/G6). R4 asks for flow-connected stations in one fold. Add a nearest-training-station distance report and a ≥ 50 km sensitivity stratum (§14.2). The wording "follow the water-agency basins" is also loose: G1 merges two agencies, and G3 and G5 cross agencies.
13. **L326:** the derivation of the 32-dimensional embedding from the 256-wide trunk is unspecified. State whether the `ens_f` member is fixed across tracks.
14. **L3-4:** the registry key `bioassess/prereg_snapshot` differs from M5-X's `bioassess_x/prereg_snapshot`. The stamp wording should follow the current lead procedure (sha256, mode 0444, registry row, commit deferred).
15. **Cross-workstream:** the draft is silent on the single shared Naïades pull, the role partition with M7 P3, and the Fuse prohibition (SYNTHESIS L349, X5). L226-227 says "public source" for toxicity values; SYNTHESIS C4 requires measured ECOTOX values only.
16. **Licence:** there is no mention of a `docs/licenses.md` entry (R11). Modes 0700/0400 are not access control when every agent runs as the same user; say the guard is procedural plus `ftrack`.
17. **L128-129:** the "≤ 2020-12-31" rule for new UK statistics conflicts with the on-disk `pm30d` summaries of December 2020 samples, which include January 2021 observations. This has no effect on the French test.
18. **L288-292:** the TabPFN sentence names only 2.5, while the README lists 2.6, 3 and 3.5.

## Recommendations on the 12 questions

1. **Holm family:** 36. The target-level label accepts either co-primary metric (L387-388), so both must be in the family.
2. **Folds:** department groups as primary, because they can be fixed with no data and the HER field is unverified. Add HER-1 and ≥ 50 km as sensitivity analyses.
3. **Lock boundary:** 2022-01-01. The prior-365-day window of a January 2023 operation reaches January 2022. Biology inputs can be readable, with the exceptions of must-fix 5 enumerated.
4. **Cap accounting:** accept. Eight listed configurations equal the cap of 8, and the draft is stricter than the plan.
5. **F-Z pool asymmetry:** accept. It favours the comparators; state it in the label and fix the `ens_zs` pools.
6. **Documentation fetches before the stamp:** yes, limited to a listed set of reference URLs, logged, with no endpoint that returns observation rows.
7. **French events:** a new module with `EventDef` objects passed through the v1.0 API. The `events.py` header forbids edits, and `get_event` accepts `EventDef`.
8. **GJAM:** accept "not compared, with reason". Word the reason as fold 0 of each block at 100 taxa, and cite the result files.
9. **TabPFN:** extend the wording to "2.5 and any later release", and state the hybrid backbone version.
10. **SPEAR and published score tables:** accept, licence-gated. Add measured-values-only and a recorded licence check per source.
11. **Diatom window:** accept ±90 days fixed now, with the tie rule and lock-key restriction from should-fix 3, and a |offset| ≤ 30 sensitivity analysis.
12. **File name:** keep `bioassess_f_protocol.md`. It parallels M5-X and `ftrack.PROTOCOL_FILE` points to it. The lead records it as M5's rule-1 amendment, with key `bioassess_f/prereg_snapshot`.

## Checked and found sound

- **Departments:** all 96 metropolitan departments are assigned exactly once across G1–G8 (sizes 11, 13, 9, 14, 9, 15, 11, 14).
- **Power table L397-399:** recomputed, all five rows match (650 → 0.030–0.048; 3,000 → 0.014–0.022).
- **Multipliers:** 4.04 for a family of 36 is correct (4.039). M5-X's 3.71 for 12 is correct. A family of 18 would give 3.83.
- **Molar factors L197-198:** 0.2259, 0.7765 and 0.3261 are correct.
- **Lock logic:** no chemistry row is shared between training labels (≤ 2021-12-31) and any test-row window (≥ 2022-01-01). Locking chemistry by date alone is sound.
- **Scope:** one look, one family, F-E and F-MA secondary, F-TOX exploratory, all matching the lead decision. The France block is reserved in `sota_campaign.md` L114-115.
- **Disk state:** no French data under `sentinel_data` or `sentinel_runs`, and 179 GB free. No other pre-registration file mentions French data.
- **UK rows:** the allowed-rows definition is consistent with the M5-X exclusions and does not weaken M5-X. Window definitions match `bio_uk_ea.py`.
- **GJAM:** `fidelity_pass` is FALSE in both blocks. In-sample r is 0.73–0.88 on X-L and 0.83–0.92 on X-U, both fold 0.
- **`ftrack.py`:** its constants match the draft (lock 2022-01-01, 650, 100, cap 8, family 36). It was edited during my review and is now 412 lines; I re-checked the constants only.

## Not verified

- **SANDRE parameter codes (L199-203):** they match my recollection, which is not verification. Decisions depend on them, because they select the target rows.
- **Fraction, unit, index and support codes:** not written in the draft. Unit conversion, comparators and row eligibility depend on them.
- **Export field names:** not decision-critical, except the sampling-date routing field and the station join key.
- **Good-status limits (L216-218):** they match my recollection, and affect the secondary track only. I recall the French total-P limit as 0.2 mg/L and the pH range as 6–9; the draft uses 0.1 and 6.5–9, apparently for M5-X comparability, and should say so.
- **Not checked at all:**
  - national chemistry export size;
  - licence strings at source;
  - HER availability;
  - pairing rate and diatom co-sampling rate;
  - TabPFN memory;
  - SPEAR licence;
  - whether UK diatoms (metabarcoding per RESEARCH_PLAN §5.5) are comparable to French microscopy counts.
- **River splits in should-fix 12:** these rest on my own geography, not on a network file.
