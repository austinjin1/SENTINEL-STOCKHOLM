# M3 SENTINEL-Microbe — external evaluation track "EA" (NEON → England river biofilms): pre-registration

**Status:** pre-registered 2026-11-10, approved by the lead the same day. Written **before**:
- any EA chemistry value was downloaded or joined;
- any EA sequence was processed beyond a one-run format probe (see §2.2);
- any model saw EA data.

**Role:** a separate, secondary track with its own freeze and its own single look (`M3E/test`). It never
changes the NEON primary endpoint (H-M3). The result is reported win or lose.

## 1. Why this track (RESEARCH_PLAN §15.5 item 5; SOTA campaign rule 2)
The practical use case is a model trained on one monitoring network and applied in another region. This track is
strict out-of-region transfer: train on NEON (USA, water-column filtrate) and test on England (EA River
Surveillance Network, benthic biofilm). It is also the first test of the V4 harmonisation on an independent
source. It was chosen from the data-availability check of 2026-11-10, before any result on it existed.

## 2. Data

### 2.1 Sources and licences
- **Sequences:** ENA PRJEB90117 (Warren et al. 2026, *Water Research* 300:125959; EA / UKCEH river-biofilm
  programme).
  - 1,649 paired NextSeq 1000 runs, first public 2026-10-29.
  - V4, 515F (GTGYCAGCMGCCGCGGTAA) / 806R.
  - Only R1 is used.
  - INSDC data: free and unrestricted; cite the paper and the accession.
- **Chemistry (labels):** `Example_data/filtered_metadata_normal.csv` in GitHub
  `Micro-Biology/Warren_et_al_-_microbial_indicators`.
  - Content: per sample (`ANON_ID` = ENA `sample_title`), the 3-month mean of "Nitrate as N" and of
    "Nitrogen, Total Oxidised as N", derived by the authors from the EA Water Quality Archive (open WIMS).
  - Licence: WIMS is under the Open Government Licence v3. The repository carries **no licence file**, so the
    derived table is used for research evaluation only and never redistributed (flagged in `docs/licenses.md`).
  - Only the header row has been read (column names). No value has been seen.
- **Not available:**
  - site coordinates: ENA lists one institutional coordinate for every run;
  - stable site ids: `ANON_ID` site codes are anonymised per year;
  - SC, TP, DO, DOC, turbidity, chl-a.

  Hence the WIMS station list cannot be linked. The track therefore has **one target**.

### 2.2 One-run format probe (disclosed)
Run ERR15105460 R1 only, first 20,000 reads, downloaded to an isolated folder and deleted afterwards. Findings:
- R1 = 251 bp (82%) or 35 bp (15%);
- the 515F 3′ core ends at position 19 in 82% of reads.

So the harmonised V4 window (230 bp after 515F) fits in R1. No chemistry was involved.

### 2.3 Read processing (fixed now; dereplication without DADA2, disclosed)
- **Window.** For each R1 read, locate the 515F 3′ core `GC[AC]GCCGCGGT[AG]A` within bases 0–40 and take the
  next 230 bases. This is the same window as `dna.v4_window` for V3V4/V4V5. Reads without the core, or shorter
  than core end + 230, are dropped.
- **Quality filter (fixed now).** Expected errors over the window, EE = Σ 10^(−Q/10), must be ≤ 1.0. Windows
  containing N are dropped.
- **Units.** Exact-window dereplication per sample. There is **no DADA2 denoising**: our NEON units are NEON's
  pipeline outputs, while EA must be processed by us. Instead, an **abundance filter is fixed now**: a window
  must have ≥ 2 reads in its sample, and singletons are discarded. Two unit views are stored: the 230-bp window
  and its first 150 bp. The 150-bp view is more robust to R1 tail errors and matches EMP and `kmer5_v4_150`.
- **Sample filter.** ≥ 1,000 retained reads, the same as NEON.
- **Storage.** Stream one run at a time; keep only `(sample_id, window_sha1, window, count)` for retained
  windows; delete each FASTQ immediately; check `df` per batch of 50 runs. Cap ≤ 5 GB stored; expected < 0.3 GB.

### 2.4 Labels and target
- **Target:** `log_no3` = log10(max(TON as N mg/L, 0.001)). NEON's `log_no3` is NO3+NO2-N, which is TON.
- **Sensitivity:** Nitrate-as-N.
- **Disclosure: the chemistry is not same-visit.** EA labels are 3-month means at the WIMS point used by the
  authors; NEON labels are same-visit grabs.
- **Disclosure: matrix shift.** EA biofilm (epilithon) vs NEON water-column filtrate.
- **Censoring.** As provided by the authors (Kaplan–Meier imputation to the lowest LOD).

### 2.5 Frozen sample ids
- The ids are the ENA runs whose `sample_title` appears in the authors' metadata file: 1,620 of 1,649, linked by
  id only.
- After read processing, the ids passing §2.3 are written with sha256 to `docs/preregistration/microbe_ea_ids.json`,
  **before** the label file is downloaded.
- The cluster unit is the anonymised letter group, the first character of `ANON_ID` (22 groups, A–X). It is
  presumed to be an EA area code; that is undocumented and is stated as an assumption.
- A row bootstrap is also reported.

## 3. Models and comparators (all trained on NEON only; no EA row ever enters fitting, tuning or calibration)
- **SENTINEL models scored.** At most 6, fixed in the EA freeze:
  - the NEON-frozen M\*;
  - up to 5 further pre-listed SENTINEL candidates, e.g. its 150-bp-window variant.

  Each is fitted on all NEON development events (plus .001 2017–18 rows if its track is `lodo_x001`) and
  predicts every EA sample once.
- **Comparators:**
  - RF[genus-CLR] and XGB[genus-CLR] (the stronger of tuned/untuned as frozen for NEON). EA genus labels come from
    an exact match of the EA window to the RDP genus of the NEON sequences with the same window (read-weighted
    majority); unmatched windows count as unassigned.
  - Tier-0: GlobalMean (NEON training mean) and Ridge[DOY] (DOY harmonics only, because EA has no coordinates).
- **Descriptors.** Disabled for every model: sequencing depth and platform differ.

## 4. Metrics and claims
- **Primary metric: Spearman ρ** between prediction and label (rank skill). A large level shift is expected from
  the matrix and the 3-month aggregation, and rank skill is invariant to it.
- **Secondary:** pooled R², and R² after per-group mean centring (anomaly).
- **Paired comparisons:** Δρ and ΔR², SENTINEL − comparator, with a cluster bootstrap over the 22 letter groups
  (B = 2,000). A row bootstrap is reported.
- **Holm** across (SENTINEL model × comparator) for Δρ.
- **Claim.** "Transfers to England biofilms better than <comparator>" requires Δρ > 0 with the group-cluster CI
  excluding 0 after Holm. A SOTA or first-in-class label on this track is scoped to the track.
- **Power (before scoring; no labels needed).** With n ≈ 1,600, the SE of a single ρ is about (1 − ρ²)/√(n − 3)
  ≈ 0.024 at ρ = 0.2 under independence. With 22 clusters the effective n is smaller. The realised n, cluster
  sizes and a design-effect bound go in `microbe_ea_ids.json` before scoring.

## 5. Guard and look
- **Look `M3E/test`.** Same hardened guard as the NEON look (`sentinel/models/microbe/testlook.py`):
  - its own lead-written freeze `FREEZE_microbe_ea.json`;
  - an O_EXCL claim tied to pid/host;
  - sealed predictions checked against the registry;
  - access closed at sealing;
  - rescore refused;
  - an import-closure code hash;
  - data pins for the processed EA units and the label file.
- **Label file timing.** The label file is downloaded only after `microbe_ea_ids.json` is frozen. Its content
  sha256 goes in the EA freeze. It is read only by the scorer under the open claim.

## 6. Cross-workstream rule
EA labels and derived EA tables are M3 evaluation data only. They are never passed to other workstreams. M5-X's
England track uses WIMS independently. This is stated in the datasheet.
