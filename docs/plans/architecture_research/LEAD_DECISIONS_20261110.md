# Architecture round v2: lead decisions (2026-11-10, 12:10 EST)

**Trigger.** User directive 2026-11-10: "deep research into these architectures ... build novel architectures and also
if simple ML is doing the best the task needs to be harder so more data more predictive targets".
**Inputs.** `M{1,2,3,4,5,7}_architecture_plan.md` in this directory (M6 and `SYNTHESIS.md` still being written).
**Written before** any result exists on any arm or track approved below. Nothing here lowers a bar: RESEARCH_PLAN
§15.1 and `sota_campaign.md` rules 1–8 are unchanged.

## Rules for this round (all workstreams)

1. **Amendment first.** Each workstream drafts ONE new file `docs/preregistration/<ws>_arch_v2_amendment.md`
   (never an edit to a frozen file). It holds, for every approved arm and track: the practical reason, the unseen
   test block, power from counts/metadata only, comparators with parity, configuration caps, the selection rule,
   and every gate or kill rule. The lead reviews and stamps it (sha256 + `<ws>/prereg_snapshot` registry row +
   git commit).
2. **Before the stamp** a workstream may write code and tests, build data (labels of protected rows: counts only),
   and run smoke tests on synthetic or train-only data. **No validation number on a new arm or track before the
   stamp.**
3. **References first.** The strongest published method (re-run here) and the strongest simple baseline, with the
   same inputs, labels, auxiliary targets and ensembling as the candidate, are built before or with the candidate.
4. **Nothing already queued is delayed.** Existing freezes and looks proceed; v1 results stay reported.
5. **Compute.** The plans assumed 2 shared GPUs. GPUs 0–9 are now allowed (lock helper, `--min-free-mb`, ≤ 2 h per
   job, never a GPU with another user's process). CPU ≤ 6 threads per job; load is ~40 of 48 cores, so prefer GPU
   work and keep CPU jobs sequential per workstream. Disk floor 60 GB free on `/`.
6. **Every arm and track is reported**, wins and losses, scoped to its track.

## M1 Sonde

- **Approved to build:** references R-NH (Nearing-style hindcast–forecast LSTM), GBM-NWP, PF-GBM, Chronos-2-cov
  zero-shot; candidates SONDE-HF (rank 1) and E2 with the matched baseline-ensemble control E2-B.
- **Gated:** PhiDA-Lite (gate C3) and Chronos-2-NWP, both by the step-1 information-use test (train-block inner CV
  only, decision rule as in plan §8).
- **Tracks:** H-1 (M1-N forecast mode with GEFS) approved subject to the HEAD-request size check; transfer only as
  stream-extract-delete, extractions < 1 GB, disk floor respected. H-2 (M1-H hourly DO / hypoxia hours) approved;
  the protocol must state that targets are hourly IV values themselves, so the failed IV→daily rule does not apply.
  H-3 forcing strata approved as secondary. H-5 StreamPULSE approved as secondary, lock-on-download, protocol
  before download. H-6 (EW-H1h) is **descriptive by default**; confirmatory only if the metadata-only power check
  written into the M1-H protocol passes, and always disclosed as defined after the IV-rule failure. H-4 waits for
  M6. H-7/H-8 unchanged.
- **Time-critical:** the amendment must be stamped before any NWP-based M1-N validation number exists.

## M2 Orbit

- **Approved:** Appendix A.1 (Test-S), A.2 (arms A0 and A1; caps A0 1, A1 3), A.3 (Track N: TP, TN, DOC), and the
  dev_v5m multi-target build. **Seed unit** = 3 inits (mixture average) + EMA is approved **only with parity**:
  every learned comparator gets the same 3-init averaging, and the GBM gets 3-seed bagging.
- **Gated / not funded yet:** A2 (gate as written), A3, A4, Track G amendments (draft them; no download yet).
- **The headline M2/test look with v1 is not delayed.**
- **Test-S is reserved from now** (see `sota_campaign.md`).

## M3 Microbe

- **Approved:** §8.1 items 1–3 and 5–6: GEO protocol (GEO-8) in the same sealed look as H-M3; reserve slots
  #6 NicheInvert (D1), #7 V4-MR (D3a), #8 NicheGeo-Phys (D2) only if the D1 gate passes, else stacking (D3b); the
  DNA-LM slot is dropped; the M3E multi-determinand amendment; GAS-3 amendment before its download; qPCR input
  amendment.
- **Ruling 4a (H4, EA-internal area-blocked):** approved as a separate conditional track. The EA-dev / EA-test
  partition is written to `microbe_ea_ids.json` before any EA label is downloaded. EA-dev rows are never used to
  train the NEON→England (H3) models. Any EA chemistry row that is a frozen M5-X test key is excluded from H4
  training.
- **Ruling 4b:** M6 is told about M3's EA use by the lead (no overlap: M6 reserves Hydrology-API sonde data).

## M4 ChemTox

- **On hold.** The M4 workstream session was stopped by the platform's safety system on 2026-11-10 and the user
  has not yet chosen how to resume. The plan is on file; nothing in it is dispatched. Jobs already queued run to
  completion unsupervised.

## M5 BioAssess

- **M5-X is left alone** (freeze candidate → dry run → pre-test review, as planned).
- **Approved:** the France block (M5-F; tracks F-A, F-Z, F-T in one look and one Holm family; F-E, F-MA secondary;
  F-TOX exploratory), BioSet-MA plus the TabPFN hybrid, cap ≤ 8 configurations. New-data cap for M5-F: 5 GB.
  UK diatoms and prior-365-day windows (on disk) may be used for development on allowed pool rows only.
- **Not funded:** full GeoNiche. TabPFN-2.5 weights need a human licence decision and are not used; the installed
  TabPFN v2 stays the comparator.
- **M5-F reservation** recorded in `sota_campaign.md`.

## M7 Lite

- **Approved:** label recovery (FNU, NTRU, SSC, censored rows) as a new data version; ANCHOR-Lite (rank 1) and
  WaterSet-Lite as its alternative residual branch; re-run published comparators (HydroVision-style DenseNet121
  regressor, CyFi-style LightGBM); track **P2** (rivers/sondes FNU turbidity at unseen water bodies).
- **Ruling on labels:** the frozen `lite_protocol.md` §8a rule stands for the national track. **No target-scoped
  label is added there after validation has been seen.** A scoped SOTA claim is possible only on P2, pre-registered
  before any result.
- **Gated / deferred:** CIR-Lite-R (calibration gate), SparK SSL, P3 France and P4 NLA.

## M6 HydroMet

- Pending its plan. Track R and the lever build continue.
