# downstream_toxinwatch — test-look results and post-freeze write-ups

**Moved here 2026-11-09.** This text was appended to the frozen protocol `downstream_toxinwatch.md` after its FREEZE
record was written (2026-11-08): the single-test-look result (2026-11-08) and the post-results-review relabelling
and caveats (2026-11-08/09). Appending it broke the protocol's frozen sha256, so on 2026-11-09 the protocol was
restored byte-for-byte from the prereg archive (sha256 cc15ccb073d098a0… verified against
`FREEZE_downstream_toxinwatch.json`) and set read-only; the appended text follows **verbatim** (no edits).

---

- **Single test look — result (2026-11-08 20:22–20:25; FREEZE_downstream_toxinwatch.json; registry
  `toxinwatch/mc_exceed8@it3/test_look`).** Known lakes 2022–2026 (n = 3,190, 231 exceedances): GBM 0.933
  [0.739, 0.974], strongest incumbent (v2 pooled) 0.928, location kNN 0.893; test lakes (n = 568, 53): GBM 0.858
  [0.640, 0.943], v2 pooled 0.835, lake history 0.831. Paired (lake clusters): GBM − v2_pooled **+0.004
  [−0.004, 0.031]** (p 0.211) known, **+0.022 [−0.000, 0.059]** (p 0.053) new — incumbent family not rejected;
  GBM − location_knn +0.040 [0.013, 0.182] (known, rejected); − lake history +0.026 [−0.019, 0.115] (new, not).
  Seed rule holds. **§5 rule not met → label: competitive** (the "not robust" label does not arise: the
  governing rule already fails). Lake Washington dominates the known-lake bootstrap (wide CIs). Caveat (post-results review): Lake Washington ≈ 60 % of known-lake rows,
    one lake ≈ 39 % of new-lake rows; the incumbent is an INLA-style re-implementation (regularised GLMM-style
    logistic with splines), not INLA. **Future D2 track (pre-declared):** Brooks-style RF and GBM comparators,
    2026-season test block.
