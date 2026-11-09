# downstream_d1n_beaches — test-look results and post-freeze write-ups

**Moved here 2026-11-09.** This text was appended to the frozen protocol `downstream_d1n_beaches.md` after its FREEZE
record was written (2026-11-08): the single-test-look result (2026-11-08) and the post-results-review relabelling
and caveats (2026-11-08/09). Appending it broke the protocol's frozen sha256, so on 2026-11-09 the protocol was
restored byte-for-byte from the prereg archive (sha256 73cd840de98cd641… verified against
`FREEZE_downstream_d1n_beaches.json`) and set read-only; the appended text follows **verbatim** (no edits).

---


## 11. Single test look — result (2026-11-08 20:33–20:40; FREEZE_downstream_d1n_beaches.json; registry `d1n_beaches/test_look`)
Frozen logistic (C = 1), refit on train + val rows. AUROC: `test_known` 2021–2025 (n = 30,983, 4,037
exceedances) logistic 0.815 [0.787, 0.838], VB-MLR incumbent 0.803 [0.770, 0.829], history rate 0.758, GBM 0.820;
`test_new` (never-fitted HUC4s; n = 10,249, 847) logistic 0.786 [0.749, 0.818], VB-MLR 0.770 [0.735, 0.804],
history rate 0.709, GBM 0.792. Paired (HUC8): logistic − VB-MLR **+0.0125 [0.0044, 0.0215]** (p 0.004) known,
**+0.0157 [0.0023, 0.0262]** (p 0.016) new; logistic − history rate +0.057 [0.043, 0.079], +0.077
[0.048, 0.105]. Holm over all four rejects all; deterministic challenger (seed rule trivially met).
§5 claim rule met. **Label (amended by the post-results review, 2026-11-08): beats its re-implemented
Virtual-Beach incumbent and the strongest simple baseline on the D1n track; NOT SOTA** (§15.1(i): the strongest
published method family — e.g. RF per Brooks et al. 2016 — was not compared; the incumbent is a simplified
re-implementation). Counts for H-G4 as D1's national track with this caveat. 20 configurations; logistic chosen
over the GBM on validation. A reviewer's post-hoc, unregistered GBM-vs-logistic check is descriptive only.
**Future D1n track (pre-declared before any fit):** comparators must include a Brooks-style RF (tuning parity)
and the GBM; test block = the unseen 2026 season.
Caveats: "new" = new HUC4 (63/160 test-new clusters share a Great Lake with fitted HUC4s); margins are small
(≈ +0.013–0.016 AUROC).
