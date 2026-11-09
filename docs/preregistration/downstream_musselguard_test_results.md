# downstream_musselguard — test-look results and post-freeze write-ups

**Moved here 2026-11-09.** This text was appended to the frozen protocol `downstream_musselguard.md` after its FREEZE
record was written (2026-11-08): the single-test-look result (2026-11-08) and the post-results-review relabelling
and caveats (2026-11-08/09). Appending it broke the protocol's frozen sha256, so on 2026-11-09 the protocol was
restored byte-for-byte from the prereg archive (sha256 c2ad8032ad841881… verified against
`FREEZE_downstream_musselguard.json`) and set read-only; the appended text follows **verbatim** (no edits).

---


## 13. Single test look — result (2026-11-08 20:40–22:12; FREEZE_downstream_musselguard.json; registry `musselguard/test_look`)
Front-stratum AUPRC. **T = 2018** (n = 36,450, 102 positives): pressure_gbm 0.061 [0.030, 0.109] vs incumbent
(Kanankege re-impl) 0.017; Δ +0.044 [0.014, 0.085]; all 7 baselines Holm-rejected (Δ +0.037 … +0.058).
**T = 2021** (n = 38,390, 66 positives): pressure_gbm 0.018 [0.008, 0.044] vs incumbent 0.007; Δ +0.011
[0.001, 0.035] (p 0.020) — only the SC rule is Holm-rejected (m = 14); vs distance-only +0.005 [−0.032, 0.032],
vs pressure logistic +0.006 [−0.008, 0.026]. Seed rule holds. **Both-origins rule fails → label:
competitive** (strong at T = 2018, not at T = 2021, where labels for 2022–2024 may be incomplete — §12 W3).
Pooled origins (secondary): 0.034 vs incumbent 0.011. Not counted toward H-G4 "beats incumbent" (§12).
Caveat (post-results review): T = 2021 has only 66 front positives in 38,390 rows.
