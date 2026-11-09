# downstream_safeswim — test-look results and post-freeze write-ups

**Moved here 2026-11-09.** This text was appended to the frozen protocol `downstream_safeswim.md` after its FREEZE
record was written (2026-11-08): the single-test-look result (2026-11-08) and the post-results-review relabelling
and caveats (2026-11-08/09). Appending it broke the protocol's frozen sha256, so on 2026-11-09 the protocol was
restored byte-for-byte from the prereg archive (sha256 4722ce2437196e0f… verified against
`FREEZE_downstream_safeswim.json`) and set read-only; the appended text follows **verbatim** (no edits).

---


## 12. Single test look — result (2026-11-08 20:32–20:33; FREEZE_downstream_safeswim.json; registry `safeswim/d1b_nowcast/test_look`)
D1b qPCR *Enterococci* > 1,000 CCE nowcast, 2024–2025 official rows (n = 2,300, 101 exceedances; 2026 never
loaded). AUROC: frozen logistic 0.729 [0.679, 0.785]; official City model 0.686 [0.626, 0.744]; feature-beach
GM 0.730 [0.689, 0.772]; GBM 0.713; persistence 0.680. Paired: logistic − official **+0.043** [−0.009, 0.102]
(week, p 0.101) / [0.003, 0.105] (beach, p 0.038); logistic − GM **−0.000** [−0.036, 0.037] (week) /
[−0.030, 0.025] (beach). Holm (m = 4) rejects nothing under either cluster unit → **§4a not met.
Label: competitive** (CI overlaps the incumbent under ISO-week clustering; no gain over the same-day
feature-beach GM). The validation gain over GM (+0.043) did not replicate (selection over 15 feature sets).
Caveat (post-results review): the beach unit has only 12 clusters (week unit 32), so beach-cluster percentile
CIs are likely too narrow.
