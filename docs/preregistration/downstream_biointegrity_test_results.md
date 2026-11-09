# downstream_biointegrity — test-look results and post-freeze write-ups

**Moved here 2026-11-09.** This text was appended to the frozen protocol `downstream_biointegrity.md` after its FREEZE
record was written (2026-11-08): the single-test-look result (2026-11-08) and the post-results-review relabelling
and caveats (2026-11-08/09). Appending it broke the protocol's frozen sha256, so on 2026-11-09 the protocol was
restored byte-for-byte from the prereg archive (sha256 ef6e23b2fca503b1… verified against
`FREEZE_downstream_biointegrity.json`) and set read-only; the appended text follows **verbatim** (no edits).

---


## 11. Single test look — result (2026-11-08 20:25–20:32; FREEZE_downstream_biointegrity.json; registry `biointegrity/test_look`)
NRSA 2023-24 first visits at new locations (n = 979, 600 Poor). AUROC: ensemble 0.861 [0.836, 0.884]; Hill-2017
RF (published) 0.849, tuned 0.848; feature-matched RF 0.853, tuned 0.852; tuned GBM 0.860; location kNN 0.727.
Paired (HUC4): ensemble − Hill published **+0.012 [0.003, 0.021]** (p 0.005), − Hill tuned **+0.013 [0.005,
0.021]** (p 0.002) — Holm-rejected; − feature-matched RF +0.008 [−0.001, 0.017] (p 0.062), − tuned RF +0.009
[0.001, 0.018] (p 0.027) — not Holm-rejected. Seed rule holds (all per-seed Δ > 0). **Pre-registered claim
(all four RF rows) not met → label: competitive.** Against the §8 incumbent (Hill et al. 2017 RF, both
variants) the gain is Holm-significant. Secondary: all first visits ens 0.885 vs Hill 0.875; revisited
locations ens 0.919 vs Hill 0.918 vs persistence 0.880.
H-G4 (post-results review): the pre-registered 2026-10-24 rule governs → D5 does **not** count toward
"beats its incumbent".
