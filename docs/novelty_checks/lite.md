# Novelty check — M7 SENTINEL-Lite (RGB+NIR drone student)

**Status:** provisional, web-search pass only (2026-10-24). A Scopus / Google Scholar pass is still required
before any "first-in-class" wording leaves the repo (RESEARCH_PLAN §10, §15.1). A novelty finding does **not** by
itself earn the label: the model must also beat every applicable baseline with CI excluding 0 (H-G6).

## Claim under test (RESEARCH_PLAN §10, research 08 §e.11)
"The first openly released RGB+NIR water-quality student distilled from a red-edge-capable Sentinel-2 teacher using
only public data, evaluated on held-out water bodies against in-situ measurements, reporting the red-edge gap, the
distillation gain and INT8 edge latency." Optional dataset claim: "first public NAIP–in-situ water-quality matchup set".

## Explicitly NOT claimed (prior art exists)
First drone/UAV water-quality model (e.g. Su & Chou 2015; Cheng et al. 2020; many UAV multispectral studies 2020–2026,
e.g. ISPRS Archives 2025 Ibirité reservoir chl-a/turbidity; MDPI Remote Sensing 2026 Thrace river); first RGB
water-quality app (HydroColor, HydroVision); first Raspberry Pi RGB+NoIR drone (Belcore et al. 2019); cross-modal
distillation in remote sensing generally (e.g. S2→S1 flood mapping, Environmental Data Science; multi-scale
underwater→drone distillation for coral reefs, Ecological Informatics 2025; dual-teacher distillation for
multispectral EO, arXiv 2602.19863).

## Searches run (2026-10-24, general web search engine; US results)
| # | Query | Relevant hits | Conflicts with claim? |
|---|---|---|---|
| 1 | knowledge distillation Sentinel-2 teacher drone RGB NIR student water quality Secchi turbidity | cross-modal flood distillation (S2→S1); coral-reef underwater→UAV distillation; weed-mapping KD with drones; dual-teacher EO distillation | No water-quality S2→drone distillation found |
| 2 | NAIP imagery in situ water quality Secchi depth deep learning lakes matchup | Landsat Secchi deep learning (global lakes), Adriatic 1D-CNN | No NAIP–in-situ WQ matchup study found |
| 3 | UAV multispectral water quality model edge deployment INT8 quantized lightweight CNN turbidity chlorophyll 2025 | UAV multispectral ML for chl-a/turbidity (ensembles, SVR, hyperspectral attention CNN) | None report on-board INT8 inference |
| 4 | "NAIP" lake water clarity OR turbidity OR chlorophyll aerial imagery 1 m estimation | reviews, NAIP program pages | No quantitative NAIP lake WQ retrieval found |

## Provisional conclusion
No conflicting prior work found by web search for (a) S2-teacher → RGB+NIR-student distillation for water quality or
(b) a public NAIP × in-situ water-quality matchup set. Web search is not a systematic database search; recall is
limited. **To do before claiming:** Scopus/Scholar queries ("NAIP" AND (Secchi OR turbidity OR chlorophyll));
("knowledge distillation" AND "water quality" AND (UAV OR drone)); ("cross-sensor" AND "Sentinel-2" AND UAV AND
"water quality"); check citing papers of Kloiber 2002 and Olmanson et al. for aerial-photo clarity work (MN lake
clarity from aerial imagery has a long state-agency history, which may pre-empt the dataset claim).

## Sources
- https://www.cambridge.org/core/journals/environmental-data-science/article/crossmodal-distillation-for-flood-extent-mapping/ED8B97A83F4A12983C08A81C95ED4838
- https://www.sciencedirect.com/science/article/pii/S157495412500158X
- https://arxiv.org/html/2602.19863v3
- https://www.researchgate.net/publication/374702857_Applying_Knowledge_Distillation_to_Improve_Weed_Mapping_With_Drones
- https://www.sciencedirect.com/science/article/abs/pii/S0043135422002044
- https://www.tandfonline.com/doi/full/10.1080/20964471.2023.2273058
- https://isprs-archives.copernicus.org/articles/XLVIII-2-W11-2025/263/2025/
- https://www.mdpi.com/2072-4292/18/2/347
- https://www.sciencedirect.com/science/article/pii/S2666017225000884
- https://www.preprints.org/manuscript/202309.0489
