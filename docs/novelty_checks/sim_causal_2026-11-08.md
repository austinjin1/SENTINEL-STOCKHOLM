# Novelty re-check: causal discovery (PCMCI/tigramite) on multi-site river WQ sensor networks, 2026-11-08

**Claim checked (sim_protocol.md §7, S-C ladder).** PCMCI-family, CCM and TE methods applied to continuous river
WQ sensor series, validated with:
- physics positive controls (T → DOsat, upstream → downstream);
- negative controls (time reversal, disconnected pairs, placebo);
- simulation recovery on Sim output;
- a cross-correlation baseline.

This extends sim_causal_2026-10-25.md.

**Method caveat.** Searches used the agent's general web index (WebSearch/WebFetch), not Scholar, Scopus or WoS.
Forward citations of CausalRivers (Stein et al. 2025) and tigramite papers were *approximated* by topical queries.
Chinese-language journals are likely under-indexed. ACS and ScienceDirect pages returned 403.

## Queries run (all 2026-11-08)
1. PCMCI tigramite river water quality sensors dissolved oxygen turbidity nitrate causal discovery 2025 2026
   (extended)
2. causal discovery high-frequency stream water quality monitoring network negative controls time reversal
   synthetic benchmark validation hydrology arXiv 2026 (extended)
3. convergent cross mapping OR transfer entropy OR Granger dissolved oxygen river sensor network causal drivers
   specific conductance 2025 2026 Water Research OR STOTEN (extended)
4. PCMCI+ river dissolved oxygen drivers automatic monitoring stations causal network temperature discharge 2025
   2026 journal (extended)

## New relevant items
**None new** that apply PCMCI-family, CCM or TE to high-frequency *river WQ* sensor networks with
positive/negative controls.

New method-adjacent items:
- **Ruiz, Arana-Catania, Ardila & Ventura 2026, arXiv:2608.00198** (submitted 2026-07-31), "AutoCause".
  - An environmental time-series causal-discovery workflow: four methods plus "non-causal reference models".
  - Tested on 145 datasets (DGP-Atlas, TimeGraph, CausalRivers topology).
  - Majority-supported links were more precise on synthetic data **but not against river topology**.
  - No WQ sensor data.
  - **Reinforces the requirement to report a non-causal/cross-correlation baseline.**
- **Fesanghary & Havaldar 2026, arXiv:2606.23880** (submitted 2026-06-22), "GRACE". A refinement of
  constraint-based discovery, tested against PCMCI. On an Elbe "river flow" dataset it recovers 9 of 11 edges.
  Flow, not WQ.
- **"Evaluating the Robustness of PCMCI+ for Causal Discovery of Flood Drivers", *AI for the Earth Systems*
  4(4), 2025, doi:10.1175/AIES-D-24-0114.1** (authors not recorded [U]). Discharge in 45 catchments; many edges
  stay inconsistent even with large samples. A relevant caveat.
- [U] Riverine chlorophyll-a forecasting with CCM + ML, *J. Hydrol.* 2024 (S0022169424014689; 403, authors and
  data frequency not seen). A river WQ CCM application, without the controls (as far as can be seen).
- Already known: CEcBaN (*Water* 18(6):764, 2026; lake); the Long Island Sound PCMCI+ estuarine DO study
  (S1001074225007399, monthly); and Langendorf et al. 2021 Chesapeake (doi:10.1021/acsestwater.0c00307). A search
  summary says that paper compared PCMCI with CCM [U; page 403].

## Effect on novelty wording
**Unchanged.** The classification stays "partially novel (methodological extension)". The permitted wording from
sim_causal_2026-10-25.md stands: "To our knowledge, causal discovery has not been applied to continuous river WQ
sensor networks with physics-based positive controls, time-reversal negative controls and simulation recovery."

Two adjustments to the surrounding text:
1. Soften "almost all applications are estuarine or lake" to note at least one riverine WQ CCM application
   (chl-a, J. Hydrol. 2024 [U]).
2. Cite AutoCause 2026 and PCMCI+ flood-drivers 2025 next to CausalRivers as evidence that consensus or causal
   methods do not reliably beat naive baselines on river topology. This strengthens the case for the
   cross-correlation baseline already in §7.
