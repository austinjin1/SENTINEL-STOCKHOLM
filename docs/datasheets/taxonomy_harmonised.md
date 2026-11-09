# Datasheet — `taxonomy_harmonised` (NARS + Environment Agency source taxa → one hierarchy)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/taxonomy_harmonised/` · manifest `manifests/taxonomy_harmonised.json` · GBIF cache `raw/taxonomy_gbif/` (manifest `taxonomy_gbif_raw`) |
| **Inputs** | `processed/nars_streams` + `processed/nars_lakes` (`taxa_source`, `*_counts`), `processed/uk_ea_ecology` (`taxon_dictionary`, `counts_*`), GBIF Backbone Taxonomy via `https://api.gbif.org/v1/species/match` (8,946 name queries + 309 rank-value checks, 2026-10-22; no key, no personal data) |
| **License** | derived from U.S. Government data (NARS), OGL v3.0 (EA) and the GBIF Backbone Taxonomy (CC BY 4.0, GBIF Secretariat — cite GBIF) |
| **Builder** | `sentinel/data/bio_taxonomy.py` · `scripts/data/download_taxonomy_gbif.py` · `scripts/data/build_taxonomy.py` |
| **Synthetic data** | none |

## What it is

One row per **source taxon** (dataset × source taxon id × name; 22,720 rows) mapping the name used by a survey to a harmonised lineage and to a fixed **analysis resolution**, so that models can work across surveys and continents with mixed taxonomic resolution.

| Source | Group | Source taxa |
|---|---|---|
| `nars_streams` | macroinvertebrate (NRSA benthic) / fish | 4,963 / 3,000 |
| `nars_lakes` | macroinvertebrate (NLA benthic) / zooplankton / algae (phytoplankton) | 2,928 / 1,957 / 5,134 |
| `uk_ea` | macroinvertebrate / algae (diatoms + algae in invertebrate samples) / macrophyte / fish / other vertebrate | 2,301 / 1,814 / 504 / 79 / 40 |

Distinct cleaned names: macroinvertebrates 3,166, algae 3,631, fish 990, zooplankton 663, macrophytes 398, other vertebrates 40. EA records are grouped by the EA taxon type, because EA invertebrate samples also list macrophytes, algae, fish and other vertebrates seen in the net — **filter `group == "macroinvertebrate"` for invertebrate community work**.

## Files

| File | Content |
|---|---|
| `taxon_map.parquet` | `source`, `dataset`, `assemblage`, `group`, `source_taxon_id` (NARS TAXA_ID / EA TAXON_LIST_ITEM_KEY), `source_taxon_name`, `n_records`, `total_count`, `name_clean`, `identification_rank`, harmonised lineage `h_kingdom` … `h_genus`, `h_species`, `h_subfamily`, `h_tribe`, `lineage_source`, GBIF match (`gbif_usage_key`, `gbif_match_type`, `gbif_rank`, `gbif_status`, `gbif_confidence`, `gbif_kingdom` … `gbif_species` — the backbone's accepted lineage, kept separately), `gbif_rejected_homonym_or_synonym`, conflict flags, `analysis_rank`, `analysis_taxon`, `analysis_taxon_rank`, `analysis_coarser`, `mixed_otu` |
| `conflicts.parquet` | (group, name, rank) with all distinct values and the datasets using each, `conflict_type` (`between_sources`, `source_vs_gbif`, `spelling_corrected`), `pattern`, `pattern_n_names`, `systematic`, `chosen`, `lineage_source` |
| `otu_rules.parquet` | macroinvertebrate families per data source with the share of records identified to genus or finer and the `genus_consistent` decision |
| `name_backbone.parquet` | every (group, cleaned name) GBIF match |
| `rank_value_checks.parquet` | GBIF verification of 309 source phylum/class/order/family values that no name match used (164 EXACT, 27 FUZZY, 111 NONE, 7 HIGHERRANK) and whether a reviewed fix applies |
| `summary.json` | counts below |

## Rules (fixed in code; see the module docstring)

1. **Name cleaning:** upper-case; drop vernacular parentheses ("Sphaeriidae (Pea mussels)"), trailing `SP.`, `SPP.`, `GR.`, `AGG.`, `S.L.`, `CF.`, "UNDETERMINED"/"UNIDENTIFIED"/"IMMATURE"/"LARVAE"; slash complexes (`CRICOTOPUS/ORTHOCLADIUS`) kept as one taxon of rank `complex`. EA names use the EA preferred (current) name; NARS fish common names (`FINAL_NAME`) are mapped to genus + species through the 2013-14 and 2023-24 fish taxa lists.
2. **Identification rank:** the finest published lineage level equal to the name (species/genus/tribe/subfamily/family/…); EA rank labels when the lineage cannot tell.
3. **Lineage:** family by vote over distinct datasets plus one vote for the GBIF match (ties → GBIF, then most recent NARS list). Ranks above family from a **family → (kingdom, phylum, class, order) lookup** built only from names whose GBIF family equals the voted family — every member of a family therefore gets the same higher lineage, and GBIF homonym matches are rejected (204 names, e.g. the beetle genus *Sphaeridium* matched to a bivalve family stays in Hydrophilidae/Coleoptera). Lineage sources: `family_lookup` 21,166 rows, `source_consensus` 1,092, `gbif` 347, `gbif_complex_shared` 30, `gbif_rank_mismatch` 8, `none` 77. EA parent chains contribute only family and below (the EA dictionary labels some orders as "Class", e.g. Diptera).
4. **Spelling:** only the reviewed list `REVIEWED_SPELLING_FIXES` (31 corrections, 122 rows; e.g. `ANNILIDA`→`ANNELIDA`, `MAXILLIPODA`→`MAXILLOPODA`, `DECOPODA`→`DECAPODA`, `CHLORELLACAE`→`CHLORELLACEAE`, older `-OIDA` order spellings → `-IDA`). Automatic fuzzy correction (difflib ≥ 0.85, with and without GBIF verification) was tested and **rejected**: it merged distinct real taxa (Phoridae→Phoronidae, Leptosomatidae→Lepidostomatidae, Mesogastropoda→Neogastropoda, Chrysococcaceae→Chroococcaceae, superfamilies into one family).
5. **Analysis resolution (fixed):** macroinvertebrates **family**; fish **species**; zooplankton **genus**; algae/diatoms **genus**. Identifications coarser than the analysis rank keep their own name (`analysis_coarser = True`, e.g. OLIGOCHAETA) — mixed resolution, nothing dropped. Share of records at or finer than the analysis rank: NRSA benthic 0.970, NLA benthic 0.964, EA invertebrates 0.961, NRSA fish 0.995, NLA zooplankton 0.764, NLA phytoplankton 0.952, EA diatoms/algae 0.975.
6. **`mixed_otu`** (macroinvertebrates): genus where the family is consistently identified to genus in that data source (≥ 90% of its records at genus or finer, ≥ 20 records), else family. Genus-consistent families: NRSA 70 of 278, NLA 47 of 205, EA 17 of 370 (EA routine analyses are family-level). Result: 606 family-level analysis taxa and 1,036 mixed OTUs for macroinvertebrates.

## Conflicts (documented, not hidden)

2,314 names have at least one disagreement between sources or with GBIF. Most are **systematic classification-system differences** (a value pattern shared by ≥ 5 names): `phylum CYANOBACTERIA|CYANOPHYTA` (240 names), `class CYANOBACTERIIA|CYANOPHYCEAE` (190), `phylum CHRYSOPHYTA|HETEROKONTOPHYTA|OCHROPHYTA` (143), `order CHLOROCOCCALES|SPHAEROPLEALES` (124), `class COPEPODA|MAXILLOPODA` (105), `phylum CHAROPHYTA|CHLOROPHYTA` (98), `class CLITELLATA|OLIGOCHAETA` (63) … 985 names carry an *isolated* conflict (likely source errors or genuine reclassifications, e.g. *Amphora* placed in Thoracosphaeraceae (a dinoflagellate family) in one NLA 2022 row; *Gyrosigma* in Naviculaceae in NLA 2022 vs Pleurosigmataceae in five other sources; water mite *Protzia* in Protziidae (NRSA) vs Hydryphantidae (NLA, GBIF)). 1,157 names disagree **between sources** (not only with GBIF). Rows by type: between_sources 1,717 systematic / 463 isolated; source_vs_gbif 1,532 / 735; spelling_corrected 122.

## Known issues

- GBIF backbone gives no class for ray-finned fishes (Actinopterygii is not a ranked class there), so `h_class` is null for most fish.
- 77 source taxa have no lineage (unmatched names, e.g. vague NARS fish names or NLA "UNKNOWN" algae).
- Synonymy is **not** resolved into the harmonised name: `name_clean` keeps the source identification; GBIF's accepted names are in `gbif_genus`/`gbif_species` (e.g. EA *Baetis digitatus* → GBIF *Nigrobaetis*). Use those columns if backbone-accepted names are wanted.
- NLA zooplankton/phytoplankton taxonomy changed strongly between 2007 and 2012; harmonise at genus and keep a cycle indicator.

## Rebuild

```bash
cd /home/bcheng/STOCKHOLM/SENTINEL-STOCKHOLM
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/download_taxonomy_gbif.py --workers 4   # resume-safe GBIF cache
/home/bcheng/miniconda3/envs/sentinel2/bin/python -B scripts/data/build_taxonomy.py
```
(requires the NARS and EA processed datasets; ~5 min)
