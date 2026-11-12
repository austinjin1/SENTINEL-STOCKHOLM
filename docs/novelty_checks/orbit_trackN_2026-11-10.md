# Literature check — Track N (lake TP, TN, DOC from Sentinel-2 / Landsat), 2026-11-10

Written by the M2 engineer for lead ruling R-12 (`docs/preregistration/orbit_arch_v2_lead_rulings.md`, CMP-3),
on 2026-11-10 at 20:26 EST, before any Track N validation number of any arm or comparator exists and before
dev_v5m is built. The lead reads this file before the Test-S freeze. It supplements
`docs/novelty_checks/orbit_2026-10-24.md` (base targets).

**Question (R-12).** Is there a published, reproducible Sentinel-2 or Landsat retrieval method for lake total
phosphorus, total nitrogen or dissolved organic carbon that the comparator list of amendment §2.5 (retrained
MDN_B, GBM families incl. the per-context-date member, location kNN, season + location GBM, trophic OLS chain,
the MDN_B + GBM average of R-10) does not cover?

## How the check was done, and its limits

- Six web searches and one page read, all on 2026-11-10 (about 20:20–20:26 EST). No Scholar, Scopus or Web of Science
  query. Apart from item 1 below (publisher page read, first 100,000 characters), **only search-result excerpts
  were read**: figures quoted from excerpts are unverified and are not to be cited from this file.
- No code of any paper was downloaded or run. No number of ours is compared with any number below; the studies
  use other data, other splits and other metrics.
- Queries: (i) "Sentinel-2 OR Landsat retrieval of lake total phosphorus total nitrogen machine learning
  non-optically active water quality parameters review 2024 2025 2026"; (ii) "Landsat Sentinel-2 dissolved
  organic carbon DOC lakes retrieval CDOM algorithm continental scale AquaSat LIMNADES unseen lakes validation";
  (iii) "national-scale OR continental-scale lakes total phosphorus estimation Landsat OR Sentinel-2 machine
  learning thousands of lakes held-out lakes 'total phosphorus' United States LAGOS"; (iv) "AquaSat Landsat
  matchups dissolved organic carbon lakes rivers machine learning model 'DOC' prediction leave-site-out United
  States"; (v) "Harkort Duan 2022 'dissolved organic carbon' inland waters machine learning Landsat ERA5-Land
  AquaSat Water Research validation results RMSE"; (vi) "Sentinel-2 total nitrogen total phosphorus lakes deep
  learning transfer multiple lakes 'leave-one-lake-out' OR 'unseen lakes' OR 'spatial cross-validation' 2025 2026".

## What was found

| # | Study | Target(s) | Sensor, inputs | Scale and split (as far as read) | Method class | Covered by §2.5? |
|---|---|---|---|---|---|---|
| 1 | Hanly, Webster, Soranno 2025, *Sci. Data* 12:1315, "LAGOS-US LANDSAT" (doi 10.1038/s41597-025-05600-w). Page read. | DOC (also chl-a, Secchi, colour, TSS, turbidity). **No TP, no TN.** | Landsat; six bands and band ratios; no lake, catchment or climate covariate | 740,627 matchups, 13,756 US lakes; 75/25 spatially balanced split of matchups (not stated as whole-lake hold-out); no held-out years | Random forest per variable | In kind: a retrained tree ensemble on band and band-ratio features is the single-target GBM family. The paper's own RF was not re-run. Code on Zenodo (10.5281/zenodo.11110283) |
| 2 | Harkort & Duan, *Water Research* 229 (2022/2023), doi 10.1016/j.watres.2022.119478. Excerpt only. | DOC, inland waters | Landsat 5/7/8 bands **plus ERA5-Land environmental variables** (monthly air temperature named as a top predictor) | > 16,000 AquaSat samples, continental US; split not seen | RF, SVR, Gaussian process regression (best), MLP | **Not covered**: no §2.5 comparator has reanalysis covariates. The season + location GBM is the nearest (it can learn climate from latitude, longitude and day of year) |
| 3 | Phan & Lima (IBM; INFORMS 2023 abstract). Excerpt only. | DOC | Landsat (AquaSat) with spatio-temporal inputs | US; split not seen | ML; abstract only | Covered in kind by GBM imagery + season + location members. Abstract only: not reproducible |
| 4 | Systematic review, *Remote Sensing* 18:1098 (2026). Excerpt only. | TP, TN (and optically active) | S2 / Landsat | Review; reports nutrient retrieval as less consistent and more site-dependent than turbidity or chl-a | — | — |
| 5 | NE China lakes (Xingkai, Chagan, Songhua), *Remote Sensing* 17:267 (2025). Excerpt only. | TP, TN | Sentinel-2 bands | 3 lakes; within-region | RF, XGBoost | Covered: GBM family |
| 6 | Toming et al. 2024, *Remote Sensing* 16:464, Estonian lakes. Excerpt only. | 16 parameters incl. TP, TN | Sentinel-2; genetic-algorithm band selection + XGBoost | Several lakes, five dates; not a stated lake hold-out | XGBoost | Covered: GBM family |
| 7 | Single-lake ML studies (Lake Tana Landsat 8, 2024; Dianshan Lake S2 / CatBoost, 2023; East Juyan Lake S2 / XGBoost; Chaohu 2022; Lake Simcoe EGU 2022 abstract, multimodal DL). Excerpts only. | TP, TN | S2 or Landsat 8 bands | One lake each, tens to hundreds of samples, random splits | Tree ensembles, small NNs | Covered: GBM family; MDN_B stands for the NN class. Lake-specific models cannot be transferred to unseen lakes without refitting, which is what the retrained references do |
| 8 | Yangtze–Huaihe lake group, *Remote Sensing* 15:4487 (2023). Excerpt only. | TP | **Sentinel-3 OLCI** | Lake group | ML | Other sensor: outside R-12 |
| 9 | CDOM / DOC regional algorithms (northern lakes S2 CDOM, *Remote Sens.* 12:157, 2020; Saginaw plume L8 + S2 DOC, 2020; estuarine aCDOM-slope DOC algorithm, NOAA; Chinese lakes DOC, *STOTEN* 2024). Excerpts only. | CDOM, DOC | S2 / L8 band ratios, or two-step CDOM → DOC | Regional | Empirical band-ratio regressions | Covered in kind: band ratios are GBM features; the two-step idea (optically active intermediate → target) is the trophic OLS chain for TP/TN/DOC. None is a fixed published formula validated for US/EU lakes at large |

Not found: any published Sentinel-2 or Landsat TP, TN or DOC retrieval evaluated on **held-out lakes in a
multi-thousand-lake set**, any released pretrained model for lake TP or TN, and any deep network for lake TP / TN
tested across unseen lakes (query vi returned none).

## Assessment against R-12

1. **TP and TN.** Every method found is a regression of the nutrient on band reflectances or indices (tree
   ensemble or small network), fitted per lake or per region. That class is what §2.5 retrains on our own rows
   (GBM families, MDN_B), and the indirect route through chl-a and Secchi is the trophic OLS chain. No
   reproducible published method outside §2.5 was found for TP or TN.
2. **DOC.** Two continental US Landsat studies exist (items 1, 2). Item 1 is covered in kind. **Item 2 is the one
   candidate under R-12**: it is reproducible in principle (public AquaSat and ERA5-Land data, standard GPR / RF)
   and uses an input class (reanalysis air temperature and other ERA5-Land variables) that no §2.5 comparator
   has. Re-running it would mean a new reference "GBM / GPR on band medians + ERA5-Land monthly covariates",
   which needs an ERA5-Land download (not imagery; size and source to be approved under R-25) and a dated
   addendum **before any Track N validation number of the arm**. Whether to add it is the lead's decision; M2 has
   not added it and has downloaded nothing.
3. **Wording.** Unless the lead adds a comparator by dated addendum, every Track N label reads, as R-12 fixes:
   "better than every retrained reference of arch-v2 §2.5; no published method was re-run", and the words
   "state of the art" are not used for TP, TN or DOC. If item 2 is not added, the DOC label should also say that
   a published Landsat + ERA5-Land DOC method (Harkort & Duan) was not re-run.

## Before any paper text

Full texts of items 2, 4, 5 and 6 must be read (splits, metrics, code availability); Scholar / Scopus must be
searched for "lake total phosphorus Sentinel-2 unseen lakes" and "DOC Landsat lake-level cross-validation"; and
the AquaSat v2 / AquaMatch papers checked for a DOC or nutrient benchmark.

## Sources

- https://pmc.ncbi.nlm.nih.gov/articles/PMC12307578/ (item 1; read)
- https://lup.lub.lu.se/record/edc8c04a-8b4d-4951-8a4b-0bead1aae1b6 (item 2; record)
- https://researcher.draco.res.ibm.com/publications/improved-dissolved-organic-carbon-prediction-in-diverse-inland-water-bodies-utilizing-machine-learning-and-remote-sensing (item 3)
- https://mdpi-res.com/d_attachment/remotesensing/remotesensing-18-01098/article_deploy/remotesensing-18-01098.pdf (item 4)
- https://mdpi.com/2072-4292/17/2/267 (item 5)
- https://mdpi.com/2072-4292/16/3/464 (item 6)
- https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2024.1432280/pdf , https://doaj.org/article/a4ce025911b649acab10e197a4cd75ae , https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2022.1014155/epub , https://meetingorganizer.copernicus.org/EGU22/EGU22-4885.html (item 7)
- https://doi.org/10.3390/rs15184487 (item 8)
- https://mdpi.com/2072-4292/12/1/157 , https://repository.library.noaa.gov/view/noaa/72890 , https://ouci.dntb.gov.ua/works/40NEbO59 (item 9)
