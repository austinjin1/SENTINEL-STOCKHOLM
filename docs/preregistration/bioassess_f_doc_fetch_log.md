# M5-F — documentation fetch log (pre-stamp, lead ruling Q6)

- **Written:** 2026-11-10T18:14Z by the M5 engineer (AI-assisted). Hashes and sizes were computed by script from the files on disk; the fetch time of each file is its modification time on disk (UTC).
- **Rule (lead ruling on the pre-stamp review, question 6):** before the stamp, only reference and documentation pages may be fetched: SANDRE reference records, Naïades / Hub'eau documentation and data dictionaries, and the official good-status text. No endpoint that returns observation rows. No station-level query.
- **What was fetched:** the 60 files below (8,952,203 bytes in total), kept in `sentinel_runs/bioassess/f_track/reference_docs/`. They are reference records and documentation only.
- **What was not fetched:** no Hub'eau or Naïades data endpoint (`analyse_pc`, `operation_pc`, `station_pc`, `condition_environnementale_pc`, `taxons`, `indices`, `stations_hydrobio`, any `.csv` export, any Naïades export), no station code, no taxon list, no index value, no chemistry value. No French biology or chemistry file is on disk.
- **Other requests in the same session, not kept:** (1) two probe requests for parameter 1302 (`https://id.eaufrance.fr/par/1302` and the SANDRE record URL), written to a temporary file and deleted; (2) four nomenclature requests with `outputSchema=SANDREv4`, which returned HTTP 400 (the nomenclature set is not served in version 4), deleted and repeated with `SANDREv3.1`; (3) one web search to locate the official good-status document; (4) one HEAD request on the guide URL. None of them returned observation rows.
- **Untrusted-content handling:** the files sit in their own directory; the scripts that read them live elsewhere and were run with `python -I`.

## Files

| File | URL | Fetched (UTC) | Bytes | sha256 |
|---|---|---|---|---|
| `eaufrance_guide_reee_decembre_2023.pdf` | https://www.eaufrance.fr/sites/default/files/2023-12/guide_reee_decembre_2023.pdf | 2026-11-10T18:06:24Z | 8191069 | `0a1f9df50e9f70d4052cf309d9e0bc4c60857f89731c425a8c364dd87f6c663d` |
| `hubeau_hydrobio_v1_api-docs.json` | https://hubeau.eaufrance.fr/api/v1/hydrobio/api-docs | 2026-11-10T18:04:56Z | 58609 | `ae6e1c0762e93d7e2c5fa7022564303ca3e1ce97b5ee935c3ce1def982557d1a` |
| `hubeau_page_api-hydrobiologie.html` | https://hubeau.eaufrance.fr/page/api-hydrobiologie | 2026-11-10T18:04:57Z | 40134 | `b74fa1725e83cf8821fd0f8a7083cf897d53f2998a79528c2a81d8ece0a147ed` |
| `hubeau_page_api-qualite-cours-deau.html` | https://hubeau.eaufrance.fr/page/api-qualite-cours-deau | 2026-11-10T18:04:57Z | 37631 | `4a2b81ac4a60071306f9d3d9d447ad75496c2e5ea41585a3044009c129ae8f99` |
| `hubeau_page_conditions.html` | https://hubeau.eaufrance.fr/page/conditions-generales | 2026-11-10T18:05:00Z | 36786 | `08a79bc0f18347c5a81b636ac858979b6c51ee4ecd92e9a6dd0aa622078c7d52` |
| `hubeau_qualite_rivieres_v2_api-docs.json` | https://hubeau.eaufrance.fr/api/v2/qualite_rivieres/api-docs | 2026-11-10T18:04:55Z | 168473 | `2140152f91c5d5b5d7b1dd30d0b91fa069ab3b91a5b1d99bab8e1becbaba32a2` |
| `naiades_acces-donnees.html` | https://naiades.eaufrance.fr/acces-donnees | 2026-11-10T18:04:58Z | 31484 | `3bd16ec449f81b874f83eaa31f3ddb0c6a5322f906bf4700287342def3a9a705` |
| `naiades_donnees-disponibles.html` | https://naiades.eaufrance.fr/donnees-disponibles | 2026-11-10T18:05:50Z | 27382 | `a9da1d5bcb560a49549bf534d0327a55d59b0e2a116139795391d71b0559ad40` |
| `naiades_france-entiere.html` | https://naiades.eaufrance.fr/france-entiere | 2026-11-10T18:04:59Z | 31429 | `8ffa3c689a23dc3f58c7e38981514ae497bd6c5737f1d1899400e5ea658ccbf6` |
| `naiades_gestion-des-url.html` | https://naiades.eaufrance.fr/gestion-des-url | 2026-11-10T18:05:52Z | 38245 | `856237807890e923b984eff423dfc0f43c329c561ae9f905f3cce9cee4a6399e` |
| `naiades_mentions-legales.html` | https://naiades.eaufrance.fr/mentions-legales | 2026-11-10T18:05:51Z | 26529 | `3d01dae9f48280b5277267b90a1a4651f3e61e33cfef0bf83bca4672ac4af874` |
| `naiades_organisation-des-donnees.html` | https://naiades.eaufrance.fr/organisation-des-donnees | 2026-11-10T18:05:49Z | 24437 | `0be6c8fa61c6230efcfed8982c881b75528a45b268e09fb93425072c88d5c43e` |
| `naiades_validation-des-donnees.html` | https://naiades.eaufrance.fr/validation-des-donnees | 2026-11-10T18:05:48Z | 24897 | `20f6c7db13a39fb3ce9cbc9f1c81b3eb6ccbd83be28494f26cae42ac9868b63d` |
| `sandre_fan_23.json` | https://api.sandre.eaufrance.fr/referentiels/v1/fan/23.json?outputSchema=SANDREv4 | 2026-11-10T18:04:06Z | 1831 | `24109e9430e1611023e30bf9f361592dd009a98f82450549d81efedd563f5a46` |
| `sandre_fan_3.json` | https://api.sandre.eaufrance.fr/referentiels/v1/fan/3.json?outputSchema=SANDREv4 | 2026-11-10T18:04:05Z | 2020 | `7d5b1114ddc01c11e07c08c3bd2bda95f3197b7e81f915057125b99a14bf8af8` |
| `sandre_nsa_155.json` | https://api.sandre.eaufrance.fr/referentiels/v1/nsa/155.json?outputSchema=SANDREv3.1 | 2026-11-10T18:04:33Z | 7469 | `64ec85fcbcf8650953f30c324a30ec44e6d042ba3108a536c17b6d92582993fa` |
| `sandre_nsa_156.json` | https://api.sandre.eaufrance.fr/referentiels/v1/nsa/156.json?outputSchema=SANDREv3.1 | 2026-11-10T18:04:36Z | 2506 | `c27f03cc91604511950ef0fad2a4e839be885b3c0b883d14431bbbe2d5ac8cd4` |
| `sandre_nsa_414.json` | https://api.sandre.eaufrance.fr/referentiels/v1/nsa/414.json?outputSchema=SANDREv3.1 | 2026-11-10T18:04:34Z | 3100 | `5279f67b8e74d749540eaad112df8bcff55b57d878435c930ae2b878da317209` |
| `sandre_nsa_446.json` | https://api.sandre.eaufrance.fr/referentiels/v1/nsa/446.json?outputSchema=SANDREv3.1 | 2026-11-10T18:04:35Z | 2653 | `b2b86ed5437d01b0bb7fc3084f22ce2675684d6ff8112f3164251f71a5c134ed` |
| `sandre_par_1000.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1000.json?outputSchema=SANDREv4 | 2026-11-10T18:04:05Z | 5108 | `d013927e0230c5fd3acb754680cd37c575c092c846627c0b56e3a68212a689bd` |
| `sandre_par_1022.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1022.json?outputSchema=SANDREv4 | 2026-11-10T18:03:53Z | 25657 | `7968441403894f3b0c3d4ab9dc0eda2ae73d26cf6c0af1f84af8fd8c5ecdb73c` |
| `sandre_par_1295.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1295.json?outputSchema=SANDREv4 | 2026-11-10T18:03:48Z | 4509 | `3ffdfb2521af5977432fe3da2230331cb15e15d50b32d894078940c566093927` |
| `sandre_par_1301.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1301.json?outputSchema=SANDREv4 | 2026-11-10T18:03:31Z | 4145 | `20fcc477ea2613329531bc323570f461e4fc7ed5f558e7dd6c6c0092849e3935` |
| `sandre_par_1302.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1302.json?outputSchema=SANDREv4 | 2026-11-10T18:03:32Z | 4019 | `3a13bd6066bbc5bee7c3f0eb7cf1bed6786c709dcd11350d79f4cea0a3dc6c9e` |
| `sandre_par_1303.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1303.json?outputSchema=SANDREv4 | 2026-11-10T18:03:33Z | 3826 | `7122191332bc8e4ffe84eda5e8d5f7f21e7e146d0b373fdba34b70deb3172c3d` |
| `sandre_par_1304.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1304.json?outputSchema=SANDREv4 | 2026-11-10T18:03:34Z | 2539 | `a4ca8dd7049e1f7575f27cb890d8f919cf3b0eef5ae043030cd2bfda5079618c` |
| `sandre_par_1305.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1305.json?outputSchema=SANDREv4 | 2026-11-10T18:03:35Z | 5539 | `ae57818468394cf2c8a5254bfdb97b0b18feac50f07bf77e37b970464e018651` |
| `sandre_par_1311.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1311.json?outputSchema=SANDREv4 | 2026-11-10T18:03:36Z | 5153 | `e163a7ba5e382e7efc7b0f5134ebc8487b95d7622057a9ac133ff80f1b3baa6c` |
| `sandre_par_1312.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1312.json?outputSchema=SANDREv4 | 2026-11-10T18:03:37Z | 4267 | `7c847a26c7c30751918520aa5ffb232637df22ee2af20069e5a1487c0559d2a0` |
| `sandre_par_1313.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1313.json?outputSchema=SANDREv4 | 2026-11-10T18:03:38Z | 4149 | `36c20001a79d78def0297ff9c7b59cb57102480a5ab10e4c1990d317832fe12c` |
| `sandre_par_1319.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1319.json?outputSchema=SANDREv4 | 2026-11-10T18:03:39Z | 6061 | `40377783d14ce4ecb8a237ce0611053df16eab927f4ef31af67decb6c66860f1` |
| `sandre_par_1335.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1335.json?outputSchema=SANDREv4 | 2026-11-10T18:03:40Z | 8072 | `3f7635a816965dda1626e6ac7abd389194873dc37f9b9f5eec59535342e79e41` |
| `sandre_par_1337.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1337.json?outputSchema=SANDREv4 | 2026-11-10T18:03:40Z | 5610 | `684470986918ac79c85aba597fb1579b218ebc341f23ca068af9bd71c9ea2d73` |
| `sandre_par_1339.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1339.json?outputSchema=SANDREv4 | 2026-11-10T18:03:41Z | 6919 | `2d7abc3f2bc70650a66db46eb11ae332109aa35eec355181705f19d98f539d3b` |
| `sandre_par_1340.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1340.json?outputSchema=SANDREv4 | 2026-11-10T18:03:42Z | 6892 | `04618449a3393284ef3d1dd4ec012b85ea443a34c9f8f6b198766fb98c8354e4` |
| `sandre_par_1345.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1345.json?outputSchema=SANDREv4 | 2026-11-10T18:03:43Z | 4640 | `f329ee5d067d7090ca89ff1e46e3608bbc887b09b04731f0ccb99f9b27042377` |
| `sandre_par_1347.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1347.json?outputSchema=SANDREv4 | 2026-11-10T18:03:44Z | 3458 | `e4ae98cd8af805d97bfbc0c04bdd8dcc0d9cd64367a27b720138f51c4dfdcd54` |
| `sandre_par_1350.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1350.json?outputSchema=SANDREv4 | 2026-11-10T18:03:45Z | 7424 | `a2a300760174eec756d778dc18942f42c6eed22352105795334907ebbcea0a48` |
| `sandre_par_1433.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1433.json?outputSchema=SANDREv4 | 2026-11-10T18:03:46Z | 6652 | `e801192c6ba86c55836ddbf03a987e3fd94668bd8e75011fc630aa1e3aa78604` |
| `sandre_par_1841.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/1841.json?outputSchema=SANDREv4 | 2026-11-10T18:03:47Z | 7373 | `cf44eccf37768d3ffa8978e8576027e33117bad6464818ed73cee54a6094efb1` |
| `sandre_par_5856.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/5856.json?outputSchema=SANDREv4 | 2026-11-10T18:03:50Z | 25113 | `bd5316581e962f60007d9e52904b76c4e1865c1f8599fa5aad195181c353f8b9` |
| `sandre_par_5910.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/5910.json?outputSchema=SANDREv4 | 2026-11-10T18:03:51Z | 6148 | `164ad51ea03714b1e298a24e196cf6822b25414a886c7f2f976123396a9756e5` |
| `sandre_par_6951.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/6951.json?outputSchema=SANDREv4 | 2026-11-10T18:03:52Z | 5040 | `506b660d2c88a7e04840a44e2da37c07a0a6259f4a24a298456b9f9ce07ce86e` |
| `sandre_par_7613.json` | https://api.sandre.eaufrance.fr/referentiels/v1/par/7613.json?outputSchema=SANDREv4 | 2026-11-10T18:03:49Z | 3110 | `45da54bfcff9a7beaf6acd1333d3eecd028d41ab33958ca15284fcfd17f3ddcc` |
| `sandre_sup_10.json` | https://api.sandre.eaufrance.fr/referentiels/v1/sup/10.json?outputSchema=SANDREv4 | 2026-11-10T18:04:08Z | 1498 | `57469591e7ceaab97c63697cdd870d1b600998e694666f564c9d4879f2d9ca64` |
| `sandre_sup_13.json` | https://api.sandre.eaufrance.fr/referentiels/v1/sup/13.json?outputSchema=SANDREv4 | 2026-11-10T18:04:09Z | 1685 | `efe7f4cd200ecdefe2e6f47c6881aac7daba248bec4d0c9a4692e6806763c978` |
| `sandre_sup_3.json` | https://api.sandre.eaufrance.fr/referentiels/v1/sup/3.json?outputSchema=SANDREv4 | 2026-11-10T18:04:07Z | 1280 | `11cea706faffb3ec61af5b617b602df64fd8a367ba8521a1545b91dbfaac2360` |
| `sandre_urf_147.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/147.json?outputSchema=SANDREv4 | 2026-11-10T18:04:10Z | 1452 | `8cf78ece51cb81216e40b73f66cd89fd2f41468d4ace038b5ee882283e528594` |
| `sandre_urf_162.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/162.json?outputSchema=SANDREv4 | 2026-11-10T18:04:11Z | 1478 | `ccd4011dc05707ecde6db775cbe2c91fb18e10d34e2bbf23bff1e3259b182948` |
| `sandre_urf_164.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/164.json?outputSchema=SANDREv4 | 2026-11-10T18:04:12Z | 1458 | `aa189cdd339404b31fa51c56271d8323089575714b2ffa5661bfc40bc0484a01` |
| `sandre_urf_168.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/168.json?outputSchema=SANDREv4 | 2026-11-10T18:04:13Z | 1455 | `cdf4922c7c03050ce2fbf24614a01885fa8a57cdb53560866b4b1e98dfe0e214` |
| `sandre_urf_169.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/169.json?outputSchema=SANDREv4 | 2026-11-10T18:04:13Z | 1460 | `5fb1b0b52f04b63ed7e957c0a0fd55849f0c53f51c09ae71512f07bae569ce2f` |
| `sandre_urf_171.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/171.json?outputSchema=SANDREv4 | 2026-11-10T18:04:14Z | 1460 | `6448c14bb013ffe16710191717e391263fba4d34e89a47ca303673b4c96d7707` |
| `sandre_urf_173.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/173.json?outputSchema=SANDREv4 | 2026-11-10T18:04:15Z | 1460 | `fb829e482b87a37317d841b7e365e7d139a45ad751c142baf0ba12c3c630aa93` |
| `sandre_urf_175.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/175.json?outputSchema=SANDREv4 | 2026-11-10T18:04:16Z | 1655 | `0ddf7610bbcd01e7e2fa5605b9e46bebe57191601dbb40c48d6fd4c33553b1eb` |
| `sandre_urf_176.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/176.json?outputSchema=SANDREv4 | 2026-11-10T18:04:16Z | 1462 | `685e8d529943e8d244f739bc08c9403282ac5725b8046f53397a4d7bf282fe13` |
| `sandre_urf_177.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/177.json?outputSchema=SANDREv4 | 2026-11-10T18:04:17Z | 1564 | `2b8920dd3e4958bc30893894eea26fd51b9103331f031a313f3fccb6ceb43b98` |
| `sandre_urf_243.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/243.json?outputSchema=SANDREv4 | 2026-11-10T18:04:18Z | 1545 | `41cc6c954bf819b025b3539de71e01b87ee13ea298e2959b8a06afb9f54d65ce` |
| `sandre_urf_264.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/264.json?outputSchema=SANDREv4 | 2026-11-10T18:04:19Z | 1499 | `39f4a57d240bdb5b1408d9a2756e84b4f9c01dd8c8bac61a3d84b3d9ac2c20ae` |
| `sandre_urf_27.json` | https://api.sandre.eaufrance.fr/referentiels/v1/urf/27.json?outputSchema=SANDREv4 | 2026-11-10T18:04:10Z | 1685 | `626fb3f18dd9287fe77d1f49c31b0fb334b1c759f836fb3fdb83848d23f98fa9` |

## Fetches after the stamp (released by the stamp of 2026-11-10, rulings §"order of work" item 4)

| File | URL | Fetched (UTC) | Bytes | sha256 |
|---|---|---|---|---|
| `sun_holden_birks_2024_cp-20-2373.pdf` | https://cp.copernicus.org/articles/20/2373/2024/cp-20-2373-2024.pdf | 2026-11-10T23:55:18Z | 9367408 | `542beabefbdbbde6e54640ec36739ace2bd2fe99d30e44bd85ca6d31b367ed22` |

- Sun, P., Holden, P. B. and Birks, H. J. B. (2024), "Can machine-learning algorithms improve upon classical
  palaeoenvironmental reconstruction models?", Clim. Past 20, 2373–2398, doi:10.5194/cp-20-2373-2024 (CC BY 4.0).
  Fetched for the MEMLM comparator line (protocol §8 item 2). The article pages
  https://cp.copernicus.org/articles/20/2373/2024/ and `…/cp-20-2373-2024.html` were also requested in the same
  session (HTTP 200, not kept). A paper, not a data endpoint; read as untrusted text (`pdftotext`, no code run).
- What the paper states (checked in the PDF text): first layer = random forest, extra random trees and LightGBM on
  taxon abundances, "an ensemble of 1000 decision trees"; second layer = multiple linear regression fitted on the
  predictions of an internal 5-fold cross-validation of each first-layer model; scikit-learn. Other
  hyperparameters, transformations and taxon filters are not stated in the text.
