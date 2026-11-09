# Datasheet — Chemical identity table (`chemicals`)

| | |
|---|---|
| **Purpose** | One identity row per CAS number used by SENTINEL ChemTox: CAS → DTXSID → canonical SMILES / InChIKey / MW, with flags for structures that cannot be modelled (mixtures, UVCBs, polymers, inorganics) |
| **Scope** | every chemical in ADORE (all challenge files), in ECOTOX aquatic tests (`organism_habitat == Water`, 2026-09-15 release) and in NRC material reports with a valid CAS |
| **Sources** | (1) EPA CompTox Chemicals Dashboard search, keyless: `https://comptox.epa.gov/dashboard-api/ccdapp1/search/chemical/equal-with-detail/<CAS>` (curated DSSTox record; the documented CTX API needs a key and is not used); (2) PubChem PUG-REST name search `…/rest/pug/compound/name/<CAS>/property/SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight,IUPACName/JSON`, only for CAS numbers CompTox does not know; (3) seeds/QC from ECOTOX `chemicals.txt` (CAS, name, DTXSID) and ADORE (`chem_dtxsid`, `chem_inchikey`, SMILES) |
| **License** | DSSTox/CompTox: US EPA public data; PubChem: NCBI public-domain policy; ADORE seeds CC BY 4.0; ECOTOX: US Government work |
| **Retrieved** | 2026-10-22 (per-request timestamps in the caches) |
| **Manifest** | `chemicals` (table, summary, request caches) |
| **Code** | `sentinel/data/sources/chem_ids.py`, `scripts/data/build_chemicals.py` |
| **Local paths** | `$SENTINEL_DATA/processed/chemicals/chem_ids.parquet`, `summary.json`; caches `$SENTINEL_DATA/interim/chemicals/{comptox,pubchem}_cache.jsonl` |

## Contents (computed by `build_chemicals.py`; `summary.json`)

| Subset | CAS | with DTXSID | usable single structure (SMILES + MW) | organic | organic multicomponent | organometallic | inorganic | no_structure | unresolved |
|---|---|---|---|---|---|---|---|---|---|
| all | 10,444 | 10,437 | 9,099 | 7,193 | 1,140 | 159 | 607 | 1,340 | 5 |
| ADORE | 2,408 | 2,408 | 2,408 | 2,191 | 207 | 0 | 10 | 0 | 0 |
| ECOTOX aquatic tests | 10,343 | 10,338 | 9,038 | 7,162 | 1,134 | 159 | 583 | 1,302 | 3 |
| NRC materials | 417 | 415 | 360 | 207 | 15 | 2 | 136 | 55 | 2 |

`no_structure_hint` for the 1,345 no-structure/unresolved rows: petroleum_or_uvcb 116, isomer_or_markush 105,
mixture_or_uvcb 100, polymer 73, variable_stoichiometry 29, biological 13, unspecified 907 (mostly trade-name
formulations, natural products and undefined inorganic solids such as asbestos), 2 without a name.

Columns: `cas`, `dtxsid_final`, `name`, `smiles_canonical` (RDKit), `inchikey`, `mw_g_mol`, `structure_class`,
`structure_source`, `n_components`, `has_metal`, `dtxsid_conflict`, `no_structure_hint`, flags (`has_smiles`,
`is_multicomponent`, `is_inorganic`, `no_structure`), membership (`in_adore`, `in_ecotox_aquatic`, `in_nrc`) and all
source fields (`comptox_*`, `pubchem_*`, `ecotox_*`, `adore_*`, `smiles_source`, `inchikey_rdkit`, `mw_rdkit`).

## Method
1. **CAS normalisation** (`normalize_cas`): ECOTOX digit form (`50000`) and zero-padded NRC form (`000050-00-0`) →
   `50-00-0`; the check digit is verified; placeholders (`000000-00-0`) and invalid numbers are dropped.
2. **CompTox** (≤ 4 requests/s, 6 threads sharing one rate limiter, retries with backoff on 429/5xx; the request
   contains only the CAS number). The record whose *current* CASRN equals the query (`searchMatch == CASRN`) is
   used; otherwise a single non-CASRN hit is accepted and labelled; several non-exact hits → `ambiguous`.
3. **PubChem** only when CompTox has no record (first CID of the property table; `pubchem_n_cids` kept). It is
   never used to "fill in" a structure CompTox deliberately leaves empty (UVCBs, mixtures).
4. **RDKit 2025.09.3** recomputes canonical isomeric SMILES, InChIKey and MolWt (incl. counter-ions/hydrates) from
   the source SMILES so every SENTINEL table uses one SMILES dialect; ADORE's SMILES are the last fallback.
5. **Structure class**: `organic`; `organic_multicomponent` (salts, ion pairs, hydrates with carbon);
   `organometallic` (C–metal bond, metalloids excluded); `inorganic` (no carbon, or only inorganic carbon:
   carbonate, cyanide, CO/CO₂, cyanate, thiocyanate, CS₂, elemental C); `no_structure` (record exists without a
   usable structure — UVCBs, petroleum products, mixtures, polymers, Markush/`*` SMILES); `unresolved` (no record in
   CompTox or PubChem). For `no_structure`/`unresolved` rows `no_structure_hint` gives a name-keyword triage label
   (`petroleum_or_uvcb`, `polymer`, `mixture_or_formulation`, `biological`, `unspecified`) — heuristic only.
6. Flags: `has_smiles`, `is_multicomponent`, `is_inorganic`, `no_structure`, `dtxsid_conflict` (CompTox DTXSID ≠
   ECOTOX DTXSID), membership `in_adore` / `in_ecotox_aquatic` / `in_nrc`; source columns (`comptox_*`,
   `pubchem_*`, `ecotox_*`, `adore_*`) are kept for audit.

## QC
* 10,444 CompTox requests (0 failures, ≤ 4 req/s): 10,435 exact current-CASRN matches, 2 matched via a deleted
  CASRN, 7 without a record → PubChem resolved 2, **5 unresolved**.
* 122 CompTox SMILES are CXSMILES with attachment points/R-groups (positional-isomer mixtures such as
  "Dichlorobenzene" `Cl*.ClC1=CC=CC=C1 |m:…|`, PCB homologue classes, polymers `|Sg:n|`); they are classed
  `no_structure` and get no canonical SMILES/InChIKey/MW (10 of them RDKit cannot parse at all).
* Agreement with ADORE's own identifiers (2,408 chemicals): DTXSID 99.9% (2,406); InChIKey 98.4% identical and
  99.9% identical in the connectivity block — 37 of the 39 differences are only in the stereo/charge blocks; the 2 connectivity
  differences are a sodium salt recorded as its parent phenol in ADORE (50540-61-9) and **92-04-6, whose ADORE
  structure is a different chlorobiphenylol isomer than the CompTox record** (worth checking before modelling).
* 5 CAS numbers whose CompTox DTXSID differs from ECOTOX's mapping (`dtxsid_conflict`; CompTox's current record is
  used), e.g. zinc cyanide 557-21-1.
* ADORE SMILES were needed for 2 chemicals whose CompTox record has no structure (Disperse Blue 3, cimetidine).

## Known issues
* CompTox's dashboard backend is not a documented API; endpoint and payload may change (the raw responses are
  cached, so the table can be rebuilt offline from the caches).
* Salt/parent ambiguity: a CAS identifies the tested substance (often a salt); `mw_g_mol` is the salt MW. Models
  that need the parent compound must strip counter-ions themselves.
* `no_structure_hint` is keyword-based and not curated.

## Rebuild
```bash
python -B scripts/data/build_chemicals.py      # needs download_adore, download_ecotox and build_nrc first;
                                               # re-uses the JSONL caches (only new CAS numbers are requested)
python -B scripts/data/build_ecotox.py && python -B scripts/data/build_nrc.py   # attach identities downstream
```
