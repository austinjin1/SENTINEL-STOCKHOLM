# SENTINEL v2 — display names (canonical table)

**Decided:** 2026-11-10, on the project author's instruction: main models are named `SENTINEL-<one word>`; the
set of downstream applications has its own family name, and each application is `<family>-<one word>`.

**Scope: display names only.** These names are used in prose: reports, papers, slides, the README and the
platform's user interface. They do **not** change:

- code identifiers, module paths and config names (`sentinel/models/chemtox`, `hydromet_sota`, `bioassess_x`, …);
- registry keys (`workstream`, `experiment`, `model`) and run-folder names;
- data paths;
- any frozen, read-only (0444) file, freeze record or pre-registration snapshot. Those keep the old names for good.

When an old name appears in a frozen file, a registry row or a path, it refers to the same model as the new name
in the table below. Reports that quote such a file give the new name once with the old one in brackets, for
example "SENTINEL-Toxin (registry: `chemtox`)".

## Main models

| ID | Display name | Old display name | What it reads | Code / registry key (unchanged) |
|---|---|---|---|---|
| M1 | **SENTINEL-Sonde** | SENTINEL-Sonde | In-situ sensor time series | `sonde` |
| M2 | **SENTINEL-Orbit** | SENTINEL-Orbit | Satellite imagery | `orbit` |
| M3 | **SENTINEL-Microbe** | SENTINEL-Microbe | 16S / ASV microbial communities | `microbe` |
| M4 | **SENTINEL-Toxin** | SENTINEL-ChemTox | Chemical structure → aquatic toxicity | `chemtox` |
| M5 | **SENTINEL-Biota** | SENTINEL-BioAssess | Macroinvertebrate and diatom communities | `bioassess` |
| M6 | **SENTINEL-Hydro** | SENTINEL-HydroMet | Weather and catchment attributes | `hydromet` |
| M7 | **SENTINEL-Lite** | SENTINEL-Lite | Drone RGB + NIR imagery | `lite` |

## System components

| Display name | Role | Code / registry key (unchanged) |
|---|---|---|
| **SENTINEL-Fuse** | Multimodal fusion | `fuse` |
| **SENTINEL-Sim** | Physics-guided simulation and scenarios | `sim` |
| **SENTINEL-Bench** | Open benchmark release (manifests, splits, scoring code, baselines, model cards) | none yet (release bundle) |
| **SENTINEL Console** | Platform (API and web app) | `sentinel/platform` |

## Downstream applications — the **AEGIS** family

| ID | Display name | Old display name | Task | Code / registry key (unchanged) |
|---|---|---|---|---|
| D1 | **AEGIS-Swim** | SafeSwim | Beach E. coli exceedance | `safeswim` |
| D2 | **AEGIS-Bloom** | ToxinWatch | Microcystin / harmful-algal-bloom advisory | `toxinwatch` |
| D3 | **AEGIS-Fish** | KillAlert | Fish-kill risk | `killalert` |
| D4 | **AEGIS-Tap** | TapRisk | Drinking-water nitrate and by-product-precursor risk | `taprisk` |
| D5 | **AEGIS-Integrity** | BioIntegrity | Biological condition class | `biointegrity` |
| D6 | **AEGIS-Mussel** | MusselGuard | Zebra / quagga mussel invasion | `musselguard` |
| D7 | **AEGIS-Trout** | TroutRefuge | Cold-water fish occupancy and thermal stress | `troutrefuge` |
| D8 | **AEGIS-Pulse** | RiverPulse | Stream metabolism at unsensed reaches | `riverpulse` |

The key column gives the usual stem. The exact identifier of a given run is whatever its registry row says.

## Names of arms inside a workstream

Architecture arms (for example SONDE-HF, BioSet-MA, GSO-MT, ThermOx-T, the M6 stack) are internal names of
candidate designs. They are not main-model names. A report names the main model first and the arm second, for
example "SENTINEL-Hydro (arm GSO-MT)".

## Changing a name later

Edit this table and the README only. Nothing else depends on a display name.
