# Architecture round v2: cross-model synthesis and 4-week roadmap (M1–M7)

**Date:** 2026-11-10, about 12:30 EST. **Type:** read-only research synthesis for the lead's decision. AI-assisted.
**Not:** a pre-registration, a result, or an edit to any protocol. Every protocol item below is a *proposal*. Each
workstream writes it into its own `docs/preregistration/<ws>_arch_v2_amendment.md` (LEAD_DECISIONS rule 1), and the
lead stamps it before any validation number exists on a new arm or track.

**Inputs.**
- The seven plans in this directory: `M1`–`M7_architecture_plan.md`.
- `LEAD_DECISIONS_20261110.md`, re-read after its 12:26 EST update. It approves, gates or defers items for all seven
  models, including the M6 rulings of 12:26.
- `docs/plans/sota_campaign.md`, `ea_england_allocation.md`, RESEARCH_PLAN §15.1–§15.5, `LANDSCAPE_SURVEY.md`.
- The M1 amendment `docs/preregistration/sonde_arch_v2_amendment.md`, **stamped 12:23 EST** (sha `6c7b169d…`), and
  `sonde_arch_v2_lead_rulings.md` (rulings R1–R5). I read the header, §4 (M1-SP) and the rulings only.

**How the evidence was handled (read this first).**

| Kind | Rule used here |
|---|---|
| Repo numbers | Carried from the plans, which re-read their run files this session. I recomputed none of them. |
| What I measured | GPU, disk and load snapshot at 12:18 EST (§2.1) and the label-count probability arithmetic (§5.2). |
| Plan probabilities | Subjective estimates by each plan's author. This project's dev→test history shows large shrinkage: the M4 ensemble gain halved, −0.062 → −0.029; the M7 student lost 7.8 MdSA from val to Test-A; BioSet alone was significantly worse on M5 Test A. So §5 also gives a discounted ("low") portfolio. |
| Citations | This file cites only what I found by web search or fetched **this session** (Sources, end). Everything else points to a plan section and keeps that plan's tag ([S]/[F]/[R0x]/[U]). |
| Files written | This one file. No training, download, install, git or process action. |

---

## 0. Executive summary

1. **One diagnosis covers all seven models.** Simple ML ties because of **information and test power**, not capacity:
   - Candidate and GBM see the same inputs. M1 sees lag-only daily data. M2 sees one date and the centre 150 m. M4 has
     one endpoint and a 4-level duration code. M6's GBM already holds the memory an LSTM would learn. M7's student
     lacks the scene statistics the GBM has.
   - Ensembling, not architecture, explains most of each current lead:
     - M1: a baseline-only ensemble lands within ±0.01;
     - M2: avg(MDN_B, GBM) scores 49.7 against SatWQ-Net's 48.3;
     - M5: ensemble C;
     - M4: the stack gain halves on new chemicals.
   - Test blocks are underpowered:
     - M2 headline joint power 0.21;
     - M6 MDE 0.014 KGE against observed gaps of ~0.006;
     - M3 holdout MDD ≈ 0.16–0.45 R² at ~32 effective sites.

   No plan recommends a bigger backbone, and I agree.
2. **The proposed "novel architectures" share one motif, and that is good:** *information-carrying, bounded-residual
   hybrids*. A learned component sits on top of a strong anchor:
   - M1 SONDE-HF: damped-anomaly anchor plus an NWP forecast branch;
   - M2 A1: lake-context set separating persistent bias from scene error;
   - M4 TED: duration composed through a monotone law plus a bounded residual;
   - M6 D2: energy-balance core plus a residual; M6 D1: diel observation operator for grab samples;
   - M7 ANCHOR: cross-fitted GBM anchor plus a bounded CNN residual;
   - M3 D1: inversion with a context prior.

   Each has a falsifiable "what the GBM lacks" claim and a mandatory ablation that turns the novelty off.
3. **The round fits in GPU-hours even on the 2-GPU floor.**
   - Demand: ≈ 180 GPU-h (core) to ≈ 280 GPU-h (with every gated or optional arm) over 4 weeks.
   - Floor capacity: ≈ 150–165 GPU-h/week realistically available to this round (§2).
   - The binding constraints are **serial dependency chains, CPU contention** (load 45 on 48 cores at 12:18) and
     **lead stamps**.
   - The plans' own per-day assumptions (3–7 GPU-h/day each) add up to ≈ 240 GPU-h/week, which over-subscribes the
     floor. Their *total* needs do not.
4. **The best payoff per GPU-hour is cheap information and power, not big networks** (§1):
   - zero-GPU items first: M2 Test-S protocol text, M4 T1 cluster freeze plus the R1–R3 references, M3 D1/D3 on CPU,
     M6 label builds, M7 national-2b labels;
   - then the 1–3 GPU-h items: M4 embedding cache and TimeHead, M2 A0, M7 ANCHOR/WaterSet, M1 E2 and the step-1
     information-use gate;
   - the large arms come after: SONDE-HF, A1, TED, BioSet-MA, GSO-MT/ThermOx-T.
5. **New cross-model findings from this synthesis** (§3):
   - **(a)** M3's D1 "NicheInvert" is, in ter Braak's terms, *classical* (forward-model) calibration. That is the
     family that failed worst in M5 (bio.infer r2_anomaly −1.08 / −1.14, slopes 0.2–0.4). D1 needs a bio.infer-style
     ablation and a calibration-slope report before its gate.
   - **(b)** M4 T6a needs a published mechanistic per-pair LC50(t) extrapolation baseline (DEBtox-style; found this
     session). TED's time law has closer prior art than GUTS alone.
   - **(c)** StreamPULSE overlap between M1-SP (test block, lock-on-download) and training on the processed
     StreamPULSE metabolism labels. The stamped M1 amendment §4 already handles most of it: sites whose labels anyone
     has read form a hard tier outside the M1-SP primary. **Remaining:** M6's DO-stage use of those labels must be
     visible in the registry before M1's exposure audit runs.
   - **(d)** M7's KD teacher and CIR-Lite-R's S2 calibration targets must exclude the newly reserved M2 Test-S
     site-dates.
   - **(e)** Six workstreams use unverified physical constants. One shared, verified constants module saves five
     separate checks.
6. **Honest outlook** (§5):
   - M5 already holds a scoped SOTA label (ensemble C on Test A).
   - The chance that ≥ 2 *more* models earn a scoped label in this round is ≈ 0.6 on the plans' numbers, or ≈ 0.4
     discounted.
   - The chance that ≥ 4 more do (i.e. ≥ 5 of 7 with *any* scoped label) is ≈ 0.07 (0.02–0.15).
   - RESEARCH_PLAN H-G1 counts only *pre-registered primary endpoints*, so it is lower still.
   - Most likely outcome: 2–3 models with new scoped labels (M2 Test-S Secchi, M4 T6a/T1, M5 France or M5-X,
     M7 P2), and well-documented "competitive" results elsewhere.

---

## 1. Ranking: expected SOTA payoff per GPU-hour

**Definitions (mine).**
- **ΔP** is the increase, in percentage points, in the probability that the owning model earns ≥ 1 scoped §15.1 label
  in this round, attributable to the item. It assumes the mandatory references are built. It is derived from each
  plan's §9 probabilities and its arm ranking. It is subjective, and the range is wide.
- **GPU-h** is the item's own GPU time. For gated items it is the *expected* cost: gate cost + P(pass) × run cost.
  References are a fixed cost, listed separately in §1.3.
- **Payoff** = ΔP / GPU-h.

### 1.1 Zero-GPU items (do first; ranked by ΔP and urgency)

| # | Item | Model | CPU / effort | ΔP | Why first | Status (LEAD_DECISIONS) |
|---|---|---|---|---|---|---|
| Z1 | **Test-S protocol text** (§11.7: scene-disjointness rule, MDE rules) + Track N + dev_v5m join | M2 | 6–10 CPU-h | **+10 to +15** | Roughly doubles test blocks (143–222 vs 87). Must precede the headline M2/test look (chain6/7 finish ~12–13 Nov) | Approved and reserved; text pending |
| Z2 | **M5-X single look** proceeds as planned | M5 | lead slot | P(≥ 1 target) ≈ 0.45 | The nearest-term label in the project. Do not delay it | Untouched (rule 4) |
| Z3 | **T1 ECFP4-Butina cluster file + hash**, class-level T2 exclusion, **R1 monotone GBM, R2 AFT GBM, R3 multitask GBM** | M4 | ~35–55 CPU-h | enabling (M4 0.35–0.40 rests on it) | R1/R2 falsify two "GBM-impossible" claims. The T1 freeze must precede the embedding cache | Approved |
| Z4 | **M1-N v0.6 + M1-H + M1-SP amendment** | M1 | done | protects ≈ 0.30 | Had to precede any NWP-based val number | **Stamped 12:23 EST**; M1 is cleared to train |
| Z5 | **D1 NicheInvert, D3a/D3b V4-MR, D4 qPCR** + never-run parity baselines (RF/TabPFN on V4-CLR, XGB-3000, WA-PLS) | M3 | ~35–50 CPU-h | +3 to +5 | M3's whole architecture path is CPU. It gates D2 | Approved (#6–#8) |
| Z6 | **M3E multi-determinand amendment** + header-only ID check of the Thorpe file; `microbe_ea_ids.json` partition | M3 | hours | enabling H3 (0.25–0.30) and H4 (0.15–0.20) | The best M3 route. Must precede any EA label download | Approved |
| Z7 | **National-2b labels** (FNU 471, NTRU 323, SSC 456, censored 786; 0 GB) + C2 GBM-parity + OOF anchors + **N8 amendment** | M7 | ~5 CPU-h | enabling | National freeze deadline ≈ 2026-11-20 | Approved |
| Z8 | **Label builds**: WQP grab samples WY2008–21 (2.16M T / 1.86M DO station-days), not-in-split DV sites, VAA/solar channels; G1/G2 PF-GBM; EA group metadata count | M6 | ~25–40 CPU-h | enabling (G-D1a) | The largest zero-download data lever in the project | Approved 12:26 (hygiene rules 1–5 mandatory); amendment drafted by M6, then stamped |
| Z9 | **M5-F protocol v0** + France availability/pairing count (masks only) | M5 | ~8 CPU-h + ≤ 5 GB | enabling F-A 0.35–0.45 | Must precede any French download | Approved (cap 5 GB) |
| Z10 | GBM-NWP + PF-GBM references; GEFS pilot (one reforecast WY + one operational month) | M1 | ~10–20 CPU-h + pilot | enabling | References first (rule 3) | Approved (R3: c00 only, ≈ 0.6 TB streamed, < 1 GB kept, separate decode env) |
| Z11 | D3 SENTINEL-Stack + references-only control | M6 | CPU | +1 to +2 | Cheap. Varadharajan 2025 (M6 plan §2 [F]) found ensembles most accurate | Approved 12:26 |

### 1.2 GPU items ranked by payoff per GPU-hour

Ref-cost = the mandatory references the item cannot be claimed without (§1.3).

| Rank | Item | Model | GPU-h (exp.) | ΔP (pts) | Payoff (pts/GPU-h) | Gate / kill rule | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **ChemTox-RX**: concatenated pretrained-embedding cache (~7k structures) + decorrelation-selected ensemble | M4 | 1.5 | +5 (2–8) | **3.3** | Concatenate, never substitute; parity R4 to every comparator | No novelty, highest expected gain |
| 2 | **ChemTox-TimeHead** (monotone time-law head over a frozen GBM) | M4 | 1 | +3 (1–5) | **3.0** | Runs before TED. Must beat R1 *and* the new per-pair DEBtox-style LC50(t) baseline (§3.4) | Decides whether TED's trunk earns its GPU |
| 3 | **M1 step-1 information-use test** (train-block inner CV) | M1 | 2 | ≈ 0 direct | gate | < 0.01 CRPS skill over GBM-NWP at h3–7 for T and DO → skip PhiDA-Lite and Chronos-2-NWP | Value of information: decides ≈ 32 GPU-h |
| 4 | **ANCHOR-Lite** (GBM-anchored, context-complete residual CNN): national slot 7 + A1 slot 8 + P2 configs | M7 | 6 | +8 (4–12) | **1.3** | National: freeze without it if not ready by ~11-16. Ablations A1/A3–A5 | Ref-cost: C4b DenseNet121 2.5 GPU-h |
| 5 | **WaterSet-Lite** (set encoder over water pixels; ANCHOR's alternative residual branch) | M7 | 1.5 | +2 | **1.3** | P2 cap | CPU-capable |
| 6 | **A0 variance-control arm** (3-init + EMA seed unit) | M2 | 3 | +3 (2–5) | **1.0** | Approved **only with parity**: comparators get 3-init averaging, GBM 3-seed bagging. This shrinks A0's edge | Attribution control for A1 |
| 7 | **E2** CV-weighted ensemble + matched control **E2-B** | M1 | 2 | +2 | **1.0** | Credited margin = non-baseline members only | Fixes the temperature-h1 loss (−0.014) |
| 8 | **SONDE-HF** (multi-resolution hindcast→forecast multi-task LSTM; GEFS members) + M1-H hourly variant | M1 | 24 | +15 (10–20) | **0.63** | Kill: R-NH alone matches → label "competitive" | Ref-cost: R-NH 5 + Chronos-2-cov 2 |
| 9 | **D1 GSO-MT** (cuDNN multi-task, learned diel observation operator, ~20k grab-sample stations) | M6 | 8–10 | +5 (3–8) | **0.55** | G-D1a: grab samples must cut inner MSE ≥ 3 %, else dropped for all models. G-D1b val go/no-go | Ref-cost: R-TD 3 |
| 10 | **D2 ThermOx-T**, temperature stage (energy balance + source mixing, attribute-learned bounded parameters) | M6 | 9–10 | +5 (2–8) | **0.53** | Smoke > 0.5 s/step → trim, then drop | Its ground: T-E zero-shot and T-H |
| 11 | **A1 SatWQ-LakeSet-MT** (multi-date same-lake set context + land ring + TP/TN/DOC heads) | M2 | 16 | +7 (4–10) | **0.44** | Kill: < 2 MdSA over v1-dev_v5 on val chl-a and Secchi. c3 (no context) ≈ c1 → drop phase B and A3 | Parity: GBM/MDN_B also get the context summaries |
| 12 | **ChemTox-TED** (time-and-endpoint-decomposed multitask D-MPNN, Tobit, ACR) | M4 | 26 (22–30, cap 32) | +10 (5–15) | **0.38** | Fallback to 8 configs if the smoke exceeds 30 min/job. A1 (mechanism off) within MDE → no mechanism claim | Ref-cost: MMoE 8–12. R1–R3 on CPU |
| 13 | **BioSet-MA** (multi-assemblage, multi-target Student-t set model; P3 region token inside) | M5 | 17 | +6 over re-using XBioSet as the ensemble member | **0.35** | Kill: > 0.02 below the best single comparator on ≥ 4 of 6 targets → ensemble member only | Ref-cost: TabPFN v2 comparators 8–12 |
| 14 | **PhiDA-Lite** (heat + one-station DO physics on SONDE-HF, Powell latents) | M1 | 2.5 + 0.5 × 17.5 ≈ 11 | +4 | **0.35** | Gate C3 (physics-only beats damped anomaly; metamorphic checks; GPP ρ ≥ 0.5) | Carries X3-heat and the GPP/ER first-in-class claim |
| 15 | **BioPFN** (TabPFN v2 + learned embedding + retrieval context) | M5 | 6 | +2 | 0.33 | TabPFN-2.5 not used (lead) | 8–10 GB VRAM: needs a card to itself |
| 16 | **NicheGeo-Phys D2** (gated on D1) | M3 | ≈ 0.35 × 12 ≈ 4 | +1.5 | 0.35 | D1 gate, then stop after 4 GPU-h | XGB-chained-physics is mandatory |
| 17 | **Chronos-2-NWP** fine-tune with NWP covariates | M1 | 12 (only if step 1 says go) | +1.5 | 0.13 | Step 1 | VRAM risk on 11 GB |
| 18 | **A2 SatWQ-IOP-lite** (physics decoder ELBO) | M2 | ≈ 9 (gate ~0.5 × 18) | +1.5 | 0.17 | chain9 SSL ≥ 1 point, or A1 fails | Constants unverified |
| 19 | **A3 TOA dual-stream** | M2 | 5 (+ 0.9 GB) | +1 | 0.2 | Disk approval; access cost [U] | Test-C / Track G |
| 20 | **CIR-Lite-R** (reduced physics, 4 latents) | M7 | ≈ 3.5 (CPU calibration gate first) | +0.7 | 0.2 | ≥ 10 % calibration gain gate | Must exclude Test-S site-dates (§3.3) |
| 21 | D2 DO stage | M6 | 7 (gated by G-DO) | +0.5 | 0.07 | G-DO | DO information largely absent (M6 §1.6) |
| 22 | SparK SSL init | M7 | 4 | +0.3 | 0.08 | Only if a P2 slot remains | |
| 23 | ΔBio-Siamese | M5 | 3 | +0.2 | 0.07 | Secondary | |
| 24 | Upstream GAT / upstream features | M1 | 6 | +0.3 | 0.05 | **Ruled out for v0.6** (lead ruling R2: no upstream-sensor inputs for any model) | Median edge 116 km |
| 25 | Cycle-6 SeqSet (already approved, ≤ 6 GPU-h) | M3 | 6 | +0.3 | 0.05 | MiCoDa only if #4 > #3 | Schedule last |
| 26 | A4 AlphaEarth prior; R-EA continuity re-run | M2 / M6 | 3 / 7 | ≈ 0 | ≈ 0 | Ablation / descriptive only | Skip unless GPUs are idle |

**Reading.**
- The top seven items cost ≈ 17 GPU-h together and carry roughly as much expected label value as the next four large
  arms (≈ 76 GPU-h).
- The large arms are still worth running. They are where any *architecture* claim (as opposed to a data or ensemble
  claim) can come from, and each has a kill rule.
- Items 17–26 should run only on otherwise-idle GPUs, or not at all.

### 1.3 Fixed cost: mandatory references (rule 3; not optional, not ranked)

| Model | References (GPU) | GPU-h | CPU references |
|---|---|---|---|
| M1 | R-NH (Nearing-style hindcast–forecast LSTM), Chronos-2-cov zero-shot, E2-B | ≈ 7.5 | GBM-NWP (quantile LightGBM, trajectory features), PF-GBM |
| M2 | — | 0 | MDN_B family (±context/ring, joint head), GBM family (+context summaries, +phys), kNN, trophic-chain for TP/TN |
| M3 | — | 0 | RF/TabPFN on V4-CLR, XGB-3000, WA-PLS, XGB on z-posterior, XGB-chained-physics |
| M4 | MMoE (Wu et al. 2026 form; comparator and ensemble member) | 8–12 | R1–R3; R5 TRIDENT per-SMILES errors if the deposit is verified; **new: per-pair DEBtox-style LC50(t) for T6a** (§3.4) |
| M5 | TabPFN v2 2.2.1 comparators | 8–12 | RF, LightGBM, multi-output RF, WA-PLS2, kNN-MAT, bio.infer, I2M2/IBD index regressions, RF[geography], region-shift-corrected baseline ensemble |
| M6 | R-TD (published top-down LSTM, re-run) | ≈ 3 | G1, G2 PF-GBM with the grab-sample formulation, air2stream/dosat |
| M7 | C4b HydroVision-style DenseNet121 | ≈ 2.5 | C2 GBM-parity, C2-AFT, C3 stacking, C4a band ratios, C4c CyFi-style LightGBM |
| **Total** | | **≈ 30–37** | |

### 1.4 Harder tracks ranked by P(label) per incremental cost

| Rank | Track | Model | P(label) (plan) | Incremental GPU-h | New data | Approval |
|---|---|---|---|---|---|---|
| 1 | **Test-S** (test-role lakes × 2015–2023): chl-a, Secchi | M2 | Secchi 0.35–0.5; chl-a 0.2–0.3 (v1), 0.3–0.45 (v2) | ≈ 1 (inference) | 0 GB | Reserved; text pending |
| 2 | **France F-A / F-Z / F-T** (one look, one Holm family) | M5 | F-A ensemble 0.35–0.45; F-T 0.30 (correlated); F-Z 0.20 | ≈ 6 test + shared dev | ≤ 5 GB | Approved |
| 3 | **T6a duration extrapolation** / **T1 unseen ECFP4 clusters** | M4 | 0.30 / 0.25 (MDE 0.006) | shared with TED | 0 GB | Approved |
| 4 | **P2 rivers/sondes FNU at new water bodies** | M7 | 0.30 (0.15–0.45) | 3–9 | ≤ 0.5 GB | Approved |
| 5 | **H3 EA NEON→England** (Spearman) | M3 | 0.25–0.30 | ≈ 0 (CPU) | 85 MB + < 0.3 GB | Approved (M3E) |
| 6 | **M1-H** hourly DO / hypoxia hours | M1 | 0.20–0.25 | ≈ 6 | 0 GB (hourly tier on disk) | Stamped. Scored in the **same** single look as M1-N, as its own family (R4) |
| 7 | Track N (TP, TN, DOC on Test-S) | M2 | 0.1–0.2 each; DOC 0.15–0.3 | 0 (inside A1) | 0 GB | Approved |
| 8 | H4 EA-internal area-blocked | M3 | 0.15–0.20 | 0 | as H3 | Conditional (ruling 4a) |
| 9 | **T-E EA England zero-shot** | M6 | 0.15 (0.10–0.25) | ≈ 6 | EA build (M6 cap 20 GB) | Approved as M6's primary opportunity (12:26). Look ≥ 5–7 weeks away |
| 10 | Track G (international lakes, coordinate-free) | M2 | 0.15–0.3 | 3–6 | ≤ 8 GB; licences [U] | Drafted, not funded |
| 11 | M1-N-F with GEFS | M1 | 0.10 (vs 0.06 without) | in SONDE-HF | ≈ 0.6 TB streamed (c00 only), < 1 GB kept | Approved with pilot and stop conditions (R3) |
| 12 | GEO-8; GAS-3 | M3 | 0.10; 0.10–0.15 | 0 | 0 GB; GAS-3 tens–hundreds MB [U] | GEO approved; GAS-3 before download |
| 13 | X3-heat / X1-storm strata; M1-SP StreamPULSE | M1 | 0.10–0.15 each | 0 / small | 0 / ~0.5 GB | Stamped (secondary); M1-SP exposure audit and hard tier (§3.3 C1) |
| 14 | P3 France zero-shot; P4 NLA habitat | M7 | 0.10–0.15; 0.20 | ≈ 0.3 inference | 0.2–1.4 GB | Deferred / conditional |
| 15 | T5 cross-source; T6b | M4 | 0.15; 0.15 | — | licence [U] | Not approved / stratum |
| 16 | T-H heatwave; T-B/N2 (Q3 2027) | M6 | 0.06; 0.05 each | small | — | Approved. T-B is **one joint look** with the v1 `test_b` models after 2027-06-30. T-D is descriptive only |
| — | T2 unseen taxa (in flight) | M4 | 0.10 | 8–15 remaining (queued) | — | Runs as registered |

**Pattern.** Six of the top eight tracks win mainly through **power**: more independent clusters (Test-S, T1, France,
P2, EA) or new information that a sequence or spatial model can use (T6 shape, hourly DO). This matches campaign
rule 4: more data first, with parity.

---

## 2. Compute: realistic budget and allocation

### 2.1 What is actually available (snapshot, 2026-11-10 12:18 EST)

**The rule.**
- The computed task assumed GPUs 0–1 only.
- `LEAD_DECISIONS` rule 5 and the lock file `sentinel_runs/.gpu_locks/ALLOWED` (contents `0,1,…,9`) allow GPUs 0–9.
  Conditions: the lock helper with `--min-free-mb`, ≤ 2 h per job, never a GPU with another user's process.

**Memory in use at 12:18 EST** (`nvidia-smi`, read-only):

| GPU | Used / 11,019 MiB | State |
|---|---|---|
| 1, 4, 6, 9 | 8 MiB | idle |
| 0 | 1,095 MiB | light |
| 2, 3, 5, 7 | 3.4–6.8 GB | partly used |
| 8 | 10.8 GB | full |

**Host.**
- CPU load average **45 / 48 cores**.
- Disk **183 GB free** (floor 60 GB, so ~123 GB of headroom). The per-workstream caps are unchanged and still bind.

**Decision needed (U1, §4.3).** Which regime this round plans against. I size everything for the **2-GPU floor**; the
0–9 regime only shortens wall-clock time.

### 2.2 Realistic GPU-hours per week

| Regime | Gross | Utilisation assumption | Available to this round |
|---|---|---|---|
| **Floor (GPUs 0–1)** | 336 GPU-h/week | ≈ 65 % (lock waits, ≤ 2 h chunks, smoke failures) → ≈ 220; minus ≈ 25–30 % for queued and non-round work (M4 T2 chemprop 8–15 GPU-h, M2 chain6–9, M7 running configs, M5-X and M1 Test-A looks, Sim, Fuse, EW) | **≈ 150–165 GPU-h/week**, ≈ 21–23 per day across the round, ≈ 3 per workstream-day |
| **Ceiling (0–9, ~5 usable now)** | — | — | ≈ 500+ GPU-h/week. The whole round (≤ 280 GPU-h) fits in about 1 week of compute |

**Demand over 4 weeks.**
- Core: ≈ 180 GPU-h = references ≈ 33 + candidates ≈ 150.
- With every gated or optional arm: ≈ 280 GPU-h.

On the floor that is ≈ 30–45 % of 4 weeks' capacity, but **weeks 2–3 peak at ≈ 75–110 GPU-h**.

**Contention rule (proposal for the lead).** When a lock is contended, serve jobs in this order:
1. **Deadline-bound frozen paths:** M7 national freeze (~11-20), M5-X look, M2 headline look, M4 T2 completion.
2. **Mandatory references** (§1.3).
3. **Candidates by §1.2 payoff.**
4. **Gated or optional arms.**

Peak VRAM of the main arms is ≤ 4–5 GB (BioPFN 8–10 GB and C4b ≈ 7 GB excepted), so two arms can co-reside on one
card where the lock helper allows it.

### 2.3 CPU (the real bottleneck)

- **Total demand.** About 170–275 heavy CPU job-hours at 6 threads over 4 weeks:

  | Model | CPU job-hours |
  |---|---|
  | M1 | 10–20 |
  | M2 | 16–28 |
  | M3 | 35–50 |
  | M4 | 35–55 |
  | M5 | 30–55 |
  | M6 | 25–40 |
  | M7 | ≈ 8 |

  The total is small. The problem is *concurrency*: seven workstreams × 6 threads = 42 threads, on a host already at
  load 45.
- **Proposal 1: cap concurrency.** At most **4 heavy CPU jobs round-wide** at any time, sequential within each
  workstream (lead rule 5). In week 1, order the queue as:
  1. M2 dev_v5m (blocks A0/A1 and the Test-S power check);
  2. M4 R1–R3 + T1 clusters;
  3. M7 national-2b + C2 (deadline);
  4. M3 D1/D3a;
  5. M6 label builds;
  6. M5 France counts;
  7. M1 GBM-NWP.
- **Proposal 2: move GBMs onto idle GPUs.** Fit GBM references with XGBoost `device="cuda"` on idle GPUs. The
  device must be fixed per workstream in the amendment, before any val number, and identical for candidate-side and
  reference GBMs. That moves the heaviest CPU load to idle GPUs.

### 2.4 Allocation and week-by-week sequence (floor regime; GPU-h are round-new work only)

| Week | Lead / user decisions and stamps | CPU (≤ 4 concurrent heavy jobs) | GPU (floor) | Looks and freezes |
|---|---|---|---|---|
| **W1, Nov 10–16** | M1 stamped (12:23); M6 ruled (12:26). M2 Test-S text **before** the headline look. M4 amendment (T1 cluster hash, T6 bands, R1–R5, DEBtox baseline). M5-F v0. M6 amendment drafted by M6, then stamped. M7 N8 + `lite_p2_protocol.md`. M3 GEO + M3E. Cross-model items X1–X8 (§4.1). Decisions U1–U6 | Z1–Z10 in the queue order above. GEFS HEAD request. Thorpe header check | ≈ 12–15: M7 ANCHOR + A1 (1.4) + C4b national (1.5); M4 RX cache (1.5, after the T1 hash); M2 A0 (3, after dev_v5m); M1 step-1 test (2; cleared by the stamp); M3 cycle-5 #1–2 (0.7); smokes M1/M6/M5 (≈ 2). *Queued:* T2 chemprop, chain6–9 | M5-X single look. M2 headline M2/test (after Test-S text) |
| **W2, Nov 17–23** | M2 A2 gate (chain9 result). D1 gate for M3 D2. GEFS pilot outcome (auto-continue unless a stop condition fires). EA split freeze (M6) unblocks T-E and M1 H-4 | M3 D3b, GEO baselines, EA read processing. M5 F baselines. M6 G1/G2 on enlarged labels. M1 PF-GBM | ≈ 75–90: M1 R-NH 5 + Chronos-2-cov 2 + SONDE-HF CV 15; M2 A1 16; M4 TimeHead 1 + TED ≈ 12 + MMoE ≈ 4; M6 R-TD 3 + D1 7.5; M7 P2 ANCHOR/WaterSet 3 + C4b-P2 1; M3 cycle 6 ≤ 6 (+ D2 ≤ 4 if gated); M5 TabPFN v2 dev 8–12 (if France counts are done) | M7 national freeze + Test-N look (~11-18 to 11-20). M4 T2 look |
| **W3, Nov 24–30** | M1 PhiDA gate outcome; Chronos-2-NWP go/no-go. M6 G-D1a / G-DO. Independent review of all new single-look guard extensions in one pass (§3.6) | M3 freeze build. M6 D3 stack + control. M5 P1 ensembles. M2 Test-S power from protected counts | ≈ 50–110: M1 PhiDA gate 2.5 (+ CV 16 if passed) or M1-H 6; Chronos-2-NWP 12 only if go; M4 TED rest ≈ 14 + MMoE ≈ 6; M6 D2-temperature 8.5; M5 P1 F dev 12 + BioPFN 4; M2 A2 ≤ 18 if gated; M7 P2 ablations / CIR-Lite-R ≤ 7 | M2 M2S/test (Test-S + Track N; v1 + v2, Holm over 2 arms). M3 H-M3 + GEO-8 sealed look. M7 P2 look |
| **W4, Dec 1–7** | Freeze records for M1-N, M4 T1/T6, M5-F | Pre-test reviews. Write-ups to `*_test_results.md` | ≈ 40–60: M1 finals + E2/E2-B 8 + rehearsal 2; M4 seed refits 2 + looks ≈ 4 (needs the >2 h exception); M5 test pass 6; M6 D2-DO 7 if G-DO; M2 Track G 3–6 if licences clear; contingency ≈ 15 | M1-N + M1-H in **one** single look, two families (R4). M4 T1 + T6 looks. M5-F look (may slip to W5 on data access). M3E look. Stamps need their git commit and public push before any freeze (updated stamp procedure) |
| **After W4** | — | — | — | M6 `M6EA/test` (≈ late Dec–Jan). M2 Track G. M3 Test-B (Jan 2027). M6/M1 joint Test-B (after 2027-06-30). M2 Test-C (mid-2027) |

**Per-workstream GPU-h over the 4 weeks.** These are my allocation, consistent with the plans' own estimates:

| Model | Core | With gated or optional arms |
|---|---|---|
| M1 | ≈ 42 | ≈ 70 |
| M2 | ≈ 20 | ≈ 48 |
| M3 | ≈ 9 | ≈ 21 |
| M4 | ≈ 40 (incl. MMoE) | ≈ 48, plus T2 remaining |
| M5 | ≈ 30 | ≈ 33 |
| M6 | ≈ 30 | ≈ 37 |
| M7 | ≈ 11 | ≈ 20 |
| **Total** | **≈ 182** | **≈ 277** |

---

## 3. Cross-model synergies (and conflicts)

### 3.1 One parity toolkit and one control rule (engineering synergy; zero GPU)

**Parity references.** Every plan builds a "GBM with the candidate's information" reference:
- M1 GBM-NWP / PF-GBM;
- M2 GBM + context summaries + GBM-phys;
- M3 XGB-chained-physics and XGB on the z-posterior;
- M4 R1–R4;
- M5 multi-output RF + region-shift-corrected ensemble;
- M6 G2 with the grab-sample formulation;
- M7 C2/C3.

**Proposal: one shared parity assertion.** Adopt M4's mechanical parity check (§4.7: row-count and column diff per
fold, asserted in code and logged) as one utility used by every workstream, instead of seven promises.

**Proposal: one matched-control rule, project-wide.** Any SENTINEL ensemble or stack is reported next to a
**baseline-only ensemble that uses the identical weighting rule**. Today this exists as M1 E2-B and M6's
references-only stack. M2 reports avg(MDN_B, GBM) only descriptively. Without the rule, ensembling gains get credited
to SENTINEL. Ensembling is the shared source of every current "lead" (§0.1).

### 3.2 Shared physics primitives: verify once, use six times

These constants are marked **UNVERIFIED** in the plans, and the same functions recur:

| Primitive | Used by |
|---|---|
| Benson–Krause DO saturation (+ elevation pressure) | M1 PhiDA-Lite and DO %sat target; M3 D2; M6 G2/D2; M5 targets |
| Raymond 2012 K600 | M1 PhiDA-Lite; M6 D2-DO |
| McCleskey 2012 conductivity from ions | M3 D2; M5 P4 |
| Carbonate system (Plummer–Busenberg) | M3 D2 |
| QAA / Lee 2015 Secchi / Bricaud / Pope & Fry / Nechad / Dogliotti | M2 A2; M7 CIR-Lite-R; GBM-phys features |

**Proposal (X7).** One `sentinel/physics_constants` module with unit tests against values transcribed from the
primary texts, with page references, verified once. It must exist before any amendment that uses these constants is
stamped. It replaces five or six separate checks and removes the risk of two workstreams using different constants
for the same quantity.

### 3.3 Shared data: reuse, and four conflicts to resolve before any download or fit

**Reuse.**

| Asset | Used by | Proposal |
|---|---|---|
| USGS sonde tier (`processed/sonde_daily`, `sonde_hourly`) | M1 targets and hourly context; M6 auxiliary labels (min/max, DO max, Q, hourly); M2 Track F FNU; M7 P2 FNU | One sha-pinned **role table** with the union of exclusions: 729 `sonde_newsites`, EW-v2 WY2024–25 test-role, M1 `sonde_split` test sites, M6 test-role sites |
| Powell metabolism | M1 PhiDA latents (M1 split; 48 new-site overlaps excluded); M6 D2-DO (FINAL split; 117 test-role sites reserved by Sim H-G3) | One role table that honours all three reservations |
| WQP discrete | M2 Track N (TP/TN/DOC); M6 grab samples (streams); M7 P2 (FNU/NTRU/tube/DOC at new water bodies) | A shared station-role column, and counts-only overlap reports between M7 P2 test water bodies and M2 test-role lakes |
| EA England | M6 T-E (Hydrology API sondes); M1 H-4 (M1-test groups); M3 H3/H4 (Thorpe biofilm chemistry, a different dataset); M5-X X-U (frozen test keys) | The **M6 EA build and split freeze is a critical-path item**: it unblocks M6 T-E and M1 H-4 |
| France (Naïades / Hub'eau) | M5-F (biology + paired chemistry; 2023–25 reserved); M7 P3 (Secchi/turbidity/MES, zero-shot evaluation only); M6 listed France for hydromet | **One pull, one manifest, one licence record (Etalab)**. Role partition written before download. Both uses are evaluation-only on any overlapping rows, so neither contaminates the other. Fuse must never combine M5-F and M7-P3 outputs on those rows |
| GLORIA, AquaCorr, S2 chips | M2 pretraining and matchups; M7 S2 teacher and calibration targets | See conflict C2 |

**Conflicts found by this synthesis** (each needs a dated rule before the relevant fit or download):

- **C1. StreamPULSE.**
  - M1-SP uses StreamPULSE sites as a **new-source test block, locked on download**.
  - M6 D2-DO plans to *train* on the already-processed `labels_streampulse` (62,383 site-days; 738 site records per
    M6, 133 sites per M1).
  - **Mostly resolved** by the stamped M1 amendment §4:
    - sites whose labels any workstream has read form a **hard tier**, excluded from the M1-SP primary;
    - M1-SP metabolism labels are never used for PhiDA-Lite;
    - the exposure audit reads file and registry metadata only.
  - **Remaining action.** The audit can only see reads that are registered. So M6 should register which
    StreamPULSE sites its DO stage will read *before* M1's audit runs. M6 should also restrict that use to sites
    already in the hard tier, so M1-SP loses no power to a gated arm worth ≈ +0.5 points (§1.2 item 21).
- **C2. M2 Test-S vs M7.** Test-S (reserved 11-10) forbids training or tuning any model that takes Sentinel-2 input
  or is fused with M2. Two M7 components train with S2-derived supervision:
  - the KD teacher's targets;
  - CIR-Lite-R's S2 L2A calibration targets (12,132 Secchi rows have an S2 chip).

  Proposed rule: exclude test-role-lake × 2015–2023 site-dates from both, and report the overlap counts. The M7
  handoff already notes that 545 Lite train sites lie in Orbit test-role areas.
- **C3. M3 H4 vs M5-X.** The lead already ruled that EA chemistry rows that are frozen M5-X test keys are excluded
  from H4 training. Also add a **counts-only overlap report** between the Thorpe WIMS sites and dates and the M5-X
  X-U test keys before H4 is frozen.
- **C4. M5 F-TOX labels vs M4 predictions.** F-TOX toxic units need toxicity values. Use **measured ECOTOX values
  only**, never M4 model predictions. Otherwise the M5 label depends on an M4 model, and a later Fuse claim would be
  circular.

### 3.4 Evidence that transfers between models (changes to designs)

1. **M5 → M3 (important).**
   - M3's D1 "NicheInvert" fits a Gaussian-logit response per unit and then maximises the likelihood over the
     environment. In ter Braak's terminology that is **classical calibration**, the bio.infer family.
   - In M5, bio.infer scored r2_anomaly **−1.08 / −1.14**, with over-dispersed slopes of 0.2–0.4 (M5 plan §1.2
     item 4, §2.1).
   - D1's mitigations are real: the context prior (tier-0 inside a Bayesian fusion), sequence-shrunk niches and
     tempering T. But M3's argument that "inversion degrades less under blocking" (Telford & Birks) is about
     **inverse** WA-type methods, not classical ML calibration.
   - **Required before the D1 gate:**
     - (i) an unregularised bio.infer-style ablation (no context prior, no shrinkage);
     - (ii) the M5 calibration diagnostic (`m5crit/geoniche_ceiling.py` logic: within-region slope and predicted/observed
       SD) run on D1's dev predictions;
     - (iii) WA-tol (inverse) kept as the published-family comparator.
   - I would lower P(D1 mean ≥ V4-XGB − 0.02) from M3's 0.45 to **≈ 0.35**.
2. **M5 → M3 (capacity).** The BioSet set model became the best *single* model only at ~180k rows (X-U). At 3k rows
   it lost to TabPFN and RF. This independently confirms M3's rejection of larger set or DNA-LM models at 1,215
   events.
3. **New this session → M4 T6.**
   - A DEBtox-derived model expresses LC50 explicitly as a function of time with three parameters: an asymptotic
     effect concentration, a time constant and an effect velocity (INERIS HAL record `ineris-00962827`; title and
     abstract found by search; authors and year **UNVERIFIED** because the page was blocked).
   - This is closer prior art for TED/TimeHead's time law than GUTS alone. It also gives a **strong, published,
     simple T6a comparator**: fit the 3-parameter law per (chemical, species) pair on its training durations, then
     extrapolate. 11,191 pairs carry ≥ 2 durations (M4 §7.2).
   - Also found: Posthuma, Price & Viljanen 2025 (ES&T 59(31):16250–16260, doi:10.1021/acs.est.5c01289) use duration
     as a 4-level one-hot over 24–96 h, and **do not evaluate unseen durations** (PMC full text fetched). That
     supports the novelty of T6 *as an evaluation*.
   - Add both to M4's novelty check, and the per-pair law to the T6a comparator list, **before** the M4 amendment is
     stamped.
4. **M6 → M1.**
   - M6's frozen forcing→water-temperature model never trained on the 729 reserved sites. Its prediction under
     *forecast* weather is an admissible feature for M1-N and M1-H, given to every M1 comparator (as PF-GBM's
     air2stream feature already is).
   - Expected ΔP is small (+1 to +3 on temperature).
   - The M1 amendment was stamped at 12:23 without it. Adding it now needs a new, separately stamped amendment
     before any M1-N result. **Recommendation: skip it this round.**
5. **Search notes (this session; novelty still owed via dated `docs/novelty_checks/` searches).**
   - M6's diel observation operator: one search found Zhi et al. 2023 (Nat. Clim. Change) naming the daytime
     grab-sample bias, but no model of it.
   - M2's multi-date lake context: one search found Prasad 2026 (single-date spatiotemporal turbidity transformer)
     and nothing with same-lake multi-date set context.

   Both remain **UNVERIFIED absences**.

### 3.5 Shared encoders and teachers: mostly *not* worth it this round (honest)

- **M2 → M7 teacher.**
  - The stock M2 teacher is banned for Lite KD (W8 leak).
  - The Lite-safe teacher's KD gain (~2.5 MdSA on val) did not replicate on Test-A (+1.6 [−1.7, +3.5]).
  - M7's own plan says do not invest further.
  - A Lite-safe retrain of M2 v2 (A1) is **not recommended** this round (ΔP ≈ 0). If it is ever done, it must be
    trained on rows excluding M7 evaluation blocks *and* M2 Test-S, frozen before any P2 result, and counted as a
    configuration.
- **Shared sequence encoder M1 ↔ M6.** Not recommended as shared *weights*:
  - M6 forbids in-situ inputs, while M1 is built on them;
  - the two splits differ (FINAL vs `sonde_split`), so cross-initialisation is a leakage hazard on M1's original
    track.

  Share *code* only: the diel parameterisation (amplitude, phase, skew; M6 D1) and the ordered DOmin ≤ DOmean ≤ DOmax
  heads (M1 SONDE-HF) are the same mathematics.
- **Chemical embeddings (M4 RX).** Cacheable for downstream ToxinWatch and M5 F-TOX descriptors. No claim value
  outside M4.

### 3.6 Multi-task targets shared across modalities

| Target family | Models | What to share |
|---|---|---|
| Turbidity / FNU / SSC | M1 (turbidity daily max forecast), M2 (FNU auxiliary, Track F), M6 (secondary), M7 (P2 primary; NTRU unit offset) | One unit-harmonised FNU/NTU/NTRU label table and one exclusion set |
| DO, DO %sat, hypoxia hours, P(DO-min < k) | M1, M3 (field DO %sat, GEO-8), M5 (UK DO saturation, BOD), M6 | One verified DO %sat function (X7). Regulatory thresholds from `docs/research/09` with URLs |
| Chloride / SC exceedance (230 / 860 mg/L) | M1 (SC max, P(Cl > 230)), M3 (Cl, GEO-8), M5 (chloride; F-E thresholds), M6 (SC secondary) | One threshold registry (M5 marks its thresholds UNVERIFIED) |
| TP / TN / DOC | M2 Track N (S2), M3 (16S), M5 (benthos) | Not a SOTA lever. A **cross-modal comparison is a paper result** for Fuse, on whatever rows are legitimately co-located |
| GPP / ER / K600 | M1 PhiDA-Lite, M6 D2-DO, Sim | One Powell/StreamPULSE role table (§3.3) |
| Water temperature 7-DADMax | M1 (forecast), M6 (estimation) | One definition (trailing 7-day mean of daily max) |

**Guard review.** M2, M4 and M7 each extend single-look guards (`testlook.py`, `unseen_taxa_guard`, `xtestlook`).
Proposal: **one independent code review in W3** covering every new guard extension, against the rule-7 checklist.

---

## 4. What must be pre-registered, what needs approval, what needs the user

### 4.1 Pre-registration (new files only; lead-stamped before any validation number on the item)

| Model | File | Must contain | Deadline |
|---|---|---|---|
| M1 | `sonde_arch_v2_amendment.md` + `sonde_arch_v2_lead_rulings.md` | **Stamped 12:23 EST.** Rulings: R1 E2-B composition; R2 no upstream inputs; R3 GEFS c00 pilot; R4 M1-H in the M1-N look; R5 EW-H1h descriptive unless ≥ 50 onsets. **Still owed:** the git commit and public push before any freeze, and the GEFS p01–p04 ruling if members are wanted | Done. Commit before the freeze |
| M2 | protocol §11.7–11.9 (Appendix A of the M2 plan) | Test-S: scene-disjointness count rule (≥ 60 % of rows and ≥ 80 % of blocks), small-lake MDE rule, scoping. Track N: censoring, trophic-chain baseline, MDE > 15 → descriptive. Arms A0–A4 with caps, the **seed-unit rule with comparator parity**, kill and gate rules, Holm over v1/v2. §2.1 FNU amendment. dev_v5m. Chl-a method-class covariate decision. `testlook` extension + review | **Before the headline M2/test look** (~11-12 to 11-13) |
| M3 | `microbe_geo_protocol.md` + §10.9; reserve #6–#8; M3E amendment; `microbe_ea_ids.json`; GAS-3 and qPCR amendments | GEO-8 in the **same sealed look** as H-M3, cap ≤ 10, full parity list. **Add:** the D1 classical-calibration ablation and slope diagnostic (§3.4 item 1) and the Thorpe/M5-X overlap count (C3) | GEO before the H-M3 freeze (~11-20). M3E before any EA label download |
| M4 | `chemtox_arch_v2_amendment.md` | Plan §8.3 items 1–8. **Add:** the per-pair DEBtox-style LC50(t) T6a comparator; T6 duration bands and the (cluster × duration-band) resampling unit with MDE; and the novelty-check entries for Liang 2026 (Crossref record **re-verified by me today**), Wu 2026, Zhu 2026, Posthuma 2025 and the INERIS record | **Before any embedding run touches T1**, and before any T1/T6 fit |
| M5 | M5-F protocol v0 | Plan §8 items 1–8. Co-primary pooled R² and r2_anomaly fixed now. TabPFN v2 only (lead). Exceedance thresholds with official URLs. C4 (measured toxicity values only) | Before any French download |
| M6 | `hydromet_arch_v2_amendment.md` (drafted by M6, stamped by the lead) | Lead rulings of 12:26: T-E (zero-shot + fine-tuned) with the minimum group count for confirmatory endpoints; T-H inside T-E/T-B; the T-B/N2 joint look; T-D descriptive only. Arms D1/D2/D3 + control; R-TD, G1, G2 with **mandatory** grab-sample parity; cap 12; gates G-D1a/G-D1b/G-DO; claim-model rule; role-assignment sha. **Add:** the StreamPULSE registration of C1 | Before any val number on the new arms (plan steps 1–2 only until then) |
| M7 | N8 to `lite_national_protocol.md`; new `lite_p2_protocol.md` | National-2b labels; C2/C2-AFT/C3/C4b; slots 7/8; the §8a label wording stays (the lead ruled out a national target-scoped label). P2: salted 40 % split before chip extraction; Holm {FNU} × {simple, published}; cap 8. **C2 rule** | N8 before the national freeze (~11-18 to 11-20). P2 before any chip extraction |

**Cross-workstream items** (lead; proposed for a short dated addendum to `sota_campaign.md`):

| ID | Item |
|---|---|
| X1 | StreamPULSE: M6 registers its label reads before M1's exposure audit, and restricts them to hard-tier sites (C1) |
| X2 | Test-S exclusion for the M7 teacher and CIR-Lite-R (C2) |
| X3 | Powell role table honouring M1, M6 and Sim H-G3 |
| X4 | Shared sonde FNU/hourly role table with the union of exclusions |
| X5 | Single Naïades pull with role partition; Fuse never combines M5-F and M7-P3 on overlapping rows |
| X6 | **One project-wide TabPFN policy**: v2 2.2.1 installed; it is a *mandatory* strongest-simple comparator wherever n ≤ 10k rows (M3 V4-CLR, M5 F, M7 C2-TPF if installed), decided before results |
| X7 | Verified physics-constants module |
| X8 | Matched-control rule for every SENTINEL ensemble or stack |

### 4.2 Downloads needing size or licence approval

| Source | Model | Kept size / transfer | Licence status | Approval |
|---|---|---|---|---|
| NOAA GEFSv12 reforecast (AWS `noaa-gefs-retrospective`) | M1 | < 1 GB kept; ≈ 0.6 TB streamed (c00; tmax/tmin/apcp/dswrf; leads 1–7) | NOAA open data ([F] in M1 plan) | **Approved (R3)**: pilot first; stop if kept > 1 GB, > 72 h, or the correctness check fails; ≤ 5 GB transient |
| NOAA GEFS operational (`noaa-gefs-pds`) | M1 | < 0.5 GB kept; archive start [U] | NODD [U] | Lead |
| StreamPULSE bulk | M1 | ≈ 0.5 GB | ODC-By 1.0, 1-yr embargo [R01] | Lead, after C1 |
| Thorpe et al. Zenodo (10.5281/zenodo.17515762) | M3 | 84.6 MB | CC BY 4.0 (Zenodo API per M3) | Approved (M3E); ID check first |
| ENA PRJEB90117 | M3 | < 0.3 GB processed | INSDC open | Approved |
| NEON DP1.20097.001 dissolved gases | M3 | tens–hundreds of MB [U] | CC BY 4.0 | Lead, after the GAS-3 amendment |
| MiCoDa | M3 | [U] | CC BY 4.0 | Only if cycle-6 #4 > #3 |
| TRIDENT deposit (10.5281/zenodo.10260621), 2 xlsx files only | M4 | ≈ 36 MB | CC BY 4.0 **[U]** | Lead, after licence verification |
| EnviroTox; EPA Web-ICE v4.0 | M4 | [U] / tens of MB [U] | [U] | Not yet (T5 audit) |
| Naïades / Hub'eau biology + physico-chemistry | M5 (+ M7 P3) | 0.2–0.8 GB est. (cap 5 GB) | Etalab Licence Ouverte (page fetched by M5) | Approved (M5-F) |
| TabPFN-2.5+ weights | M5 (others) | — | **Non-commercial + token** | **User** (lead: not used) |
| Oliver et al. 2024 compilation (doi:10.5066/P9EMWZ35) | M6 | **[U], HEAD first** | CC0 (M6 [S][F]) | Approved 12:26 (stream-extract-delete; protected groups locked at extraction) |
| EA Hydrology API sondes | M6 (M1) | within M6's 20 GB cap | OGL v3.0 | Approved 11-09 |
| ERA5-Land via Open-Meteo | M6 | small; shared daily quota | terms [U] | Approved: size the quota cost first; **≤ 40 % of the daily budget while M1's forecast pull runs** |
| HydroATLAS subsets | M6 | [U] | CC BY 4.0 | Conditional |
| S2 L1C centre windows; lake-context bank phase B; AlphaEarth; USGS FNU chips | M2 | 0.9 / 0.55 / < 0.2 / 0.1–1 GB | ESA notice (access cost [U]) / as S2 / CC BY 4.0 [R02] / PD | Lead (gated) |
| Track G (EEA, SYKE, SLU MVM, Ontario, ECCC) | M2 | ≤ 8 GB | **licences [U]** | Not funded yet |
| NWIS FNU chips; WQP tube/DOC/FNU chips; BD ORTHO; NLA rings | M7 | 0.1–0.2 / 0.1–0.3 / 0.2–0.4 / 0.25–1 GB | PD / PD / **[U] (band layout too)** / [U] | P2 approved; P3/P4 conditional |

**Kept-disk total.** If everything were approved: ≈ 10–35 GB, against ~123 GB of headroom above the floor. The
GEFS *transfer* (≈ 0.6 TB streamed) is the only large-network item.

### 4.3 Decisions only the user can make

- **U1. GPU regime.** The computed task says GPUs 0–1. The lead decision and lock file say 0–9 (per the user's
  instruction "you can use all of the other gpus"). Confirm. The plan works on 0–1; 0–9 cuts the wall-clock time,
  not the risk.
- **U2. TabPFN-2.5+ licence.** Non-commercial terms plus a token. Agents must not handle credentials. If declined,
  record "not compared: licence" in every affected protocol. A reviewer may still raise it against M5-X and M5-F.
- **U3. GEFS bulk transfer (for information).** The lead approved it under R3: ≈ 0.6 TB streamed on a shared host,
  pilot first, automatic stop conditions. The user should know a transfer of this size is running.
- **U4. Package installs** (for example TabPFN for M7's C2-TPF comparator). Lead or infra decide, never an agent in
  this step.
- **U5. Scope extension.** M7 P4 (lake habitat is not water quality). M6 T-D is already ruled descriptive-only.
- **U6. Success framing.**
  - H-G1 needs ≥ 5 of 7 models with SOTA or first-in-class on their *pre-registered primary* endpoints. Several of
    those endpoints are spent or negative (M4 ADORE, M6 primary, M7 Test-A).
  - The new tracks give *scoped* labels, which do not satisfy H-G1 as written. Re-defining H-G1 would lower a bar,
    which the rules forbid.
  - The honest choice: report H-G1 as **not met** unless it is met, and present the scoped labels and the
    pre-registered negatives as the contribution.
- **Multiplicity disclosure.** The round adds ≈ 20–25 new confirmatory families across the project. Holm is
  within-family only. The IUT and every-seed rules keep each family's false-label rate well below 5 %, but expect
  ≈ 0.2–0.5 spurious scoped labels project-wide if every null were true. The paper should report the total number of
  tracks and families next to any label.

---

## 5. Honest summary

### 5.1 Per-model table

| Model | Current status (from the plans) | Best proposed architecture | New targets | Harder track (primary bet) | P(≥ 1 scoped §15.1 label this round) |
|---|---|---|---|---|---|
| **M1 Sonde** | No label. Val: ens3w is 3–12 % CRPS over damped anomaly; a baseline-only ensemble is within ±0.01; the bar is met only for SC on the val-role cell. EW-v2 Test-A underpowered | **SONDE-HF** (Nearing-style hindcast→forecast hand-off + hourly branch + masked-mean NWP/GEFS members + ordered 19-quantile multi-task heads), with E2/E2-B; PhiDA-Lite gated | Tmax / 7-DADMax, Tmin, DO max and amplitude, DO %sat, hypoxia hours, SC max, turbidity max; aux log Q / gage; GPP/ER/K600 | **M1-H** hourly DO / hypoxia hours (0.20–0.25); M1-N-F with GEFS (0.10) | **≈ 0.30 (0.20–0.40)**, most likely SC on M1-N or M1-H DO |
| **M2 Orbit** | Competitive on val (48.3 vs MDN_B 52.6; 2 of 3 seeds fail per-seed). Headline v1 look imminent (joint power 0.21) | **A1 SatWQ-LakeSet-MT** (same-lake multi-date set context + land ring + multi-target), with the A0 control | TP, TN, DOC (Track N); FNU/SSC auxiliary; Carlson TSI descriptive | **Test-S** (test-role lakes × 2015–23; ≈ 2× blocks) | **≈ 0.40–0.50**, most likely Secchi on Test-S; chl-a ≈ 0.25–0.35 |
| **M3 Microbe** | Pending. Dev LODO: 0 targets beat the strongest simple baseline; 1 (SC) on the H-M3 analogue. Effective n ≈ 32 sites | **D1 NicheInvert** (CPU; multivariate calibration with sequence-shrunk niches and a context prior), *with the §3.4 classical-calibration safeguards*; D3 V4-MR; D2 only if gated | GEO-8 (lab pH, Cl, SO4, Ca, DIC, DO %sat, NH4, UV254); GAS-3 (CO2, CH4, N2O) | **H3 EA NEON→England** multi-determinand, Spearman (0.25–0.30) | **≈ 0.25 by early Dec; ≈ 0.35 by Feb 2027** |
| **M4 ChemTox** | Competitive (ADORE spent; prospective block spent). T2 in flight, with a channel-matched comparator leading selection | **ChemTox-TED** (monotone duration law + non-negative ACR + Tobit), preceded by **RX** embeddings and **TimeHead**; MMoE as comparator | Chronic endpoints (225,081 rows), censored acute (11,719), ACR (10,334 pairs; reported, not a §15.1 label), σ / intervals | **T6a duration extrapolation** (0.30) and **T1 ECFP4 clusters** (0.25) | **≈ 0.35–0.40**. Most likely single outcome (~40 %): the references (R1–R3) win |
| **M5 BioAssess** | **SOTA already** (ensemble C, Test A: chloride, pH, conductivity). M5-X freeze candidate: dev predicts SOTA on 2 of 12 | **BioSet-MA** (invertebrates + diatoms, multi-target Student-t, P3 region token) inside an ensemble; BioPFN | Organic pollution (DO sat, BOD5, nitrite); prior-365-day statistics; diatom-informed nutrients; exceedances; F-TOX exploratory | **France F-A / F-Z / F-T** (F-A ensemble 0.35–0.45) | **Already met.** New label: M5-X ≈ 0.45; France ≈ 0.35–0.45; BioSet-MA alone ≈ 0.10 |
| **M6 HydroMet** | Competitive (0 of 28 won on `M6/test`). **Negative** on `M6N` temperature (+0.161 °C RMSE) | **D1 GSO-MT** (learned diel observation operator over ~20k grab-sample stations) and **D2 ThermOx-T** (energy-balance core); D3 stack with control | Daily T min/max, DO max, log Q (auxiliary), instantaneous T/DO at sample hour, GPP/ER (gated) | **T-E EA England zero-shot** (0.15) | **≈ 0.18 at the EA look (5–7 weeks, i.e. after this round); ≈ 0.25 by Test-B (Q3 2027)** |
| **M7 Lite** | Negative on 4-state Test-A. National val: Secchi 12.4 behind GBM + context; turbidity 25.7 better than GBM-RGBN | **ANCHOR-Lite** (cross-fitted GBM-parity anchor + bounded CNN residual, context-complete), with WaterSet-Lite as the alternative branch | FNU, NTRU, SSC, censored rows (0 GB); NWIS FNU; transparency tube; DOC (auxiliary) | **P2 rivers/sondes FNU** at new water bodies (0.30, 0.15–0.45) | **≈ 0.25–0.30** (national scoped labels ruled out). Secchi ≈ 0.10 |

### 5.2 Portfolio arithmetic (independence assumed; correlated failure, e.g. host contention, fattens the tails)

New scoped labels among M1, M2, M3, M4, M6 and M7. M5 already holds one.

| Scenario | Inputs | E[new] | P(0) | P(≥ 1) | P(≥ 2) | P(≥ 3) | P(≥ 4) |
|---|---|---|---|---|---|---|---|
| Plan numbers ("mid") | 0.30, 0.45, 0.25, 0.375, 0.18, 0.30 | 1.85 | 0.10 | 0.90 | 0.60 | 0.27 | 0.07 |
| Discounted ("low") | 0.20, 0.40, 0.15, 0.30, 0.12, 0.20 | 1.37 | 0.20 | 0.80 | 0.42 | 0.13 | 0.02 |
| Optimistic ("high") | 0.40, 0.50, 0.35, 0.40, 0.25, 0.40 | 2.30 | 0.05 | 0.95 | 0.74 | 0.42 | 0.15 |

So **≥ 5 of 7 models with any scoped label is ≈ 0.07 (0.02–0.15)**. H-G1 (primary endpoints) is lower still.

**What would move these numbers:**
- *Up:* GEFS approved; M1's step-1 test > 0.02; Test-S scene-disjoint subset ≥ 60 % of rows; EA group count ≥ 100;
  P2 counts at the high end.
- *Down:* R-NH alone matches SONDE-HF; the A1 c3 ablation ≈ c1; R1–R3 close M4's gap; GBM-parity ≫ GBM + context for
  M7; D1 shows bio.infer-like over-dispersion.

### 5.3 What not to do (cross-model)

- Do not add capacity (bigger backbones, Mamba/xLSTM/ViT swaps, fine-tuned EO or DNA foundation cores). Every plan's
  evidence is against it.
- Do not credit ensembling or weighting to SENTINEL without a matched baseline-only control (X8).
- Do not choose strata, targets or tracks from val wins. Specifically: M2 low-chl / high-AOT, M7 river class and glint,
  M6 outcome-defined P90, and a national M7 turbidity label.
- Do not reuse any spent block: M4 ADORE and prospective; M7 Test-A; M1 EW-v2 Test-A; M6 `M6/test` and `M6N`;
  M5 NRSA 2018–19 / NLA 2022 / UK 2021–26.
- Do not let any queued freeze or look slip for v2 work (lead rule 4): M5-X, the M2 headline, M4 T2, the M7 national
  freeze.

---

## Sources (searched or fetched by me this session)

- Liang Y., Cui B., Li F., Zhou D. (2026-11-02). Context-Aware Multitask Learning for Aquatic and Soil Ecotoxicity
  Prediction from Fragmented ECOTOX Records. *Environ. Sci. Technol.* doi:10.1021/acs.est.6c12447. Crossref record
  fetched: https://api.crossref.org/works/10.1021/acs.est.6c12447 (metadata only; results not read).
- Posthuma L., Price T., Viljanen M. (2025). Improving the Ecotoxicological Hazard Assessment of Chemicals by Pairwise
  Learning. *Environ. Sci. Technol.* 59(31):16250–16260. doi:10.1021/acs.est.5c01289. Found by search; PMC full text
  fetched: https://pmc.ncbi.nlm.nih.gov/articles/PMC12355940
- "Modeling explicitly and mechanistically median lethal concentration as a function of time for risk assessment".
  INERIS HAL record: https://hal-ineris.archives-ouvertes.fr/ineris-00962827. Title and abstract found by search;
  **authors, year and venue UNVERIFIED** (the page blocked automated access).
- Zhi W. et al. (2023). Widespread deoxygenation in warming rivers. *Nat. Clim. Change*.
  https://www.nature.com/articles/s41558-023-01793-3 (found by search; daytime grab-sample bias).
- Prasad A., Verma S., Khan M.A. (2026). A Spatiotemporal Transformer for Satellite Retrieval of Lake Turbidity.
  NeurIPS 2026 CCAI workshop. https://neurips.cc/virtual/2026/174697 (found by search).
- Every other reference above is carried from the named plan section with that plan's verification tag. It was not
  re-verified by me.
