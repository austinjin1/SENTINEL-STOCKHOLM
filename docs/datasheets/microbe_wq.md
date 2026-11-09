# Datasheet — microbe_wq (NEON 16S events × co-located water quality)

- **Built:** 2026-10-25 by `scripts/data/build_microbe_wq.py` from processed NEON datasets (manifests
  `neon_microbe_16s`, `neon_chem`, `neon_sonde_daily`, `neon_sites`). License CC BY 4.0 (NEON;
  attribution: NSF NEON, RELEASE-2026, https://data.neonscience.org).
- **Unit:** one row per community event = site × local collection date × primer set (16S
  dnaSampleIDs summed). `microbe_wq` = DP1.20141.002 (RELEASE-2026 OTUs); `microbe_wq_v1` =
  DP1.20141.001 (superseded 2014–2018 ASV pipeline; cross-release scheme only).
- **Files:** `samples.parquet` (event metadata, primer set, reads, context, raw matched targets
  `raw_*` + match diagnostics `chem_match`, `chem_dt_h`, `sonde_hor/ver`, `*_pm1d` ablation),
  `tokens.parquet` (sample_id, seq_id = sha1(sequence), count, RDP ranks), `sequences.parquet`.
- **Join rules:** docs/preregistration/microbe_protocol.md §2.2 (chemistry: same parentSampleID,
  else ±24 h at the same site; sonde: daily mean on the local-standard-time date at HOR 102 for
  wadeable streams / HOR 103 for lakes and rivers, shallowest VER).
- **Units:** TN/TP/DOC/NO3+NO2-N mg/L (lab, `milligramsPerLiter` only); SC µS/cm; DO mg/L;
  turbidity FNU; chlorophyll-a µg/L (never RFU).
- **QC:** provisional records dropped; events with < min_reads reads dropped; censored chemistry
  keeps the published value (else the detection limit), flagged `*_censored`.
- **Known issues:** primer set confounded with time; OTU ids release-specific (sequences used
  instead); COVID-19 sampling gaps 2020–2022; lab changes for chemistry.
- **Upstream data issue (workaround in place, fix requested from the NEON data workstream):**
  `neon_chem/chem_event_wide` has one row per *bottle*, keyed by the bottle's parentSampleID with
  fraction/replicate suffixes (`*.RAW.NUT`, `*.FIL.NUT`, `.2`/`.3`); from 2019 on TN/TP/NO3 appear only
  on those suffixed rows, not on the bare field-visit id that the 16S `parentSampleID` points to. A naive
  id join silently loses most post-2019 nutrient targets. Workaround: `sentinel.data.microbe_wq.visit_key`
  strips the suffixes and `collapse_visits` takes the mean over the visit's bottles that report each
  analyte (censored = any) before the parentSampleID match. Verified 2026-11-08: a rebuild against the
  2026-10-25 07:07 UTC upstream refresh is identical to the current files.
- **Rebuild:** `python -B scripts/data/build_microbe_wq.py`

## Coverage

### `microbe_wq`

```json
{
 "n_dna_samples": 1494,
 "n_after_provisional": 1494,
 "count_rows_without_sequence": 0,
 "primer_source": {
  "pcr_record": 1494
 },
 "mixed_primer_events": 0,
 "length_rule_agreement_with_pcr_records": {
  "n": 1494,
  "agree": 0.999330655957162
 },
 "n_events_all": 1432,
 "n_events_min_reads": 1429,
 "n_events": 1429,
 "n_tokens": 4252308,
 "n_sequences": 76081,
 "sites": 33,
 "domains": 19,
 "first": "2018-08-14",
 "last": "2024-11-12",
 "primer_sets": {
  "V4V5": 1429
 },
 "chem_match": {
  "parentSampleID": 1403,
  "time": 26
 },
 "chem_joined": true,
 "target_non_missing": {
  "raw_tn_mgl": 1394,
  "raw_tp_mgl": 1396,
  "raw_doc_mgl": 1358,
  "raw_no3_mgl": 1392,
  "raw_sc_uscm": 863,
  "raw_turb_fnu": 785,
  "raw_do_mgl": 812,
  "raw_chla_ugl": 419
 },
 "events_per_year": {
  "2018": 45,
  "2019": 250,
  "2020": 148,
  "2021": 216,
  "2022": 275,
  "2023": 281,
  "2024": 214
 },
 "sonde_joined": true,
 "min_reads": 1000,
 "retrieved_inputs": {
  "neon_microbe_16s": "2026-10-25T03:25:13+00:00",
  "neon_chem": "2026-10-25T03:25:12+00:00",
  "neon_sonde_daily": "2026-10-25T05:14:45+00:00",
  "neon_sites": "2026-10-25T01:54:59+00:00"
 }
}
```

### `microbe_wq_v1`

```json
{
 "n_dna_samples": 942,
 "n_after_provisional": 942,
 "count_rows_without_sequence": 0,
 "primer_source": {
  "pcr_record": 940,
  "amplicon_length": 2
 },
 "mixed_primer_events": 0,
 "length_rule_agreement_with_pcr_records": {
  "n": 940,
  "agree": 1.0
 },
 "n_events_all": 714,
 "n_events_min_reads": 707,
 "n_events": 707,
 "n_tokens": 417242,
 "n_sequences": 174007,
 "sites": 33,
 "domains": 19,
 "first": "2014-08-05",
 "last": "2018-10-30",
 "primer_sets": {
  "V3V4": 707
 },
 "chem_match": {
  "parentSampleID": 575,
  "none": 131,
  "time": 1
 },
 "chem_joined": true,
 "target_non_missing": {
  "raw_tn_mgl": 552,
  "raw_tp_mgl": 563,
  "raw_doc_mgl": 552,
  "raw_no3_mgl": 558,
  "raw_sc_uscm": 165,
  "raw_turb_fnu": 98,
  "raw_do_mgl": 151,
  "raw_chla_ugl": 148
 },
 "events_per_year": {
  "2014": 10,
  "2015": 48,
  "2016": 132,
  "2017": 271,
  "2018": 246
 },
 "sonde_joined": true,
 "min_reads": 1000,
 "retrieved_inputs": {
  "neon_microbe_16s": "2026-10-25T03:25:13+00:00",
  "neon_chem": "2026-10-25T03:25:12+00:00",
  "neon_sonde_daily": "2026-10-25T05:14:45+00:00",
  "neon_sites": "2026-10-25T01:54:59+00:00"
 }
}
```

