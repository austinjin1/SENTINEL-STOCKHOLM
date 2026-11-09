# Datasheet — emp_freshwater (EMP 16S Release 1, freshwater subset)

- **Source:** Earth Microbiome Project Release 1, ftp://ftp.microbio.me/emp/release1 (mapping `emp_qiime_mapping_release1.tsv`,
  table `emp_deblur_150bp.release1.biom`). Retrieved 2026-11-08.
- **Licence / terms:** public research data. The FTP site states no explicit licence; the EMP asks users to cite
  Thompson et al. 2017, *Nature* 551:457.
- **Use:** unlabeled community data for SENTINEL-Microbe pretraining **only**. There are no water-quality labels,
  and no EMP sample enters a WQ claim, a CV fold or a test set.
- **Content:** samples with `empo_3` = "Water (non-saline)". Deblur exact sequences of 150 bp, forward reads from
  515F (primer removed); `seq_id` = sha1(sequence). These match the first 150 bp of the SENTINEL V4 window.
- **Coverage:** 4723 samples, 7 studies, 74901 sequences,
  1800921 (sample, sequence) rows; median reads per sample 7336.
- **QC:** as released (EMP Deblur QC). No further filtering at ingest.
- **Known issues:** the studies differ in protocol details, sequencing depth and geography. Mostly read length 150
  (Deblur trims to exactly 150 bp).
- **Rebuild:** `python -B scripts/data/download_emp.py`
