# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
# M5-X protocol §6.3 fidelity check for the EPA bio.infer package (W-MX-5): re-run the package's own documented
# example pipeline on its bundled Oregon / EMAP-West data, headless (Tk unavailable), and compare every stage with
# the package's bundled intermediate objects. No SENTINEL data or code path is involved.
#   Rscript scripts/eval/bioassess_x_bioinfer_fidelity.R <out_dir>
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "."
suppressMessages(library(bio.infer))
res <- list(version = as.character(packageVersion("bio.infer")), R = R.version.string)
data(coef.west.wt); data(bcnt.tax.OR); data(bcnt.otu.OR); data(ss.OR); data(envdata.OR)
res$coef_xvar <- coef.west.wt$xvar
res$coef_ntaxa <- length(coef.west.wt$tnames)
# 1. documented mlsolve example
inf <- mlsolve(ss.OR, coef.west.wt)
res$mlsolve_n <- nrow(inf); res$mlsolve_cols <- names(inf)
res$mlsolve_consistent_frac <- if ("Inconsistent" %in% names(inf)) mean(!inf$Inconsistent) else NA
# 2. makess on the bundled OTU table reproduces the bundled site-species matrix
ss2 <- makess(bcnt.otu.OR)
same_cols <- setequal(names(ss2), names(ss.OR))
res$makess_same_columns <- same_cols
if (same_cols) {
  a <- ss2[order(ss2[, 1]), names(ss.OR)]; b <- ss.OR[order(ss.OR[, 1]), ]
  res$makess_max_abs_diff <- max(abs(as.matrix(a[, -1]) - as.matrix(b[, -1])))
}
# 3. inferred vs observed on the Oregon samples (documented behaviour: inferences track observed conditions)
m <- merge(inf, envdata.OR, by.x = "SVN", by.y = "STRM.ID", suffixes = c(".inf", ".obs"))
res$n_matched <- nrow(m)
res$r_sed <- cor(m[["sed.inf"]], m[["sed.obs"]], use = "complete.obs")          # % sand/fines
res$r_temp <- cor(m[["STRMTEMP"]], m[["temp"]], use = "complete.obs")          # stream temperature
# 4. documented taxon.env example runs
ce <- taxon.env(~sed + sed^2, bcnt.tax.OR, envdata.OR, "SVN", "CountValue", "STRM.ID")
res$taxon_env_ntaxa <- length(ce$tnames); res$taxon_env_xvar <- ce$xvar
writeLines(capture.output(str(res)), file.path(out, "bioinfer_fidelity.txt"))
write.csv(inf, file.path(out, "bioinfer_example_mlsolve.csv"), row.names = FALSE)
print(res)
