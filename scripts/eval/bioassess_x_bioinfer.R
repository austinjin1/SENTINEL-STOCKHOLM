# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
# M5-X comparator §6.3: the EPA bio.infer package (CRAN 1.3.6) run itself, per fold (amendment v4.7, v5.1).
#   Rscript scripts/eval/bioassess_x_bioinfer.R taxonomy <names.csv> <out.rds>
#   Rscript scripts/eval/bioassess_x_bioinfer.R equivalence <out_dir> [<fold_dir>]
#   Rscript scripts/eval/bioassess_x_bioinfer.R fold <fold_dir> <taxonomy.rds> <out.csv>
# Inputs are written by scripts/eval/bioassess_x_dev.py export-r (training rows passed assert_fit_ids there; the
# pseudo-test / test token files carry no label). Pipeline: get.taxonomic (once, names only) -> per fold
# taxon.env(~x + x^2) on the fold's same-continent labelled training rows -> pseudo-test rows get.otu(coef) ->
# makess -> mlsolve.
# Headless defaults: the two interactive prompts of get.taxonomic are answered by fixed defaults (no manual name
# correction; first listed ITIS candidate). taxon.env.fast = the package's taxon.env with two loops vectorised
# (site x taxon abundance matrix; ROC area); "equivalence" verifies identical tnames/coefficients/xlims/roc.
suppressMessages(library(bio.infer))
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]

headless_defaults <- function() {
  assignInNamespace("correct.taxanames", function(tname.old, get.tax.env) tname.old, "bio.infer")
  assignInNamespace("get.dupe.sel", function(sumstr) {
    w <- regexpr("-", sumstr); taxon <- substring(sumstr, 1, w - 1)
    keep <- taxon != "" & !duplicated(taxon)
    sumstr[keep]
  }, "bio.infer")
}

taxon.env.fast <- function(form, bcnt, envdata, bcnt.siteid, bcnt.abndid, env.siteid, tlevs = "all") {
  # identical to bio.infer::taxon.env (1.3.6) except: (1) ss1 built with rowsum instead of a per-site match loop;
  # (2) ROC area = sum_{present} #(absent < present) / (nx * ny) via findInterval instead of an nx x ny matrix.
  formstr <- as.character(form)[2]
  split0 <- trimws(strsplit(formstr, "\\+")[[1]])
  w <- regexpr("\\^", split0); w2 <- regexpr("\\*", split0)
  xvar <- character(0)
  for (i in seq_along(split0)) {
    if (w[i] != -1) { xvar <- c(xvar, substring(split0[i], 1, w[i] - 1)); split0[i] <- paste("I(", split0[i], ")", sep = "") }
    else if (w2[i] != -1) { xvar <- c(xvar, substring(split0[i], 1, w2[i] - 1), substring(split0[i], w2[i] + 1, nchar(split0[i]))) }
    else xvar <- c(xvar, split0[i])
  }
  xvar <- unique(trimws(xvar))
  dfenv <- na.omit(envdata[, c(env.siteid, xvar)])
  form0 <- paste("resp ~", paste(split0, collapse = "+"), sep = "")
  cutoff <- (length(split0) + 1) * 10
  tlevs.sel <- if (tlevs[1] == "all") names(bcnt)[4:length(names(bcnt))] else tlevs
  bcnt0 <- merge(bcnt, dfenv, by.x = bcnt.siteid, by.y = env.siteid)
  xlims <- as.list(rep(NA, times = length(xvar)))
  for (i in seq_along(xvar)) {
    r0 <- range(bcnt0[, xvar[i]]); xlims[[i]] <- r0
    dfenv[, xvar[i]] <- (dfenv[, xvar[i]] - r0[1]) / diff(r0)
  }
  site <- bcnt0[, bcnt.siteid]
  if (is.factor(site)) site <- levels(site)[site]
  sitenames.u <- sort(unique(site))
  nsamp <- length(sitenames.u)
  sid <- match(site, sitenames.u)
  tnames.sav <- character(0); roc <- numeric(0); tcount <- 0; csave <- NULL
  for (i in seq_along(tlevs.sel)) {
    lev <- bcnt0[, tlevs.sel[i]]
    numocc <- tapply(site, lev, function(x) length(unique(x)))
    numocc3 <- pmin(numocc, nsamp - numocc)
    tnames.loc <- names(numocc3)[numocc3 >= cutoff]
    if (length(tnames.loc) == 0) next
    ntaxa <- length(tnames.loc)
    keep <- !is.na(lev) & lev %in% tnames.loc
    key <- sid[keep] + (match(lev[keep], tnames.loc) - 1) * nsamp
    s <- rowsum(bcnt0[keep, bcnt.abndid], key)
    ss1 <- matrix(0, ncol = ntaxa, nrow = nsamp)
    ss1[as.numeric(rownames(s))] <- s[, 1]
    ss <- data.frame(sitenames.u, ss1, stringsAsFactors = FALSE)
    names(ss) <- c("SITEID", tnames.loc)
    df1 <- merge(ss, dfenv, by.x = "SITEID", by.y = env.siteid)
    for (j in 1:ntaxa) {
      resp <- df1[, tnames.loc[j]] > 0
      resp[is.na(resp)] <- FALSE
      mod1 <- suppressWarnings(glm(as.formula(form0), data = df1, family = "binomial"))
      tcount <- tcount + 1
      predout <- predict(mod1, type = "response")
      x <- predout[resp]; y <- sort(predout[!resp])
      roc <- c(roc, round(sum(findInterval(x, y, left.open = TRUE)) / (length(x) * length(y)), digits = 3))
      csave <- if (tcount == 1) as.vector(coef(mod1)) else rbind(csave, as.vector(coef(mod1)))
    }
    tnames.sav <- c(tnames.sav, tnames.loc)
  }
  list(tnames = tnames.sav, csave = csave, xvar = xvar, xlims = xlims, form = form, roc = roc)
}

if (mode == "taxonomy") {
  headless_defaults()
  nm <- read.csv(args[2], stringsAsFactors = FALSE)$name
  nm <- sort(unique(nm[!is.na(nm) & nm != ""]))
  bcnt <- data.frame(SVN = "all", Taxon = nm, CountValue = 1, stringsAsFactors = FALSE)
  owd <- setwd(dirname(args[3]))
  tax <- get.taxonomic(bcnt)
  setwd(owd)
  tax$SVN <- NULL; tax$CountValue <- NULL
  saveRDS(tax, args[3])
  cat("names", length(nm), "matched to ITIS (any level):", sum(!is.na(tax$PHYLUM)), "of", nrow(tax), "\n")
}

if (mode == "equivalence") {
  data(bcnt.tax.OR); data(envdata.OR)
  a <- taxon.env(~sed + sed^2, bcnt.tax.OR, envdata.OR, "SVN", "CountValue", "STRM.ID")
  b <- taxon.env.fast(~sed + sed^2, bcnt.tax.OR, envdata.OR, "SVN", "CountValue", "STRM.ID")
  res <- list(oregon_tnames = identical(a$tnames, b$tnames), oregon_coef_maxdiff = max(abs(a$csave - b$csave)),
              oregon_roc_identical = identical(a$roc, b$roc), oregon_xlims = identical(a$xlims, b$xlims))
  if (length(args) >= 3) {   # one development fold (training rows only), first target
    tax <- readRDS(args[3])
    tok <- read.csv(file.path(args[4], "train_tokens.csv"), stringsAsFactors = FALSE)
    y <- read.csv(file.path(args[4], "train_y.csv"), stringsAsFactors = FALSE)
    bt <- merge(data.frame(visit_id = tok$visit_id, Taxon = tok$name, count = tok$count, stringsAsFactors = FALSE),
                tax, by = "Taxon")[, c("visit_id", "Taxon", "count", setdiff(names(tax), "Taxon"))]
    v <- names(y)[2]
    f <- as.formula(paste("~", v, "+", v, "^2"))
    a2 <- taxon.env(f, bt, y, "visit_id", "count", "visit_id")
    b2 <- taxon.env.fast(f, bt, y, "visit_id", "count", "visit_id")
    res$fold_target <- v
    res$fold_tnames <- identical(a2$tnames, b2$tnames)
    res$fold_coef_maxdiff <- max(abs(a2$csave - b2$csave))
    res$fold_roc_identical <- identical(a2$roc, b2$roc)
  }
  print(res)
  writeLines(capture.output(str(res)), file.path(args[2], "bioinfer_taxon_env_fast_equivalence.txt"))
}

if (mode == "fold") {
  d <- args[2]; tax <- readRDS(args[3]); out <- args[4]
  tok <- read.csv(file.path(d, "train_tokens.csv"), stringsAsFactors = FALSE)
  te <- read.csv(file.path(d, "test_tokens.csv"), stringsAsFactors = FALSE)
  ids <- read.csv(file.path(d, "test_ids.csv"), stringsAsFactors = FALSE)$visit_id
  y <- read.csv(file.path(d, "train_y.csv"), stringsAsFactors = FALSE)
  mk <- function(t) merge(data.frame(visit_id = t$visit_id, Taxon = t$name, count = t$count, stringsAsFactors = FALSE),
                          tax, by = "Taxon")[, c("visit_id", "Taxon", "count", setdiff(names(tax), "Taxon"))]
  btr <- mk(tok); bte <- mk(te)
  pred <- data.frame(visit_id = ids, stringsAsFactors = FALSE)
  info <- list()
  for (v in names(y)[-1]) {
    yy <- y[!is.na(y[[v]]), c("visit_id", v)]
    t0 <- Sys.time()
    coef <- taxon.env.fast(as.formula(paste("~", v, "+", v, "^2")), btr[btr$visit_id %in% yy$visit_id, ], yy,
                           "visit_id", "count", "visit_id")
    p <- rep(mean(yy[[v]]), length(ids))
    n_inf <- 0; n_incons <- NA
    if (length(coef$tnames) > 0) {
      otu <- get.otu(bte, coef)
      otu <- otu[!is.na(otu$OTU), ]
      if (nrow(otu) > 0) {
        ss <- makess(otu)
        inf <- mlsolve(ss, coef)
        m <- match(ids, as.character(inf[[1]]))
        p[!is.na(m)] <- inf[[v]][m[!is.na(m)]]
        n_inf <- sum(!is.na(m)); n_incons <- sum(inf$Inconsistent)
      }
    }
    pred[[v]] <- p
    info[[v]] <- c(n_train = nrow(yy), n_taxa_modelled = length(coef$tnames), n_inferred = n_inf,
                   n_mean_filled = length(ids) - n_inf, n_inconsistent = n_incons,
                   seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
    cat(v, info[[v]], "\n")
  }
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  write.csv(pred, out, row.names = FALSE)
  write.csv(do.call(rbind, lapply(names(info), function(v) data.frame(target = v, t(info[[v]])))),
            sub("\\.csv$", "_info.csv", out), row.names = FALSE)
}
