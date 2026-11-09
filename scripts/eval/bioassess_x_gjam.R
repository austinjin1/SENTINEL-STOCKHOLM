# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
# M5-X comparator §6.4: GJAM joint-species inverse prediction (Clark et al. 2017; CRAN gjam 2.7), amendment v4.8.
#   Rscript scripts/eval/bioassess_x_gjam.R fold <fold_dir> <out.csv> [fidelity]
# Per fold: <= 5,000 same-continent labelled training rows (seed 0, stratified by region, proportional) x the 100
# most prevalent mixed_otu taxa (presence/absence, typeNames "PA"); covariates = the block's 6 targets
# standardised with these rows' statistics (missing values imputed by gjam); ng = 2000, burnin = 500.
# Inverse prediction of new samples: Gibbs sampler with gjam's posterior-mean coefficients (betaMu) and correlation
# matrix (corMu): w | y, x (truncated normal, one species at a time) and x | w (Gaussian with gjam's near-flat
# prior, truncated to the training range, one covariate at a time); 200 burn-in + 1,000 iterations; posterior
# mean. "fidelity": the same routine on the fit rows vs gjam's in-sample xpredMu (criterion r >= 0.90 per target).
suppressMessages(library(gjam))
args <- commandArgs(trailingOnly = TRUE)
d <- args[2]; out <- args[3]; fidelity <- length(args) >= 4 && args[4] == "fidelity"
set.seed(0)
N_MAX <- 5000; N_TAXA <- 100; NG <- 2000; BURN <- 500; IT_BURN <- 200; IT <- 1000
if (Sys.getenv("GJAM_SMOKE") == "1") { N_MAX <- 400; NG <- 300; BURN <- 100; IT_BURN <- 50; IT <- 100 }  # code smoke only

tok <- read.csv(file.path(d, "train_tokens.csv"), stringsAsFactors = FALSE)
y <- read.csv(file.path(d, "train_y.csv"), stringsAsFactors = FALSE)
reg <- read.csv(file.path(d, "train_region.csv"), stringsAsFactors = FALSE)
te <- read.csv(file.path(d, "test_tokens.csv"), stringsAsFactors = FALSE)
ids <- read.csv(file.path(d, "test_ids.csv"), stringsAsFactors = FALSE)$visit_id
targets <- names(y)[-1]

# ---- stratified subsample (proportional by region)
r <- reg$ag_eco9[match(y$visit_id, reg$visit_id)]
if (nrow(y) > N_MAX) {
  cnt <- table(r); alloc <- floor(cnt * N_MAX / nrow(y))
  rem <- N_MAX - sum(alloc); fr <- cnt * N_MAX / nrow(y) - alloc
  o <- order(-fr); alloc[o[seq_len(rem)]] <- alloc[o[seq_len(rem)]] + 1
  pick <- unlist(lapply(names(alloc), function(g) { w <- which(r == g); w[sample.int(length(w), alloc[[g]])] }))
  y <- y[sort(pick), ]
}
tok <- tok[tok$visit_id %in% y$visit_id & !is.na(tok$mixed_otu), ]
prev <- tapply(tok$visit_id, tok$mixed_otu, function(v) length(unique(v)))
taxa <- names(prev)[order(-prev, names(prev))][seq_len(min(N_TAXA, length(prev)))]
pa <- function(t, sid) {
  M <- matrix(0, length(sid), length(taxa), dimnames = list(sid, make.names(taxa, unique = TRUE)))
  t <- t[t$mixed_otu %in% taxa & t$visit_id %in% sid, ]
  M[cbind(match(t$visit_id, sid), match(t$mixed_otu, taxa))] <- 1
  M
}
Ytr <- pa(tok, y$visit_id)
keep <- colSums(Ytr) > 0 & colSums(Ytr) < nrow(Ytr)
Ytr <- Ytr[, keep, drop = FALSE]; taxa <- taxa[keep]
mu <- sapply(targets, function(v) mean(y[[v]], na.rm = TRUE))
sdv <- sapply(targets, function(v) sd(y[[v]], na.rm = TRUE))
xs <- as.data.frame(sapply(targets, function(v) (y[[v]] - mu[v]) / sdv[v]))
names(xs) <- paste0("x", seq_along(targets))
lo <- sapply(xs, min, na.rm = TRUE); hi <- sapply(xs, max, na.rm = TRUE)
form <- as.formula(paste("~", paste(names(xs), collapse = " + ")))
t0 <- Sys.time()
fit <- gjam(form, xdata = xs, ydata = as.data.frame(Ytr),
            modelList = list(ng = NG, burnin = BURN, typeNames = "PA"))
fit_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
B <- fit$parameters$betaMu          # (1 + q) x S
R <- fit$parameters$corMu           # S x S
B <- B[c("intercept", names(xs)), colnames(Ytr), drop = FALSE]
R <- R[colnames(Ytr), colnames(Ytr)]

rtnorm <- function(m, s, a, b) {
  pa_ <- pnorm((a - m) / s); pb_ <- pnorm((b - m) / s)
  u <- pa_ + runif(length(m)) * pmax(pb_ - pa_, 1e-12)
  u <- pmin(pmax(u, 1e-12), 1 - 1e-12)
  m + s * qnorm(u)
}

inverse <- function(Y, xobs = NULL) {
  # xobs = NULL: out-of-sample mode (w | y, current x draw). xobs given (training rows): gjam's in-sample scheme,
  # w | y, observed x (missing entries: current draw), then x | w -- used only for the fidelity check (v6).
  n <- nrow(Y); S <- ncol(Y); q <- nrow(B) - 1
  b0 <- B[1, ]; Bx <- B[-1, , drop = FALSE]
  P <- solve(R)
  x <- matrix(0, n, q)
  w <- ifelse(Y == 1, 0.5, -0.5)
  Qx <- Bx %*% P %*% t(Bx) + diag(1e-5, q)       # posterior precision of x given w (gjam priorXIV = 1e-5)
  acc <- matrix(0, n, q)
  lo_w <- ifelse(Y == 1, 0, -Inf); hi_w <- ifelse(Y == 1, Inf, 0)
  for (it in seq_len(IT_BURN + IT)) {
    xw <- x
    if (!is.null(xobs)) { ok <- !is.na(xobs); xw[ok] <- xobs[ok] }
    m <- matrix(b0, n, S, byrow = TRUE) + xw %*% Bx
    res <- w - m
    for (s in seq_len(S)) {
      cm <- m[, s] - (res %*% P[, s] - res[, s] * P[s, s]) / P[s, s]
      w[, s] <- rtnorm(cm, 1 / sqrt(P[s, s]), lo_w[, s], hi_w[, s])
      res[, s] <- w[, s] - m[, s]
    }
    mx <- ((w - matrix(b0, n, S, byrow = TRUE)) %*% P %*% t(Bx)) %*% solve(Qx)
    for (k in seq_len(q)) {
      cmk <- mx[, k] - ((x - mx)[, -k, drop = FALSE] %*% Qx[-k, k]) / Qx[k, k]
      x[, k] <- rtnorm(cmk, 1 / sqrt(Qx[k, k]), lo[k], hi[k])
    }
    if (it > IT_BURN) acc <- acc + x
  }
  acc / IT
}

info <- list(n_fit = nrow(Ytr), n_taxa = ncol(Ytr), fit_minutes = fit_min)
saveRDS(list(B = B, R = R, xpredMu = fit$prediction$xpredMu, xs = xs, Ytr = Ytr, taxa = taxa, mu = mu, sdv = sdv,
             lo = lo, hi = hi), sub("\\.csv$", "_fit.rds", out))
if (fidelity) {
  xg <- fit$prediction$xpredMu[, names(xs), drop = FALSE]
  cr <- function(a) { r <- sapply(seq_along(targets), function(k) suppressWarnings(cor(a[, k], xg[, k],
                                                                    use = "pairwise.complete.obs"))); names(r) <- targets; r }
  xin_s <- inverse(Ytr, as.matrix(xs))      # gjam's in-sample scheme (fidelity criterion, amendment v6)
  xin_o <- inverse(Ytr)                     # out-of-sample mode on the same rows (informational)
  info$fidelity_r_insample_mode <- cr(xin_s)
  info$fidelity_r_outofsample_mode <- cr(xin_o)
  info$xpredMu_n_finite <- colSums(is.finite(xg))
  rr <- info$fidelity_r_insample_mode
  info$fidelity_pass <- all(is.finite(rr)) && all(rr >= 0.90)
  print(info[c("fidelity_r_insample_mode", "fidelity_r_outofsample_mode", "xpredMu_n_finite", "fidelity_pass")])
}
t1 <- Sys.time()
Yte <- pa(te, ids)
colnames(Yte) <- colnames(Ytr)
xp <- inverse(Yte)
pred <- data.frame(visit_id = ids, stringsAsFactors = FALSE)
for (k in seq_along(targets)) pred[[targets[k]]] <- xp[, k] * sdv[k] + mu[k]
info$predict_minutes <- as.numeric(difftime(Sys.time(), t1, units = "mins"))
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
write.csv(pred, out, row.names = FALSE)
writeLines(capture.output(str(info)), sub("\\.csv$", "_info.txt", out))
print(info[c("n_fit", "n_taxa", "fit_minutes", "predict_minutes")])
