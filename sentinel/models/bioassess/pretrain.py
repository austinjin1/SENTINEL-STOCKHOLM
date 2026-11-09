# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""UK EA pretraining of the BioSet encoder (protocol §6, candidate ``bioset_st_ukpt``).

Objectives on the UK pretraining pool (reserved UK test blocks excluded):

1. **masked-taxon modelling** — 15% of a sample's taxa have all lineage ids replaced by MASK
   (their abundance features are kept); the encoder predicts the masked taxon's family and order;
2. **community -> chemistry** — masked MSE on the 12 standardised UK ±30-day chemistry targets
   (samples without matched chemistry contribute only to objective 1).

Early stopping uses the pool's validation blocks (5% of 50-km blocks), never the reserved test.
The checkpoint stores the vocabulary (UK pool + NRSA development names; names only), the encoder
weights, the ids seen in the pool (``pool_allowed``) and the encoder config; trunk and heads are
discarded and NRSA fine-tuning starts fresh heads.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from sentinel.models.bioassess import bioset as BS
from sentinel.models.bioassess.uk import UK_TARGETS

GLOB_UK = ("log10_total_count", "log10_n_taxa")


def _mask_tokens(lev, mask, frac, gen):
    import torch

    r = torch.rand(mask.shape, generator=gen, device=mask.device)
    m = (r < frac) & mask
    # guarantee >= 1 masked token in every sample with >= 2 tokens
    first = torch.where(mask.sum(1) >= 2, torch.argmax(r.masked_fill(~mask, -1.0), 1), torch.full_like(mask.sum(1), -1))
    rows = torch.arange(mask.shape[0], device=mask.device)
    ok = first >= 0
    m[rows[ok], first[ok]] = True
    lev_in = lev.clone()
    lev_in[m] = BS.MASK
    return lev_in, m


def pretrain_uk(cfg: dict, pcfg: dict, uk, nrsa_dev_tokens, *, device: str, seed: int = 0, log=print):
    """Return (checkpoint dict, history list, final validation diagnostics)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from sentinel.core.seed import seed_everything

    seed_everything(seed)
    pool = uk.samples
    if (pool["role"] == "reserved_test").any():
        raise AssertionError("reserved UK test samples passed to pretraining")
    fit_ids = pool.index[pool["role"] == "pool_fit"].to_numpy()
    val_ids = pool.index[pool["role"] == "pool_val"].to_numpy()
    uk_tok = uk.tokens.rename(columns={"analysis_id": "visit_id"})
    vocab = BS.TaxonVocab.build(uk_tok, nrsa_dev_tokens)
    min_c = int(cfg.get("min_name_count", 3))
    pool_allowed = vocab.allowed_from(uk_tok[uk_tok["visit_id"].isin(set(fit_ids))], min_c)
    T = int(pcfg.get("max_tokens", 96))

    def enc(ids):
        g = pool.loc[ids, list(GLOB_UK)].to_numpy(np.float32)
        e = BS.encode(uk_tok, ids, vocab, g, max_tokens=T)
        return BS.restrict(e, pool_allowed)

    e_fit, e_val = enc(fit_ids), enc(val_ids)
    ycols = list(UK_TARGETS)
    Yf = uk.Y.loc[fit_ids, ycols].to_numpy(np.float32)
    Yv = uk.Y.loc[val_ids, ycols].to_numpy(np.float32)
    ym, ys = np.nanmean(Yf, 0), np.nanstd(Yf, 0)
    ys = np.where(ys > 0, ys, 1.0)
    gm, gs = e_fit.glob.mean(0), e_fit.glob.std(0)
    gs = np.where(gs > 0, gs, 1.0)
    sizes = {lv: vocab.size(lv) for lv in BS.LEVELS}
    model = BS.build_model(cfg, sizes, len(GLOB_UK), len(ycols)).to(device)
    d = int(cfg["d_model"])
    j_fam, j_ord = BS.LEVELS.index("family"), BS.LEVELS.index("order")
    heads = nn.ModuleDict({"family": nn.Linear(d, sizes["family"]), "order": nn.Linear(d, sizes["order"])}).to(device)
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=float(pcfg["lr"]), weight_decay=float(pcfg["weight_decay"]))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    frac = float(pcfg.get("mask_frac", 0.15))
    w_chem = float(pcfg.get("chem_loss_weight", 1.0))

    def batch(e, Y, idx):
        mk = e.mask[idx]
        Tb = max(int(mk.sum(1).max()), 1)
        lev = torch.as_tensor(e.lev[idx, :Tb], device=device)
        ab = torch.as_tensor(e.ab[idx, :Tb], device=device)
        m = torch.as_tensor(mk[:, :Tb], device=device)
        g = torch.as_tensor(((e.glob[idx] - gm) / gs).astype(np.float32), device=device)
        y = torch.as_tensor(((Y[idx] - ym) / ys).astype(np.float32), device=device)
        return lev, ab, m, g, y

    def step(lev, ab, m, g, y, train: bool, gen_):
        if train:
            lev, ab, m = BS._augment(lev, ab, m, cfg, gen_)
        lev_in, mm = _mask_tokens(lev, m, frac, gen_)
        h = model.encoder.tokens(lev_in, ab, m)
        z = model.encoder.pool(h, ab, m)
        chem = model.head(model.trunk(torch.cat([z, g], -1)))
        hm = h[mm]
        tf, to = lev[..., j_fam][mm], lev[..., j_ord][mm]
        lf = F.cross_entropy(heads["family"](hm), tf, ignore_index=BS.PAD) if (tf > BS.MASK).any() else hm.sum() * 0
        lo = F.cross_entropy(heads["order"](hm), to, ignore_index=BS.PAD) if (to > BS.MASK).any() else hm.sum() * 0
        lc = BS.masked_mse(chem, y)
        acc = ((heads["family"](hm).argmax(-1) == tf) & (tf > BS.MASK)).sum().item(), int((tf > BS.MASK).sum())
        return lf + 0.5 * lo + w_chem * lc, (float(lf), float(lo), float(lc)), acc, chem

    bs = int(pcfg["batch_size"])
    best, best_state, best_ep, bad, hist = math.inf, None, -1, 0, []
    rs = np.random.default_rng(seed)
    for ep in range(int(pcfg["epochs"])):
        model.train()
        heads.train()
        perm = rs.permutation(len(fit_ids))
        agg = np.zeros(3)
        nb = 0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            loss, parts, _, _ = step(*batch(e_fit, Yf, idx), True, gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            agg += parts
            nb += 1
        model.eval()
        heads.eval()
        vgen = torch.Generator(device=device)
        vgen.manual_seed(12345)
        vagg, vnb, hit, tot = np.zeros(3), 0, 0, 0
        preds = []
        with torch.no_grad():
            for s in range(0, len(val_ids), 1024):
                idx = np.arange(s, min(s + 1024, len(val_ids)))
                loss, parts, (h_, t_), chem = step(*batch(e_val, Yv, idx), False, vgen)
                vagg += parts
                vnb += 1
                hit += h_
                tot += t_
                preds.append(chem.cpu().numpy())
        vparts = vagg / max(vnb, 1)
        vloss = vparts[0] + 0.5 * vparts[1] + w_chem * vparts[2]
        P = np.concatenate(preds) * ys + ym
        from sentinel.eval import metrics as EM

        r2 = {c: EM.r2(Yv[:, k], P[:, k]) for k, c in enumerate(ycols)}
        rec = {"epoch": ep, "train_family_ce": agg[0] / nb, "train_order_ce": agg[1] / nb, "train_chem_mse": agg[2] / nb,
               "val_family_ce": vparts[0], "val_order_ce": vparts[1], "val_chem_mse": vparts[2], "val_loss": vloss,
               "val_family_acc": hit / max(tot, 1), "val_chem_r2": r2}
        hist.append(rec)
        log(f"epoch {ep}: val loss {vloss:.4f} fam-acc {rec['val_family_acc']:.3f} chem-mse {vparts[2]:.4f}")
        if vloss < best - 1e-4:
            best, best_ep, bad = vloss, ep, 0
            best_state = copy.deepcopy(model.encoder.state_dict())
        else:
            bad += 1
            if bad >= int(pcfg.get("patience", 3)):
                break
    ck = {"vocab": vocab.to_dict(), "encoder_state": {k: v.cpu() for k, v in best_state.items()},
          "pool_allowed": [a.tolist() for a in pool_allowed], "cfg": dict(cfg), "pretrain_cfg": dict(pcfg),
          "best_epoch": best_ep, "n_fit": int(len(fit_ids)), "n_val": int(len(val_ids)),
          "uk_targets": ycols, "seed": seed}
    return ck, hist
