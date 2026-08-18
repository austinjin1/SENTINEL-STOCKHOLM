#!/usr/bin/env python
"""
Phase 1 — DEFINE real specific-conductance (HIGH) excursion events on NEON Track A sites.

All criteria act on the RAW QC-clean conductivity value ONLY (independent of the detector
reconstruction residual => non-circular). This gates EVALUATION only.

EVENT DEFINITION (a-priori, site-relative, defensible):
  Per site, from mask_conductivity-clean finite values:
    - robust baseline: median (m), robust sigma = IQR/1.349
    - HIGH threshold THR = max( m + 4*sigma_robust , p99.5_clean )   [both robust & tail-based;
      taking the max makes it conservative for very tight-IQR sites]
  ONSET = first QC-clean timestamp t0 where cond(t0) > THR, SUSTAINED:
    - >= SUSTAIN_H hours of clean samples above THR within the sustain window
      (>= SUSTAIN_FRAC of the clean samples in [t0, t0+SUSTAIN_H] are > THR)
  PRECEDED by clean below-threshold baseline:
    - in the RUNWAY_D days before t0, there must be clean coverage and
      NO clean sample above THR (a genuine departure from baseline), and
      >= RUNWAY_MIN_FRAC clean coverage in that window.
  DRIFT / ARTIFACT SCREEN (reject if any):
    - Monotone-ramp-that-never-returns: after onset, cond stays above THR for essentially the
      whole remaining clean record (fraction-above-after >= RAMP_FRAC) => likely drift/step, not excursion.
    - Return-to-baseline sanity: a genuine excursion should return below (m + 2*sigma) at some
      clean point after onset within the record; if it never returns AND is a long ramp => reject.
    - Flatline: excursion window has ~zero variance across many bins (stuck sensor) => reject.
    - Whole-record shift: baseline itself (pre-onset) is not distinguishable (THR within noise) —
      require THR >= m + MIN_REL*sigma effectively guaranteed by construction.
  A site is an EVENT if a qualifying onset exists.
  CONTROL = broad clean record (coverage >= CTRL_MIN_FRAC, span >= CTRL_MIN_DAYS) with NO qualifying
    onset AND clean p99.5 not implausibly close to max (no lurking excursion).

Also require the onset to fall inside the residual-cache covered-window range so a pre-onset
runway of detector windows exists (enough_for_benchmark).
"""
import numpy as np, glob, os, json, datetime as dt

RAW = '/home/bcheng/SENTINEL/data/processed/trackA'
CACHE_DIR = '/home/bcheng/SENTINEL/results/trackA_leadtime/aquassm_specificity'
OUT = '/home/bcheng/SENTINEL/results/trackA_leadtime/conductivity/conductance_events.json'
STEP = 900  # 15 min

# --- a-priori hyperparameters (fixed; NOT tuned on any lead) ---
SUSTAIN_H      = 6.0     # hours sustained above THR
SUSTAIN_FRAC   = 0.75    # >=75% of clean samples in sustain window above THR
RUNWAY_D       = 7.0     # clean below-threshold runway required before onset
RUNWAY_MIN_FRAC= 0.30    # >=30% clean coverage in the runway window (real record, not a gap)
RAMP_FRAC      = 0.95    # if >=95% of post-onset clean samples stay above THR => drift/ramp, reject
FLAT_MINSTD    = 1e-3    # sustain-window std below this (in uS/cm) => stuck/flatline, reject
CTRL_MIN_FRAC  = 0.60    # control coverage
CTRL_MIN_DAYS  = 120     # control span

def load_cache():
    cov = {}
    for cf in ['residual_cache.npz','residual_cache_expanded.npz']:
        d = np.load(os.path.join(CACHE_DIR, cf), allow_pickle=True)
        for k in d.keys():
            if k.endswith('__end_ts'):
                s = k.split('__')[0]
                ets = d[k].astype(np.int64)
                cov[s] = (int(ets.min()), int(ets.max()), int(ets.size))
    return cov

def utc(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).isoformat()

def analyze(site, f, cov):
    d = np.load(f, allow_pickle=True)
    c = d['conductivity'].astype(float)
    m = d['mask_conductivity'].astype(bool)
    ts = d['ts'].astype(np.int64)
    finite = np.isfinite(c)
    clean = m & finite
    vc = c[clean]
    n_clean = int(vc.size)
    span_days = float((ts.max()-ts.min())/86400.0)
    if n_clean < 200:
        return {'site':site,'role_cond':'excluded','reason':'insufficient clean conductance',
                'n_clean':n_clean}
    med = float(np.median(vc))
    q25,q75 = np.percentile(vc,[25,75])
    sig = float((q75-q25)/1.349)
    p995 = float(np.percentile(vc,99.5))
    p99  = float(np.percentile(vc,99.0))
    mx = float(vc.max())
    THR = max(med + 4.0*sig, p995)
    frac_cov = n_clean/len(ts)

    # clean index arrays for temporal ops
    idx = np.where(clean)[0]
    tsc = ts[idx]; cvc = c[idx]
    above = cvc > THR

    sustain_bins = int(np.ceil(SUSTAIN_H*3600.0/STEP))
    runway_sec = RUNWAY_D*86400.0

    onset = None
    onset_info = None
    for j in range(len(idx)):
        if not above[j]:
            continue
        t0 = tsc[j]
        # sustain window [t0, t0+SUSTAIN_H]
        sw = (tsc >= t0) & (tsc < t0 + SUSTAIN_H*3600.0)
        if sw.sum() < max(4, int(0.5*sustain_bins)):
            continue  # not enough clean samples to judge sustain
        if above[sw].mean() < SUSTAIN_FRAC:
            continue
        if float(np.std(cvc[sw])) < FLAT_MINSTD:
            continue  # flatline/stuck
        # runway: [t0-RUNWAY_D, t0) clean, no sample above THR, adequate coverage
        rw = (tsc >= t0 - runway_sec) & (tsc < t0)
        n_rw = int(rw.sum())
        exp_rw = runway_sec/STEP
        if n_rw < RUNWAY_MIN_FRAC*exp_rw:
            continue  # not enough clean history -> can't establish clean baseline
        if above[rw].any():
            continue  # baseline not clean/below-threshold
        # drift/ramp screen: after onset, fraction of clean above THR
        af = tsc >= t0
        frac_after = float(above[af].mean()) if af.sum()>0 else 1.0
        # return-to-baseline: does it drop below med+2sig after onset?
        returns = bool((cvc[af] < med + 2.0*sig).any())
        if frac_after >= RAMP_FRAC and not returns:
            continue  # monotone ramp never returning => drift/step artifact, reject
        # peak of the excursion (from onset onward, contiguous-ish region)
        # take peak within RUNWAY window after onset for a stable "event peak"
        pk_win = af & (tsc < t0 + 30*86400.0)
        peak = float(cvc[pk_win].max()) if pk_win.sum()>0 else float(cvc[af].max())
        onset = t0
        onset_info = dict(frac_after=frac_after, returns=returns, n_runway=n_rw,
                          runway_cov_frac=n_rw/exp_rw, peak=peak,
                          sustain_frac=float(above[sw].mean()),
                          sustain_std=float(np.std(cvc[sw])))
        break

    # excursion episode structure at THR (for honest characterization) — group above-runs
    above_all = cvc > THR
    hi = np.where(above_all)[0]
    episodes = []
    if len(hi):
        st = hi[0]; pv = hi[0]
        for k in hi[1:]:
            if tsc[k]-tsc[pv] > 12*3600:
                episodes.append((st,pv)); st = k
            pv = k
        episodes.append((st,pv))
    ep_durs = [ (tsc[b]-tsc[a])/3600.0 for a,b in episodes ]
    n_sustained_ep = int(sum(1 for dh in ep_durs if dh >= SUSTAIN_H))
    ratio_max_p995 = float(mx/p995) if p995>0 else float('nan')

    base = dict(site=site, n_clean=n_clean, frac_cov=round(frac_cov,3),
                span_days=round(span_days,1), median=round(med,2),
                iqr_sigma=round(sig,2), p99=round(p99,2), p995=round(p995,2),
                max=round(mx,2), threshold=round(THR,2),
                n_excursion_episodes=len(episodes),
                n_sustained_episodes=n_sustained_ep,
                ratio_max_over_p995=round(ratio_max_p995,2))

    if onset is not None:
        cinfo = cov.get(site)
        runway_days_pre = None; n_cov_before = None; in_cache = False
        if cinfo:
            cmin,cmax,cn = cinfo
            in_cache = (onset > cmin) and (onset <= cmax)
            # count residual windows ending before onset (pre-onset runway of detector windows)
            # reload end_ts precisely
        base.update(dict(role_cond='event',
                         onset_ts=int(onset), onset_utc=utc(onset),
                         event_peak=round(onset_info['peak'],2),
                         pre_onset_runway_days=round(onset_info['runway_cov_frac']*RUNWAY_D,2),
                         runway_clean_bins=onset_info['n_runway'],
                         frac_after=round(onset_info['frac_after'],3),
                         returns_to_baseline=onset_info['returns'],
                         sustain_frac=round(onset_info['sustain_frac'],3),
                         onset_in_cache_range=bool(in_cache)))
        return base
    else:
        # Did the site HAVE sustained excursions but fail the runway/artifact screen?
        had_sustained = base['n_sustained_episodes'] > 0
        if had_sustained:
            base.update(dict(role_cond='near_miss',
                             reason='sustained high excursion present but failed clean pre-onset '
                                    'runway or drift/return screen (ambiguous / poor runway coverage)'))
            return base
        # control eligibility (broad clean record, no excursion at all)
        is_ctrl = (frac_cov >= CTRL_MIN_FRAC) and (span_days >= CTRL_MIN_DAYS)
        base.update(dict(role_cond=('control' if is_ctrl else 'excluded'),
                         reason=None if is_ctrl else 'no excursion; record too short/sparse for control'))
        return base

def main():
    cov = load_cache()
    files = sorted(glob.glob(os.path.join(RAW,'*.npz')))
    results = {}
    for f in files:
        s = os.path.basename(f).replace('.npz','')
        results[s] = analyze(s, f, cov)

    # For event sites, count residual windows ending strictly before onset (pre-onset detector runway)
    ends = {}
    for cf in ['residual_cache.npz','residual_cache_expanded.npz']:
        d = np.load(os.path.join(CACHE_DIR, cf), allow_pickle=True)
        for k in d.keys():
            if k.endswith('__end_ts'):
                ends[k.split('__')[0]] = d[k].astype(np.int64)
    for s,r in results.items():
        if r.get('role_cond')=='event':
            ets = ends.get(s)
            if ets is not None:
                r['n_cov_windows_before_onset'] = int((ets < r['onset_ts']).sum())
                r['n_cov_windows_total'] = int(ets.size)
            else:
                r['n_cov_windows_before_onset'] = 0
                r['n_cov_windows_total'] = 0

    ev = [s for s,r in results.items() if r.get('role_cond')=='event']
    ev_with_runway = [s for s in ev if results[s].get('n_cov_windows_before_onset',0) >= 1
                      and results[s].get('onset_in_cache_range')]
    ctrl = [s for s,r in results.items() if r.get('role_cond')=='control']
    near = [s for s,r in results.items() if r.get('role_cond')=='near_miss']
    excl = [s for s,r in results.items() if r.get('role_cond') in ('excluded',)]

    manifest = dict(
        criteria=dict(SUSTAIN_H=SUSTAIN_H,SUSTAIN_FRAC=SUSTAIN_FRAC,RUNWAY_D=RUNWAY_D,
                      RUNWAY_MIN_FRAC=RUNWAY_MIN_FRAC,RAMP_FRAC=RAMP_FRAC,
                      threshold_rule="max(median+4*IQR/1.349, p99.5_clean)",
                      direction="HIGH specific conductance",
                      bin_seconds=STEP),
        n_event_sites=len(ev),
        n_event_sites_with_cache_runway=len(ev_with_runway),
        n_control_sites=len(ctrl),
        n_near_miss=len(near),
        n_excluded=len(excl),
        enough_for_benchmark=bool(len(ev_with_runway) >= 6),
        event_sites=ev, event_sites_with_cache_runway=ev_with_runway,
        control_sites=ctrl, near_miss_sites=near, excluded_sites=excl,
        sites=results)
    with open(OUT,'w') as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote", OUT)
    print("EVENT sites:", len(ev), "->", ev)
    print("EVENT with cache runway:", len(ev_with_runway), "->", ev_with_runway)
    print("CONTROL:", len(ctrl))
    print("NEAR_MISS:", len(near), "->", near)
    print("EXCLUDED:", len(excl), "->", excl)
    print()
    for s in ev:
        r=results[s]
        print(f"{s}: onset={r['onset_utc']} peak={r['event_peak']} THR={r['threshold']} "
              f"med={r['median']} sig={r['iqr_sigma']} runway_d={r['pre_onset_runway_days']} "
              f"frac_after={r['frac_after']} returns={r['returns_to_baseline']} "
              f"cov_before={r.get('n_cov_windows_before_onset')} incache={r.get('onset_in_cache_range')}")

if __name__=='__main__':
    main()
