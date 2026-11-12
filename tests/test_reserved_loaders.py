# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Shared reserved-site helper (sentinel.data.reserved) and its use in every non-frozen loader of the sonde tiers
(lead ruling 2026-11-10). Each loader is fed a reserved site and must refuse or drop it."""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from sentinel.data import reserved as R

ROOT = Path(__file__).resolve().parents[1]
LOADERS = {   # file -> owner (non-frozen; verified against every FREEZE_*/freeze-pin record on 2026-11-10)
    "scripts/eval/events_feasibility_sonde.py": "M1", "scripts/eval/events_feasibility_external.py": "M1",
    "scripts/eval/sonde_onset_counts.py": "M1", "scripts/train/sonde_virtual.py": "M1",
    "sentinel/models/sim/data.py": "Sim", "sentinel/analysis/causal/diel.py": "Sim",
    "sentinel/models/downstream/killalert.py": "Downstream",
    "sentinel/data/fuse_tier_w.py": "Fuse", "sentinel/data/fuse_tier_wp1.py": "Fuse", "scripts/eval/fuse_wp1_power.py": "Fuse",
    "sentinel/platform/data_access.py": "Platform",
}


def test_pinned_list_and_id_forms():
    some = sorted(R.reserved_ids())[0]
    assert R.is_reserved(some) and R.is_reserved(f"USGS-{some}") and R.is_reserved(f"WQP:USGS-{some}")
    assert not R.is_reserved("01646500")
    assert R.exclude_reserved(["01646500", f"USGS-{some}"]) == ["01646500"]
    with pytest.raises(R.ReservedSiteError):
        R.assert_no_reserved([some])


@pytest.mark.parametrize("f", sorted(LOADERS))
def test_every_listed_loader_uses_the_helper(f):
    assert re.search(r"(data\.reserved|from \.reserved) import", (ROOT / f).read_text()), f


def test_killalert_drops_reserved_gauges(monkeypatch):
    from sentinel.models.downstream import killalert as K
    g = K.iowa_gauges("temp_c_mean")
    if g.empty:
        pytest.skip("no gauges")
    victim = str(g["site_id"].iloc[0])
    monkeypatch.setattr(R, "reserved_ids", lambda: frozenset({victim}))
    assert victim not in set(K.iowa_gauges("temp_c_mean")["site_id"].astype(str))
    with pytest.raises(R.ReservedSiteError):
        K._gauge_anoms(victim, "temp_c_mean", range(2010, 2015), False)


def test_platform_never_serves_reserved(monkeypatch):
    from sentinel.platform import data_access as DA
    from sentinel.platform.config import Settings
    s = Settings.from_env()
    idx = DA.daily_index(s)
    if idx is None or idx.empty:
        pytest.skip("no daily index")
    victim = str(idx["site_no"].iloc[0])
    monkeypatch.setattr(R, "reserved_ids", lambda: frozenset({victim}))
    DA._CACHE.clear() if hasattr(DA, "_CACHE") else None
    with pytest.raises(DA.DataUnavailable):
        DA.observations(s, victim)
    with pytest.raises(DA.DataUnavailable):
        DA.site_record(s, victim)


def test_live_config_sites_never_reserved():
    """Review C-E1: the issuing code does not import the (still edited) helper; instead the frozen live config's
    site lists are checked here against the pinned reserved list."""
    import json
    cfg = json.loads((ROOT / "configs" / "live" / "ew_v2_live.json").read_text())
    sites = {r["site_no"] for v in cfg["sites"].values() for r in v}
    assert not (sites & R.reserved_ids())
    assert "data.reserved" not in (ROOT / "scripts" / "live" / "live_data.py").read_text()


def test_preclaim_audit_detects_reserved_ids(tmp_path):
    some = sorted(R.reserved_ids())[0]
    reg = tmp_path / "registry.jsonl"
    reg.write_text('{"id": "a", "timestamp": "2026-11-12T00:00:00", "experiment": "x", "notes": "site %s"}\n' % some
                   + '{"id": "b", "timestamp": "2026-10-01T00:00:00", "experiment": "y", "notes": "site %s"}\n' % some)
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "data_info.json").write_text('{"sites": ["%s"]}' % some)
    a = R.preclaim_audit(registry_path=reg, runs_root=tmp_path / "runs")
    assert [h["id"] for h in a["registry_hits"]] == ["a"] and len(a["run_record_hits"]) == 1
    with pytest.raises(R.ReservedSiteError):
        R.assert_preclaim_clean(registry_path=reg, runs_root=tmp_path / "runs")


# ------------------------------------------------------------------ repo-wide static guard (review warning, 2026-11-10)
SHARED_TIER = re.compile(r"sonde_daily|sonde_hourly|usgs_iv|SITE_DIR|SONDE_INDEX")
ALLOWLIST = {   # file -> reason it may reference the shared tiers without the helper
    # tier builders / snapshot tools: they WRITE the pinned shared tiers (FREEZE_ew / FREEZE_hydromet content pins)
    "scripts/data/build_sonde_daily.py": "tier builder", "scripts/data/build_sonde_hourly.py": "tier builder",
    "scripts/data/download_usgs_iv.py": "raw IV downloader (new-site raw dirs are mode 000)",
    "scripts/data/compare_sonde_daily_snapshots.py": "snapshot identity tool", "scripts/data/promote_sonde_snapshots.py":
    "snapshot promotion", "scripts/data/build_sonde_split.py": "frozen split builder (refuses to overwrite)",
    "scripts/data/build_chloride_derived.py": "builder of the FREEZE_ew-pinned chloride tier; identity with the Test-A "
                                              "pin is required (ew_v2 §9 W2); its new-site fits are disclosed, never used",
    "scripts/data/promote_chloride_derived.py": "chloride tier promotion",
    "sentinel/data/sonde_hourly.py": "QC library (no site enumeration)",
    # NEON tiers only (NEON site codes; the regex matches 'neon_sonde_*')
    "scripts/data/build_fuse_tiers.py": "NEON tiers only", "sentinel/data/fuse_tiers.py": "NEON tiers only",
    "scripts/data/build_microbe_wq.py": "NEON tiers only", "sentinel/data/microbe_wq.py": "NEON tiers only",
    "scripts/data/build_neon.py": "NEON tiers only", "sentinel/data/neon_aquatic.py": "NEON tiers only",
    # M6-owned (own reserved guard hydromet_sota/reserved.py; outside the ruling's list)
    "scripts/data/build_hydromet_wqp_national.py": "M6-owned; M6 reserved guard",
    # new-site / validation tooling (train-role split sites only, or the protected tier itself)
    "scripts/data/download_newsites.py": "protected new-site pull", "scripts/data/ivrule_revision.py":
    "train-role split sites only", "scripts/data/validate_iv_daily_rule.py": "train-role split sites only",
    "sentinel/data/reserved.py": "the helper itself",
    # Sim: sites only through sim.data.build / causal.diel (both call the helper); globs only own outputs
    "sentinel/analysis/causal/ladder.py": "via sim.data / diel", "scripts/eval/sim_causal_real.py": "via sim.data / diel",
    "scripts/eval/sim_e1.py": "via sim.data", "scripts/eval/sim_e3.py": "via sim.data",
    # M1 EW tools reading only run outputs of split-gated calibrations
    "scripts/eval/sonde_events_register.py": "run outputs only", "scripts/eval/sonde_ew_freeze_candidate.py": "run outputs",
    # live trial: issuing code must not import the helper (review C-E1); config sites tested instead
    "scripts/live/live_data.py": "C-E1 (test_live_config_sites_never_reserved)", "scripts/live/issue.py": "C-E1",
    "scripts/live/build_label_snapshot.py": "C-E1 (config sites)", "scripts/live/build_live_config.py": "split sites",
    "scripts/live/build_series_map.py": "config sites", "scripts/live/derive_qualifier_map.py": "train/val config sites",
    "scripts/live/power_projection.py": "train/val config sites",
    # Platform: all reads go through platform.data_access (helper)
    "sentinel/platform/api/app.py": "via data_access", "sentinel/platform/app/views/map_view.py": "via data_access",
    "sentinel/platform/app/views/site_page.py": "via data_access", "sentinel/platform/provenance.py": "labels only",
    "sentinel/platform/validation.py": "labels only", "sentinel/platform/variables.py": "labels only",
}


def _frozen_files() -> set:
    out = set()
    for f in (ROOT / "docs" / "preregistration").glob("*.json"):
        if "freeze" in f.name.lower():
            out |= set(re.findall(r"(?:sentinel|scripts)/[\w/]+\.py", f.read_text()))
    return out


def test_no_unguarded_reader_of_the_shared_tiers():
    frozen = _frozen_files()
    bad = []
    for f in sorted(list((ROOT / "sentinel").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))):
        rel = f.relative_to(ROOT).as_posix()
        t = f.read_text(errors="ignore")
        if not SHARED_TIER.search(t) or rel in frozen or rel in ALLOWLIST:
            continue
        if not re.search(r"(data\.reserved|from \.reserved) import", t):
            bad.append(rel)
    assert not bad, f"files reading the shared sonde tiers without sentinel.data.reserved: {bad}"
