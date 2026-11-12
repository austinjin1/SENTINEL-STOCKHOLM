# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EW-v2 live-trial pipeline (scripts/live): hash chain, qualifier mapping, series map, cube gap guard."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
import live_common as LC  # noqa: E402
import live_data as LD  # noqa: E402


def test_hash_chain_detects_tampering(tmp_path):
    log = tmp_path / "h.jsonl"
    for d in ("2026-12-01", "2026-12-02", "2026-12-03"):
        LC.append_hashlog(log, {"issue_date": d, "status": "issued", "files": {"issue.parquet": d}})
    rows = LC.read_hashlog(log)
    assert LC.verify_chain(rows) == []
    with pytest.raises(FileExistsError):
        LC.append_hashlog(log, {"issue_date": "2026-12-02", "status": "issued", "files": {}})
    rows[1]["files"]["issue.parquet"] = "edited"
    assert LC.verify_chain(rows)                       # edited row detected
    rows = LC.read_hashlog(log)
    del rows[1]
    assert LC.verify_chain(rows)                       # deleted row detected


def test_excl_write_refuses_second_write(tmp_path):
    p = tmp_path / "m.json"
    LC.excl_write(p, {"a": 1})
    with pytest.raises(FileExistsError):
        LC.excl_write(p, {"a": 2})


def test_ogc_qualifiers_mapped_to_frozen_masking():
    base = {"monitoring_location_id": "USGS-01", "parameter_code": "00300", "statistic_id": "00002",
            "time": "2026-12-01", "time_series_id": "t1"}
    recs = [{**base, "value": 5.0, "approval_status": "Provisional", "qualifier": None},
            {**base, "time": "2026-12-02", "value": 4.0, "approval_status": "Provisional", "qualifier": "['ESTIMATED']"},
            {**base, "time": "2026-12-03", "value": 0.1, "approval_status": "Approved", "qualifier": ["LESSTHAN"]},
            {**base, "time": "2026-12-04", "value": 6.0, "approval_status": "Approved", "qualifier": "['REVISED']"}]
    long, unknown = LD.ogc_long(recs)
    wide, _ = LD.wide_by_site(long)
    w = wide["01"]
    assert w.loc["2026-12-01", "do_mgl_min"] == 5.0 and bool(w.loc["2026-12-01", "do_mgl_min_prov"])
    assert "2026-12-02" not in w.index.strftime("%Y-%m-%d")      # estimated -> masked
    assert "2026-12-03" not in w.index.strftime("%Y-%m-%d")      # censored -> masked
    assert not bool(w.loc["2026-12-04", "do_mgl_min_prov"])       # approved, REVISED is kept (legacy agreement)
    assert unknown == {}


def test_series_map_keeps_only_mapped_series():
    long = pd.DataFrame({"site_no": ["01"] * 4, "date": pd.to_datetime(["2026-12-01"] * 2 + ["2026-12-02"] * 2),
                         "parm_cd": "00300", "stat_cd": "00002", "ts_id": ["a", "b", "a", "b"],
                         "value": [5.0, 9.0, 5.5, 9.5], "qualifier": "P"})
    kept, info = LD.apply_series_map(long, {"01|00300|00002": "a"})
    assert set(kept["ts_id"]) == {"a"} and info["n_dropped_rows"] == 2


def test_cube_never_contains_wy2022_2026(tmp_path, monkeypatch):
    idx = pd.date_range("2021-09-25", "2026-10-10")
    df = pd.DataFrame({"do_mgl_min": np.arange(len(idx), dtype="float32"), "do_mgl_min_prov": False}, index=idx)
    (tmp_path / "sites").mkdir()
    df.to_parquet(tmp_path / "sites" / "S1.parquet")
    monkeypatch.setitem(LD.SNAPSHOT_DIR, "dir", tmp_path)
    monkeypatch.setitem(LD.SNAPSHOT_DIR, "sha", None)
    cube, _ = LD.build_cube(["S1"], ["do_mgl_min"], "2026-10-10", window_start="2026-10-01", window_frames=None,
                            window_from_snapshot=True)
    gap = (cube.dates > "2021-09-30") & (cube.dates < "2026-10-01")
    assert not np.isfinite(cube.y[0, gap, 0]).any()
    assert np.isfinite(cube.y[0, cube.dates >= "2026-10-01", 0]).all()


def test_code_hash_covers_live_and_frozen_event_code():
    files = LC.code_files()
    for f in ("scripts/live/issue.py", "scripts/train/sonde_events.py", "sentinel/models/sonde/events.py",
              "sentinel/eval/hazards.py", "sentinel/models/sonde/features.py"):
        assert f in files
    json.dumps(files)


# ---------------------------------------------------------------- review 2026-11-09 (C-L1..C-L3, W-L6, W-L10, W-L13)
def test_status_rows_failed_repeat_issued_once(tmp_path):
    log = tmp_path / "h.jsonl"
    LC.append_hashlog(log, {"issue_date": "2026-12-01", "status": "failed", "files": {}})
    LC.append_hashlog(log, {"issue_date": "2026-12-01", "status": "failed", "files": {}})
    LC.append_hashlog(log, {"issue_date": "2026-12-01", "status": "issued", "files": {"issue.parquet": "x"}})
    with pytest.raises(FileExistsError):
        LC.append_hashlog(log, {"issue_date": "2026-12-01", "status": "issued", "files": {}})
    assert LC.verify_chain(LC.read_hashlog(log)) == []


def test_unknown_qualifier_masked():
    base = {"monitoring_location_id": "USGS-01", "parameter_code": "00300", "statistic_id": "00002",
            "time": "2026-12-01", "time_series_id": "t1", "approval_status": "Provisional"}
    long, unknown = LD.ogc_long([{**base, "value": 5.0, "qualifier": "['NEWCODE']"},
                                 {**base, "time": "2026-12-02", "value": 6.0, "qualifier": "['ICE']"}])
    wide, _ = LD.wide_by_site(long)
    assert unknown == {"NEWCODE": 1}
    assert "2026-12-01" not in wide["01"].index.strftime("%Y-%m-%d")
    assert "2026-12-02" in wide["01"].index.strftime("%Y-%m-%d")


def test_series_map_falls_back_when_mapped_series_stops():
    d = pd.date_range("2027-01-01", "2027-01-20")
    a = pd.DataFrame({"site_no": "01", "date": d[:5], "parm_cd": "00300", "stat_cd": "00002", "ts_id": "a",
                      "value": 5.0, "qualifier": "P"})            # mapped series stops on 01-05
    b = pd.DataFrame({"site_no": "01", "date": d, "parm_cd": "00300", "stat_cd": "00002", "ts_id": "b",
                      "value": 6.0, "qualifier": "P"})            # replacement sensor
    kept, info = LD.apply_series_map(pd.concat([a, b]), {"01|00300|00002": "a"}, window_end="2027-01-20")
    assert info["fallback_keys"] == ["01|00300|00002"] and set(kept["ts_id"]) == {"a", "b"}
    kept, info = LD.apply_series_map(pd.concat([a, b]), {"01|00300|00002": "b"}, window_end="2027-01-20")
    assert info["fallback_keys"] == [] and set(kept["ts_id"]) == {"b"}


def test_stamp_never_submits_without_freeze(monkeypatch):
    import stamp as ST

    def boom(*a, **k):
        raise AssertionError("external submission attempted")
    monkeypatch.setattr("requests.post", boom)
    monkeypatch.setattr(sys, "argv", ["stamp.py", "--mode", "dryrun"])
    assert ST.main() == 0
    assert "github" not in json.dumps(ST.__dict__.get("BACKENDS", {}))       # OTS only (review C-D1)


def test_evidence_refs_detect_deleted_proof(tmp_path, monkeypatch):
    """C-D1: evidence files are hash-committed in the next row; deleting one is detected."""
    import stamp as ST
    import verify_hashlog as VH
    monkeypatch.setattr(ST, "evidence_dir", lambda mode: tmp_path / "ts")
    (tmp_path / "ts").mkdir()
    (tmp_path / "ts" / "00000_abc.row.json.ots").write_bytes(b"proof")
    rows = [{"seq": 1, "evidence_refs": {"00000_abc.row.json.ots": LC.sha256_file(tmp_path / "ts" / "00000_abc.row.json.ots")}}]
    assert VH.check_evidence_refs(rows, "dryrun") == []
    (tmp_path / "ts" / "00000_abc.row.json.ots").unlink()
    assert VH.check_evidence_refs(rows, "dryrun")


def test_issuing_closure_excludes_scorer_and_matches_freeze():
    files = LC.code_files()
    assert "scripts/live/score_live.py" not in files and "scripts/live/build_live_config.py" not in files
    if LC.FREEZE_LIVE.is_file():                         # CI check after the freeze (review W-L13)
        assert json.loads(LC.FREEZE_LIVE.read_text())["code_sha256"] == LC.code_sha256()


def _cfg_and_modules():
    cfg, _ = LC.load_config()
    LD.SNAPSHOT_DIR["dir"] = Path(cfg["data_pins"]["sonde_daily_dir"])
    LD.SNAPSHOT_DIR["sha"] = cfg["data_pins"]["sonde_daily_site_sha256"]
    import issue as ISS
    return cfg, ISS, LC.sonde_events_module()


def test_span_beyond_wy2027_runs_end_to_end():
    """Review C-L2: a cube running past 2027-09-30 goes through make_data, issue_rows and score_rows."""
    cfg, ISS, se = _cfg_and_modules()
    sr = dict([(r["site_no"], r["role"]) for r in cfg["sites"]["H1_hypoxia"] if r["role"] == "train"][:3])
    days = pd.date_range("2027-11-20", "2028-01-14")
    rng = np.random.default_rng(0)
    frames = {s: pd.DataFrame({"do_mgl_min": rng.uniform(6, 9, len(days)).astype("float32"),
                               "do_mgl_mean": rng.uniform(8, 10, len(days)).astype("float32"),
                               "temp_c_mean": rng.uniform(2, 6, len(days)).astype("float32")}, index=days) for s in sr}
    data = ISS.make_data(cfg, "H1_hypoxia", sr, "2028-01-14", window=("2027-11-20", LC.LIVE_BLOCK[1]),
                         window_frames=frames, from_snapshot=False, weather_frames=[], log=lambda m: None)
    ev = ISS.issue_rows(data, "H1_hypoxia", 3)
    assert len(ev) > 0
    raw, final, thr, X = ISS.score_rows(se, data, "H1_hypoxia", 3, ev, cfg, lambda m: None)
    assert np.isfinite(final["stack"]).all()


def test_issue_rows_superset_of_frozen_event_rows_on_val():
    """Review W-L10: issue-time rows ⊇ frozen label-valid event rows at t (val block, approved data)."""
    cfg, ISS, se = _cfg_and_modules()
    from sentinel.models.sonde import events as E
    sr = dict([(r["site_no"], r["role"]) for r in cfg["sites"]["H1_hypoxia"] if r["role"] == "train"][:25])
    win = ("2022-10-01", "2023-09-30")
    full = ISS.make_data(cfg, "H1_hypoxia", sr, win[1], window=win, window_frames=None, from_snapshot=True,
                         weather_frames=[], log=lambda m: None)
    lab = E.hazard_labels(full, "H1_hypoxia", (3,))
    ev = E.event_rows(full, lab, 3, "live", roles=("train",))
    for t in ("2023-06-15", "2023-07-20", "2023-08-25"):
        want = set(ev.site_no[pd.to_datetime(ev.issue_date) == pd.Timestamp(t)])
        dt = ISS.make_data(cfg, "H1_hypoxia", sr, t, window=win, window_frames=None, from_snapshot=True,
                           weather_frames=[], log=lambda m: None)
        got = set(ISS.issue_rows(dt, "H1_hypoxia", 3).site_no)
        assert want <= got, (t, sorted(want - got))


# W-D1: qualifier handling derived from legacy agreement on val data (derive_qualifier_map.py) — table-driven
QUALIFIER_TABLE = {"ESTIMATED": "mask", "LESSTHAN": "mask", "GREATERTHAN": "mask", "INCOMPLETE": "mask",
                   "EQUIP": "mask", "MAINT": "mask", "MAINTENANCE": "mask", "BACKWATER": "mask", "FLOOD": "mask",
                   "PARTIAL": "mask", "FORCEINTERPOLATION": "keep", "REVISED": "keep", "ICE": "keep", "DRY": "keep",
                   "SEASONAL": "keep", "UNAVAIL": "keep", "DISCONTINUED": "keep", "SOMETHINGNEW": "mask"}


@pytest.mark.parametrize("q,decision", sorted(QUALIFIER_TABLE.items()))
def test_qualifier_table(q, decision):
    rec = {"monitoring_location_id": "USGS-01", "parameter_code": "00300", "statistic_id": "00002",
           "time": "2027-01-05", "time_series_id": "t1", "approval_status": "Approved", "value": 7.0,
           "qualifier": f"['{q}']"}
    long, _ = LD.ogc_long([rec])
    wide, _ = LD.wide_by_site(long)
    kept = "01" in wide and "2027-01-05" in wide["01"].index.strftime("%Y-%m-%d")
    assert kept == (decision == "keep"), q


def test_qualifier_map_matches_derivation():
    assert {k for k, v in QUALIFIER_TABLE.items() if v == "keep"} == set(LD.KEEP_OGC)
    assert {k for k, v in QUALIFIER_TABLE.items() if v == "mask" and k != "SOMETHINGNEW"} == set(LD.MASK_OGC)
    f = LC.DRYRUN_ROOT / "qualifier_map" / "derived.json"
    if f.is_file():                                      # every qualifier observed on val data: code == derivation
        for q, row in json.loads(f.read_text())["table"].items():
            assert QUALIFIER_TABLE[q] == row["decision"], q


def test_scorer_handles_zero_row_issue_day(tmp_path):
    """Review S-1: a zero-row issue day (no columns) is skipped by load_issues and accepted by the pre-claim test-load;
    a file with rows but missing key columns is refused before the claim."""
    import score_live as SL
    for d in ("2026-12-01", "2026-12-02", "2026-12-03"):
        (tmp_path / "issues" / d).mkdir(parents=True)
    pd.DataFrame().to_parquet(tmp_path / "issues" / "2026-12-01" / "issue.parquet")
    row = {c: [1] for c in SL.ISSUE_KEY_COLS}
    row.update({"issue_date": ["2026-12-02"], "data_through": ["2026-12-01"], "hazard": ["H1_hypoxia"], "site_no": ["x"],
                "role": ["train"], "basin_group": ["1"], "method": ["stack"]})
    pd.DataFrame(row).to_parquet(tmp_path / "issues" / "2026-12-02" / "issue.parquet")
    rows = [{"issue_date": d, "status": "issued", "on_time": True, "row_sha256": d} for d in ("2026-12-01", "2026-12-02")]
    assert SL.preload_issue_files(tmp_path, rows) == {"n_issue_files_with_rows": 1, "n_zero_row_issue_days": 1}
    df = SL.load_issues(tmp_path, rows, ("p",))
    assert list(df["issue_date"].unique()) == ["2026-12-02"]
    pd.DataFrame({"issue_date": ["2026-12-03"], "site_no": ["x"]}).to_parquet(tmp_path / "issues" / "2026-12-03" / "issue.parquet")
    with pytest.raises(PermissionError):
        SL.preload_issue_files(tmp_path, rows + [{"issue_date": "2026-12-03", "status": "issued", "row_sha256": "z"}])
