# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""New-site list builder: only the FINAL split (and the list's own files) are skipped when collecting site IDs that
other workstreams name (fix 2026-11-09: the DEV smoke-test split must be scanned)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("_bnl", ROOT / "scripts" / "data" / "build_newsite_list.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_only_final_split_is_skipped(tmp_path):
    b = _builder()
    (tmp_path / "sonde_split.json").write_text(json.dumps({"site_role": {"01000001": "train"}}))
    (tmp_path / "sonde_split_dev.json").write_text(json.dumps({"site_role": {"01000002": "train"}}))
    (tmp_path / "sonde_split_report.json").write_text(json.dumps({"x": "USGS-01000003"}))
    (tmp_path / "hydromet_track_n_stations.json").write_text(json.dumps({"role_of": {"WQP:USGS-0100004": "test"}}))
    (tmp_path / "sonde_newsites_list.json").write_text(json.dumps({"sites": [{"site_no": "01000005"}]}))
    used, used_by = b.other_preregistration_ids(tmp_path)
    assert "01000001" not in used                         # final split: handled separately, skipped
    assert "01000002" in used                             # DEV split: scanned (the 2026-11-09 oversight)
    assert "01000003" in used                             # other sonde_split* files: scanned
    assert "0100004" in used                              # WQP:USGS-<7 digits> form matched
    assert "01000005" not in used                         # the list itself: skipped
    assert used_by["01000002"] == ["sonde_split_dev.json"]


def test_skip_set_is_exactly_final_split_and_own_files():
    assert set(_builder().SKIP_FILES) == {"sonde_split.json", "sonde_newsites_list.json", "sonde_newsites_overlap.json"}
