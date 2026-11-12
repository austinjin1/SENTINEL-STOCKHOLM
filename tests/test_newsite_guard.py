# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""newsite_guard enforces (review C-N3): purpose, lead freeze, OPEN claim of this process read from disk, block-list."""

import json
import os
import socket

import pytest

from sentinel.data import newsite_guard as G


def test_block_list_refuses_and_excludes():
    some = sorted(G.newsite_ids())[0]
    with pytest.raises(G.ProtectedNewSiteError):
        G.assert_not_newsite(["01646500", some])
    assert G.exclude_reserved(["01646500", some]) == ["01646500"]


def test_purpose_and_freeze_refused():
    with pytest.raises(G.ProtectedNewSiteError):
        G.open_site("x", "dv", look="ew_v3", purpose="train")
    if not G.FREEZES["ew_v3"].exists():
        with pytest.raises(G.ProtectedNewSiteError):
            G.open_site("x", "dv", look="ew_v3", purpose="score")


def test_claim_must_be_this_process_and_open(tmp_path, monkeypatch):
    monkeypatch.setitem(G.LOOK_DIRS, "ew_v3", tmp_path)
    with pytest.raises(G.ProtectedNewSiteError):
        G.open_claim("ew_v3")                                     # no claim
    (tmp_path / "CLAIM.json").write_text(json.dumps({"pid": os.getpid() + 1, "host": socket.gethostname()}))
    with pytest.raises(G.ProtectedNewSiteError):
        G.open_claim("ew_v3")                                     # someone else's claim
    (tmp_path / "CLAIM.json").write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
    assert G.open_claim("ew_v3")["pid"] == os.getpid()
    (tmp_path / "SCORED.json").write_text("{}")
    with pytest.raises(G.ProtectedNewSiteError):
        G.open_claim("ew_v3")                                     # sealed: access closed
