# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
import pytest

from sentinel.models.downstream import common as C
from sentinel.models.downstream import testlook as T


def test_single_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "marker_path", lambda app: tmp_path / app / "TEST_LOOK.json")
    monkeypatch.setattr(T, "registry_test_rows", lambda app: 0)
    rec = T.claim_test_look("demo_app", "abc")
    assert rec["freeze_sha256"] == "abc"
    with pytest.raises(C.ProtectedDataError):
        T.claim_test_look("demo_app", "abc")


def test_refuses_if_registry_has_test_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "marker_path", lambda app: tmp_path / app / "TEST_LOOK.json")
    monkeypatch.setattr(T, "registry_test_rows", lambda app: 3)
    with pytest.raises(C.ProtectedDataError):
        T.claim_test_look("demo_app", "abc")
