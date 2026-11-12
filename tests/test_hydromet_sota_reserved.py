# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M6 must never train on the M1-reserved new-sites test block (lead notice 2026-11-09)."""

import pytest

from sentinel.hydromet_sota import reserved as R


def test_pinned_list_loads_and_has_729_sites():
    s = R.reserved_usgs_sites()
    assert len(s) == 729 and all(isinstance(x, str) for x in s)


def test_exclude_and_assert():
    res = sorted(R.reserved_usgs_sites())
    cand = ["00000001", res[0], "USGS-" + res[1], "WQP-ABC"]
    keep = R.exclude_reserved(cand, log=lambda *_: None)
    assert keep == ["00000001", "WQP-ABC"]
    R.assert_no_reserved(keep)
    with pytest.raises(R.ReservedSiteError):
        R.assert_no_reserved(keep + [res[2]])
    with pytest.raises(R.ReservedSiteError):
        R.assert_no_reserved(["USGS-" + res[3]])


def test_changed_list_is_refused(tmp_path):
    f = tmp_path / "list.json"
    f.write_text('{"sites": []}')
    with pytest.raises(R.ReservedSiteError, match="changed"):
        R.reserved_usgs_sites(path=f)
