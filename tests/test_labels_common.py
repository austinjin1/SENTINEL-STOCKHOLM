# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Tests for the shared label-client helpers (sentinel/data/sources/events_common.py).

No network: paging and retry logic are exercised with monkeypatched fetchers. The tiny
in-memory XLSX built below is a synthetic test object only (never a dataset).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import events_common as C

FIX = Path(__file__).parent / "fixtures"


# ----------------------------------------------------------------------------- dates / coords / ids

def test_clean_dates_mixed_formats_and_calendar_errors():
    s = pd.Series(["2020-01-05", "01/02/2021", "2555-07-01", "", "junk", "0023-06-08",
                   "2021-06-01T10:00:00-05:00", "1976-07-04"])
    d, f = C.clean_dates(s, min_date="1990-01-01", max_date="2026-12-31")
    assert d[0] == pd.Timestamp("2020-01-05") and d[1] == pd.Timestamp("2021-01-02")
    assert f.tolist() == ["", "", "out_of_range", "missing", "unparseable", "out_of_range", "",
                          "out_of_range"]
    assert d[6] == pd.Timestamp("2021-06-01 15:00:00")  # offset converted to UTC, naive
    assert d[[2, 3, 4, 5, 7]].isna().all()


def test_clean_latlon_flags_and_swap_fix():
    lat = pd.Series([41.5, None, 0, 95, -93.6, 10.0])
    lon = pd.Series([-93.6, -93.0, 0, -93, 41.5, 10.0])
    la, lo, fl = C.clean_latlon(lat, lon, bbox=(-96.85, 40.17, -89.94, 43.71))
    assert fl.tolist() == ["", "missing", "zero", "invalid", "swapped_fixed", "outside_bbox"]
    assert la[4] == 41.5 and lo[4] == -93.6
    assert la[[1, 2, 3, 5]].isna().all() and lo[[1, 2, 3, 5]].isna().all()


def test_str_id_keeps_leading_zeros_and_strips_float_suffix():
    out = C.str_id(pd.Series([7090002.0, "05427850", None, "nan", " 12 ", ""]), width=8)
    assert out.tolist()[:2] == ["07090002", "05427850"]
    assert out.tolist()[4] == "00000012"
    assert out.isna().tolist() == [False, False, True, True, False, True]
    assert str(out.dtype) == "string"


def test_to_number_masks_sentinels_and_keeps_qualifier():
    v, q = C.to_number(pd.Series(["1.5", "-999999", "ND", "<0.3", "1,234", None, "-9999"]))
    assert v[0] == 1.5 and v[4] == 1234.0
    assert np.isnan(v[[1, 2, 3, 5, 6]]).all()
    assert q.tolist() == ["", "-999999", "ND", "<0.3", "", "", "-9999"]


# ----------------------------------------------------------------------------- HTML

def test_html_tables_links_and_nested_markup():
    html = ("<p>x</p><table><tr><th>SPECIES</th><th>NUMBER<br/>KILLED</th></tr>"
            "<tr><td><p><span>Sunfish</span></p></td><td>1,231</td></tr>"
            "<tr><td>Rock&nbsp;Bass</td><td> 46 </td></tr></table>"
            "<a href='Data.aspx?SiteID=75&amp;x=1'>Lake  A</a>")
    t = C.html_tables(html)
    assert t == [[["SPECIES", "NUMBER KILLED"], ["Sunfish", "1,231"], ["Rock Bass", "46"]]]
    df = C.table_to_frame(t[0])
    assert list(df.columns) == ["SPECIES", "NUMBER KILLED"] and len(df) == 2
    assert C.html_links(html) == [("Data.aspx?SiteID=75&x=1", "Lake A")]


# ----------------------------------------------------------------------------- XLSX

def _mini_xlsx() -> bytes:
    """Synthetic 1-sheet workbook: shared + inline strings, bool, date style, a gap column."""
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    files = {
        "xl/workbook.xml": f'<workbook {ns} {rns}><sheets><sheet name="S1" sheetId="1" r:id="rId1"/>'
                           "</sheets></workbook>",
        "xl/_rels/workbook.xml.rels":
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
            "</Relationships>",
        "xl/sharedStrings.xml": f'<sst {ns}><si><t>name</t></si><si><t>when</t></si>'
                                "<si><r><t>Lake </t></r><r><t>A</t></r></si></sst>",
        "xl/styles.xml": f'<styleSheet {ns}><cellXfs count="2"><xf numFmtId="0"/><xf numFmtId="14"/>'
                         "</cellXfs></styleSheet>",
        "xl/worksheets/sheet1.xml":
            f'<worksheet {ns}><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
            '<c r="D1" t="inlineStr"><is><t>flag</t></is></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2" s="1"><v>45657</v></c>'
            '<c r="C2"><v>1.1000000000000001</v></c><c r="D2" t="b"><v>1</v></c></row>'
            "</sheetData></worksheet>",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_read_xlsx_synthetic_cell_types():
    b = _mini_xlsx()
    assert C.xlsx_sheet_names(b) == ["S1"]
    df = C.read_xlsx(b)
    assert list(df.columns) == ["name", "when", "col_2", "flag"]
    assert df.iloc[0].tolist() == ["Lake A", "2024-12-31", "1.1", "TRUE"]


def test_read_xlsx_real_noaa_fixture_dates_and_numbers():
    p = FIX / "labels_noaa_lake_erie_hab" / "curated_2000to2025.xlsx"
    df = C.read_xlsx(p)
    assert df.shape == (26, 7)
    assert df.loc[df.year == "2015", "severity_index"].item() == "8.4"
    assert df.loc[df.year == "2000", "total_CI_max"].item() == "1.1"  # not 1.1000000000000001
    assert df.loc[df.year == "2011", "Date_max"].item() == "2011-10-13"


# ----------------------------------------------------------------------------- pagers (no network)

def test_socrata_csv_pages_until_short_page(monkeypatch):
    rows = [f"{i},x{i}" for i in range(5)]
    calls = []

    def fake_get_text(url, params=None, **kw):
        calls.append(params["$offset"])
        chunk = rows[params["$offset"]: params["$offset"] + params["$limit"]]
        return "id,name\n" + "\n".join(chunk) + ("\n" if chunk else "")

    monkeypatch.setattr(C, "get_text", fake_get_text)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    df = C.socrata_csv("example.org", "abcd-1234", page_size=2)
    assert calls == [0, 2, 4] and len(df) == 5 and df["id"].tolist() == ["0", "1", "2", "3", "4"]


def test_arcgis_query_paging_with_transfer_limit(monkeypatch):
    feats = [{"attributes": {"OBJECTID": i, "v": i * 10}} for i in range(1, 6)]

    def fake_get_json(url, params=None, **kw):
        if url.endswith("/query"):
            o, n = params["resultOffset"], params["resultRecordCount"]
            batch = feats[o:o + n]
            return {"features": batch, "exceededTransferLimit": o + n < len(feats)}
        return {"maxRecordCount": 2, "objectIdField": "OBJECTID",
                "advancedQueryCapabilities": {"supportsPagination": True}}

    monkeypatch.setattr(C, "get_json", fake_get_json)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    out = C.arcgis_query("https://x/FeatureServer/0", page_size=10)
    assert [f["attributes"]["OBJECTID"] for f in out] == [1, 2, 3, 4, 5]
    assert C.esri_attributes_frame(out)["v"].tolist() == [10, 20, 30, 40, 50]


def test_ckan_datastore_paging_uses_total(monkeypatch):
    recs = [{"_id": i} for i in range(7)]

    def fake_get_json(url, params=None, **kw):
        o, n = params["offset"], params["limit"]
        return {"success": True, "result": {"total": 7, "records": recs[o:o + n], "fields": [{"id": "_id"}]}}

    monkeypatch.setattr(C, "get_json", fake_get_json)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    out, fields = C.ckan_datastore_rows("https://data.example", "rid", page_size=3)
    assert [r["_id"] for r in out] == list(range(7)) and fields == [{"id": "_id"}]


class _Resp:
    def __init__(self, code, headers=None, text=""):
        self.status_code, self.headers, self.text, self.url = code, headers or {}, text, "u"


def test_http_get_retries_429_then_succeeds_and_honours_retry_after(monkeypatch):
    seq = [_Resp(429, {"Retry-After": "7"}), _Resp(503), _Resp(200, text="ok")]
    waits = []
    monkeypatch.setattr(C.session(), "get", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(C.time, "sleep", lambda s: waits.append(s))
    r = C.http_get("https://example.org/x", backoff=1.0)
    assert r.text == "ok" and waits[0] == 7.0 and len(waits) == 2


def test_http_get_non_retryable_raises_and_404_optional(monkeypatch):
    monkeypatch.setattr(C.session(), "get", lambda *a, **k: _Resp(404))
    assert C.http_get("https://example.org/x", allow_404=True) is None
    monkeypatch.setattr(C.session(), "get", lambda *a, **k: _Resp(403, text="forbidden"))
    with pytest.raises(C.HTTPFailure):
        C.http_get("https://example.org/x")


def test_user_agent_is_generic():
    assert C.session().headers["User-Agent"] == "SENTINEL-v2 research (public-data download)"


def test_html_tables_colspan_expansion_is_opt_in():
    html = "<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td colspan='2'>x</td><td>1</td></tr></table>"
    assert C.html_tables(html)[0][1] == ["x", "1"]
    assert C.html_tables(html, expand_colspan=True)[0][1] == ["x", "", "1"]
