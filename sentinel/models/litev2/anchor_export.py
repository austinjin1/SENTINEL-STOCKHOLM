# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Compact float32 file of the anchor trees (amendment §4.6): "Size = INT8 ONNX file + the anchor file (compact
float32 format written by our exporter; parity with LightGBM within 1e-4 log10 is a unit test)" and "Parameters =
network parameters + all tree thresholds and leaf values".

The anchor is the S6 model set of ``tabular.fit_parity`` (pooled): four per-target models and one pooled turbidity
model, three seeds each. One file holds every tree of every seed; a prediction is the sum of the tree outputs of a
group divided by the number of seeds (LightGBM folds the starting average into the first tree).

File layout (little endian)::

    b"SLANCH01" | uint32 header length | header (JSON, utf-8)
    uint8   n_internal[n_trees]          internal nodes per tree (leaves = n_internal + 1)
    uint8   feature[n_nodes]             column index in the group's feature list
    int8    left[n_nodes], right[n_nodes]   child >= 0: internal node of the tree; child < 0: leaf ~child
    uint8   flags[n_nodes]               bit 0 = default left; bits 1-2 = missing type (0 none, 1 zero, 2 NaN)
    float32 threshold[n_nodes]
    float32 leaf[n_leaves]

Readings decided here (for the lead's notes):
* a threshold is stored as the largest float32 that is not above the LightGBM threshold, so for a float32 feature
  value the decision ``x <= threshold`` is exactly LightGBM's;
* the deployable pipeline computes its features in float32, so the parity reference of the unit test is LightGBM on
  the same float32 feature values; the difference between this file and the float64-feature anchors used in
  training is reported with the export, like the INT8-vs-FP32 difference;
* the count of "tree thresholds and leaf values" is ``n_nodes + n_leaves`` of this file.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import tabular as T

MAGIC = b"SLANCH01"
MISSING = {"None": 0, "Zero": 1, "NaN": 2}
ZERO_THRESHOLD = 1e-35                                           # LightGBM kZeroThreshold
GROUPS = (*T.SINGLE, "turb_pooled")


class AnchorExportError(RuntimeError):
    pass


def floor32(x) -> np.ndarray:
    """Largest float32 <= x, elementwise (x float64)."""
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore"):
        f = x.astype(np.float32)
    up = f.astype(np.float64) > x
    f[up] = np.nextafter(f[up], np.float32(-np.inf))
    return f


def _tree_arrays(tree: dict) -> dict:
    """One LightGBM tree (``dump_model`` structure) as flat arrays."""
    n_leaves = int(tree["num_leaves"])
    n_int = n_leaves - 1
    if n_leaves > 128:
        raise AnchorExportError("more than 128 leaves do not fit the int8 child index")
    a = {"feature": np.zeros(n_int, np.int64), "left": np.zeros(n_int, np.int64), "right": np.zeros(n_int, np.int64),
         "flags": np.zeros(n_int, np.uint8), "threshold": np.zeros(n_int, np.float64),
         "leaf": np.zeros(n_leaves, np.float64)}
    seen_leaf, seen_node = np.zeros(n_leaves, bool), np.zeros(n_int, bool)

    def ref(node: dict) -> int:
        if "leaf_value" in node and "split_index" not in node:
            k = int(node.get("leaf_index", 0))
            a["leaf"][k] = float(node["leaf_value"])
            seen_leaf[k] = True
            return ~k
        i = int(node["split_index"])
        if node["decision_type"] != "<=":
            raise AnchorExportError(f"unsupported decision type {node['decision_type']!r} (categorical split?)")
        if node["missing_type"] not in MISSING:
            raise AnchorExportError(f"unknown missing type {node['missing_type']!r}")
        a["feature"][i] = int(node["split_feature"])
        a["threshold"][i] = float(node["threshold"])
        a["flags"][i] = (1 if node["default_left"] else 0) | (MISSING[node["missing_type"]] << 1)
        seen_node[i] = True
        a["left"][i], a["right"][i] = ref(node["left_child"]), ref(node["right_child"])
        return i

    root = ref(tree["tree_structure"])
    if not (seen_leaf.all() and seen_node.all()) or (n_int and root != 0):
        raise AnchorExportError("tree structure does not match its leaf count")
    return a


def _booster(m):
    return m.booster_ if hasattr(m, "booster_") else m


def export_anchor(model: dict, path: str | Path) -> dict:
    """Write the anchor model set (``tabular.fit_parity(..., pooled=True)``) to ``path``; returns the header."""
    if not model.get("pooled"):
        raise AnchorExportError("the anchor is the pooled model set (S6)")
    cols, tcols = list(model["cols"]), list(model["turb_cols"]) + list(T.UNIT_COLS)
    if max(len(cols), len(tcols)) > 256:
        raise AnchorExportError("more than 256 features do not fit the uint8 feature index")
    groups, parts = [], {k: [] for k in ("n_internal", "feature", "left", "right", "flags", "threshold", "leaf")}
    for g in GROUPS:
        seeds = model["models"].get(g) or []
        if not seeds:
            raise AnchorExportError(f"no fitted model for {g!r}")
        n_feat = len(tcols) if g == "turb_pooled" else len(cols)
        n_trees = 0
        for m in seeds:
            d = _booster(m).dump_model()
            if d.get("objective", "regression").split()[0] != "regression" or int(d.get("num_class", 1)) != 1:
                raise AnchorExportError(f"{g}: not a plain regression model")
            if int(d["max_feature_idx"]) + 1 != n_feat:
                raise AnchorExportError(f"{g}: the model has {int(d['max_feature_idx']) + 1} features, expected {n_feat}")
            if d.get("average_output"):
                raise AnchorExportError(f"{g}: averaged output (random forest mode) is not supported")
            for tr in d["tree_info"]:
                a = _tree_arrays(tr)
                parts["n_internal"].append(np.array([len(a["feature"])], np.int64))
                for k in ("feature", "left", "right", "flags", "threshold", "leaf"):
                    parts[k].append(a[k])
                n_trees += 1
        groups.append({"name": g, "features": "turb" if g == "turb_pooled" else "cols", "n_seeds": len(seeds),
                       "n_trees": n_trees})
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    header = {"format": MAGIC.decode(), "cols": cols, "turb_cols": tcols, "groups": groups,
              "n_trees": int(len(cat["n_internal"])), "n_nodes": int(len(cat["feature"])),
              "n_leaves": int(len(cat["leaf"])), "n_numbers": int(len(cat["feature"]) + len(cat["leaf"])),
              "units": {u: list(v) for u, v in T.TURB_UNITS.items()}}
    hb = json.dumps(header, sort_keys=True).encode()
    with np.errstate(over="ignore"):
        leaf32 = cat["leaf"].astype(np.float32)
    if not np.isfinite(leaf32).all():
        raise AnchorExportError("a leaf value is not finite in float32")
    path = Path(path)
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(hb)) + hb)
        f.write(cat["n_internal"].astype(np.uint8).tobytes())
        f.write(cat["feature"].astype(np.uint8).tobytes())
        f.write(cat["left"].astype(np.int8).tobytes())
        f.write(cat["right"].astype(np.int8).tobytes())
        f.write(cat["flags"].astype(np.uint8).tobytes())
        f.write(floor32(cat["threshold"]).astype("<f4").tobytes())
        f.write(leaf32.astype("<f4").tobytes())
    return {**header, "bytes": path.stat().st_size}


@dataclass
class AnchorFile:
    header: dict
    groups: dict                        # name -> padded arrays of the group's trees

    @property
    def n_numbers(self) -> int:
        return int(self.header["n_numbers"])


def load_anchor(path: str | Path) -> AnchorFile:
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise AnchorExportError("not an anchor file")
    (hl,) = struct.unpack("<I", raw[8:12])
    header = json.loads(raw[12:12 + hl].decode())
    nt, nn, nl = header["n_trees"], header["n_nodes"], header["n_leaves"]
    o = 12 + hl

    def take(dtype, n):
        nonlocal o
        if o + n * np.dtype(dtype).itemsize > len(raw):
            raise AnchorExportError("the anchor file is truncated or inconsistent")
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=o)
        o += a.nbytes
        return a

    n_int = take(np.uint8, nt).astype(np.int64)
    feature, left, right, flags = take(np.uint8, nn), take(np.int8, nn), take(np.int8, nn), take(np.uint8, nn)
    thr, leaf = take("<f4", nn), take("<f4", nl)
    if o != len(raw) or int(n_int.sum()) != nn or int((n_int + 1).sum()) != nl:
        raise AnchorExportError("the anchor file is truncated or inconsistent")
    node0 = np.concatenate([[0], np.cumsum(n_int)])
    leaf0 = np.concatenate([[0], np.cumsum(n_int + 1)])
    groups, t0 = {}, 0
    for g in header["groups"]:
        k = int(g["n_trees"])
        ni = n_int[t0:t0 + k]
        W = max(int(ni.max()), 1)
        P = {"feature": np.zeros((k, W), np.int64), "left": np.full((k, W), -1, np.int64),
             "right": np.full((k, W), -1, np.int64), "flags": np.zeros((k, W), np.uint8),
             "threshold": np.zeros((k, W), np.float32), "leaf": np.zeros((k, W + 1), np.float32),
             "start": np.where(ni > 0, 0, -1).astype(np.int64), "n_seeds": int(g["n_seeds"]),
             "features": g["features"]}
        for j in range(k):
            a, b = node0[t0 + j], node0[t0 + j + 1]
            P["feature"][j, :b - a], P["left"][j, :b - a], P["right"][j, :b - a] = feature[a:b], left[a:b], right[a:b]
            P["flags"][j, :b - a], P["threshold"][j, :b - a] = flags[a:b], thr[a:b]
            P["leaf"][j, :ni[j] + 1] = leaf[leaf0[t0 + j]:leaf0[t0 + j + 1]]
        groups[g["name"]] = P
        t0 += k
    return AnchorFile(header, groups)


def _predict_group(P: dict, X32: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Sum of the tree outputs of one group / number of seeds, for float32 features (N, F)."""
    k = P["feature"].shape[0]
    tix = np.arange(k)[None, :]
    out = np.zeros(len(X32), np.float64)
    for s in range(0, len(X32), chunk):
        x = X32[s:s + chunk]
        n = len(x)
        node = np.broadcast_to(P["start"][None, :], (n, k)).copy()
        rix = np.arange(n)[:, None]
        for _ in range(P["feature"].shape[1] + 1):
            act = node >= 0
            if not act.any():
                break
            nd = np.where(act, node, 0)
            v = x[rix, P["feature"][tix, nd]]
            fl = P["flags"][tix, nd]
            mt, dl = fl >> 1, (fl & 1).astype(bool)
            nan = np.isnan(v)
            v = np.where(nan & (mt != 2), np.float32(0.0), v)                     # as LightGBM's NumericalDecision
            default = ((mt == 1) & (np.abs(v) <= ZERO_THRESHOLD)) | ((mt == 2) & nan)
            with np.errstate(invalid="ignore"):
                go_left = np.where(default, dl, v <= P["threshold"][tix, nd])
            node = np.where(act, np.where(go_left, P["left"][tix, nd], P["right"][tix, nd]), node)
        if (node >= 0).any():
            raise AnchorExportError("a tree walk did not reach a leaf")
        out[s:s + n] = P["leaf"][tix, ~node].sum(1, dtype=np.float64) / P["n_seeds"]
    return out


def predict_anchor(af: AnchorFile, X: pd.DataFrame) -> pd.DataFrame:
    """Anchors (log10) of :data:`tabular.QUANTITIES` from the file alone. Features are cast to float32 first."""
    h = af.header
    base_t = h["turb_cols"][:len(h["turb_cols"]) - len(T.UNIT_COLS)]
    Xa = X[h["cols"]].to_numpy(np.float32)
    Xt = X[base_t].to_numpy(np.float32)
    out = {t: _predict_group(af.groups[t], Xa) for t in T.SINGLE}
    for u, ind in h["units"].items():
        xi = np.column_stack([Xt, np.tile(np.asarray(ind, np.float32), (len(Xt), 1))])
        out[u] = _predict_group(af.groups["turb_pooled"], xi)
    return pd.DataFrame(out, index=X.index)[list(T.QUANTITIES)]


__all__ = ["MAGIC", "GROUPS", "AnchorExportError", "AnchorFile", "floor32", "export_anchor", "load_anchor",
           "predict_anchor"]
