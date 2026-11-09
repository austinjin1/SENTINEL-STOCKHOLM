# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""ONNX + INT8 export and desktop-proxy latency for SENTINEL-Lite (protocol §9; research 08 §d.5).

INT8 = onnxruntime ``quant_pre_process`` then static QDQ quantisation (per-channel weights, QInt8 weights,
QUInt8 activations) calibrated on train-role chips. Latency is the **desktop proxy** (ORT CPU EP, 1 intra-op
thread, sequential, batch 1, 512×512×4), never device latency.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .model import ExportWrapper, LiteStudent


def export_onnx(model: LiteStudent, path: str | Path, size: int = 512, opset: int = 17,
                scene_mode: bool = False, rgb_only: bool = False) -> Path:
    w = ExportWrapper(model.eval().cpu(), scene_mode=scene_mode, rgb_only=rgb_only).eval()
    x = torch.rand(1, 4, size, size)
    torch.onnx.export(w, (x,), str(path), input_names=["x01"], output_names=["maps"], opset_version=opset,
                      dynamic_axes={"x01": {0: "n", 2: "h", 3: "w"}, "maps": {0: "n", 2: "h4", 3: "w4"}},
                      do_constant_folding=True, dynamo=False)
    return Path(path)


class _Calib:
    def __init__(self, tiles: np.ndarray):
        self.tiles = list(tiles)

    def get_next(self):
        if not self.tiles:
            return None
        return {"x01": self.tiles.pop(0)[None].astype(np.float32)}

    def rewind(self):  # pragma: no cover - API completeness
        pass


def calibration_tiles(chips_u8: np.ndarray, n: int = 32, size: int = 512, seed: int = 0) -> np.ndarray:
    """2×2 mosaics of random train chips -> (n, 4, 512, 512) in [0, 1]."""
    rs = np.random.default_rng(seed)
    k = size // chips_u8.shape[-1]
    out = np.zeros((n, 4, size, size), np.float32)
    for i in range(n):
        idx = rs.choice(len(chips_u8), k * k, replace=False)
        rows = [np.concatenate([chips_u8[idx[r * k + c]] for c in range(k)], axis=-1) for r in range(k)]
        out[i] = np.concatenate(rows, axis=-2) / 255.0
    return out


def quantise_int8(fp32: str | Path, int8: str | Path, calib: np.ndarray) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = Path(str(fp32).replace(".onnx", ".pre.onnx"))
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=False)
    quantize_static(str(pre), str(int8), _Calib(calib), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    return Path(int8)


def session(path, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def latency(path, *, threads: int = 1, size: int = 512, warmup: int = 5, runs: int = 30) -> dict:
    s = session(path, threads)
    x = np.random.default_rng(0).random((1, 4, size, size), dtype=np.float32)
    for _ in range(warmup):
        s.run(None, {"x01": x})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {"x01": x})
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "runs": runs,
            "threads": threads, "size": size, "label": "desktop proxy (ONNX Runtime CPU EP), not device latency"}


def cpu_info() -> dict:
    info = {"machine": platform.machine(), "processor": platform.processor()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        info["governor"] = gov.read_text().strip() if gov.exists() else "unknown"
        info["loadavg"] = os.getloadavg()
    except OSError:
        pass
    import onnxruntime

    info["onnxruntime"] = onnxruntime.__version__
    info["torch"] = torch.__version__
    return info


def onnx_predict(path, chips_u8: np.ndarray, pool_w: np.ndarray, n_targets: int, threads: int = 6,
                 batch: int = 16, scene_params: np.ndarray | None = None) -> np.ndarray:
    """Chip-level log10 μ from an ONNX model (pooled exactly like the torch path). ``scene_params`` (N,4,2)
    switches to scene-normalised inputs (x = (DN - centre) / scale)."""
    s = session(path, threads)
    out = []
    for i in range(0, len(chips_u8), batch):
        x = chips_u8[i:i + batch].astype(np.float32)
        if scene_params is None:
            x = x / 255.0
        else:
            sp = scene_params[i:i + batch]
            x = (x - sp[:, :, 0, None, None]) / sp[:, :, 1, None, None]
        m = s.run(None, {"x01": x.astype(np.float32)})[0]
        out.append((m[:, :n_targets] * pool_w[i:i + batch, None]).sum((2, 3)))
    return np.concatenate(out)
