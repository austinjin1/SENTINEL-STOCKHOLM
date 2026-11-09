# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Deterministic seeding for Python, NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
