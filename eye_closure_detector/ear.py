from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EarResult:
    ear: float
    valid: bool


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear(points6: Iterable[Iterable[float]]) -> EarResult:
    pts = np.asarray(list(points6), dtype=np.float64)
    if pts.shape != (6, 2):
        raise ValueError(f"Expected (6,2) points, got {pts.shape}")

    p1, p2, p3, p4, p5, p6 = pts
    denom = 2.0 * _l2(p1, p4)
    if denom <= 1e-12:
        return EarResult(ear=0.0, valid=False)

    ear = (_l2(p2, p6) + _l2(p3, p5)) / denom
    if not np.isfinite(ear):
        return EarResult(ear=0.0, valid=False)
    return EarResult(ear=float(ear), valid=True)

