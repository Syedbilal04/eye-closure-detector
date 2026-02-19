from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    baseline_ear_mean: float
    baseline_ear_std: float
    closure_threshold: float
    samples: int


class EarCalibrator:
    def __init__(
        self,
        k_std: float = 2.5,
        min_threshold: float = 0.12,
        max_threshold: float = 0.40,
    ):
        self.k_std = float(k_std)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self._samples: List[float] = []

    def add(self, ear: float) -> None:
        if np.isfinite(ear) and ear > 0:
            self._samples.append(float(ear))

    @property
    def samples(self) -> int:
        return len(self._samples)

    def compute(self) -> Optional[CalibrationResult]:
        if len(self._samples) < 15:
            return None
        arr = np.asarray(self._samples, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        thr = mean - self.k_std * std
        thr = float(np.clip(thr, self.min_threshold, self.max_threshold))
        return CalibrationResult(
            baseline_ear_mean=mean,
            baseline_ear_std=std,
            closure_threshold=thr,
            samples=len(self._samples),
        )

