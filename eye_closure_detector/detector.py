from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple

import numpy as np

from .calibration import CalibrationResult
from .ear import EarResult, compute_ear
from .landmarks import DEFAULT_LEFT_EYE_IDX, DEFAULT_RIGHT_EYE_IDX


class EyeState(str, Enum):
    OPEN = "OPEN"
    BLINK = "BLINK"
    CLOSED = "CLOSED"
    NO_FACE = "NO_FACE"
    MULTI_FACE = "MULTI_FACE"


@dataclass
class DetectionResult:
    time_s: float
    ear: float
    left_ear: float
    right_ear: float
    state: EyeState
    face_count: int
    threshold: float
    calibrated: bool
    blink_ms: float


def _pick_face_index(landmarks_multi, image_w: int, image_h: int) -> int:
    if len(landmarks_multi) == 1:
        return 0

    def eye_distance_for_face(i: int) -> float:
        lm = landmarks_multi[i].landmark
        a = lm[DEFAULT_LEFT_EYE_IDX[0]]
        b = lm[DEFAULT_RIGHT_EYE_IDX[3]]
        ax, ay = a.x * image_w, a.y * image_h
        bx, by = b.x * image_w, b.y * image_h
        return float(np.hypot(ax - bx, ay - by))

    distances = [eye_distance_for_face(i) for i in range(len(landmarks_multi))]
    return int(np.argmax(distances))


def _extract_eye_points(
    face_landmarks,
    image_w: int,
    image_h: int,
    idx6: Tuple[int, int, int, int, int, int],
) -> np.ndarray:
    lm = face_landmarks.landmark
    pts = []
    for idx in idx6:
        p = lm[idx]
        pts.append((p.x * image_w, p.y * image_h))
    return np.asarray(pts, dtype=np.float64)


class EyeClosureDetector:
    def __init__(
        self,
        ear_threshold: float = 0.20,
        min_blink_ms: float = 80.0,
        max_blink_ms: float = 450.0,
        min_closed_ms: float = 650.0,
        smooth_alpha: float = 0.35,
        adapt_rate: float = 0.02,
        adapt_margin: float = 0.03,
    ):
        self.ear_threshold = float(ear_threshold)
        self.min_blink_ms = float(min_blink_ms)
        self.max_blink_ms = float(max_blink_ms)
        self.min_closed_ms = float(min_closed_ms)
        self.smooth_alpha = float(smooth_alpha)
        self.adapt_rate = float(adapt_rate)
        self.adapt_margin = float(adapt_margin)

        self._ear_smooth: Optional[float] = None
        self._below_since_s: Optional[float] = None
        self._last_state: EyeState = EyeState.NO_FACE
        self._calibration: Optional[CalibrationResult] = None
        self._open_ema: Optional[float] = None
        self._open_var_ema: Optional[float] = None

    def set_calibration(self, cal: CalibrationResult) -> None:
        self._calibration = cal
        self.ear_threshold = float(cal.closure_threshold)
        self._open_ema = float(cal.baseline_ear_mean)
        self._open_var_ema = float(max(1e-9, cal.baseline_ear_std ** 2))

    @property
    def calibration(self) -> Optional[CalibrationResult]:
        return self._calibration

    def update(
        self,
        *,
        time_s: float,
        face_mesh_result,
        image_size: Tuple[int, int],
    ) -> DetectionResult:
        image_w, image_h = image_size

        if not face_mesh_result.multi_face_landmarks:
            self._below_since_s = None
            self._last_state = EyeState.NO_FACE
            return DetectionResult(
                time_s=time_s,
                ear=float(self._ear_smooth or 0.0),
                left_ear=0.0,
                right_ear=0.0,
                state=EyeState.NO_FACE,
                face_count=0,
                threshold=self.ear_threshold,
                calibrated=self._calibration is not None,
                blink_ms=0.0,
            )

        face_count = len(face_mesh_result.multi_face_landmarks)
        face_idx = _pick_face_index(face_mesh_result.multi_face_landmarks, image_w, image_h)
        face_landmarks = face_mesh_result.multi_face_landmarks[face_idx]

        left_pts = _extract_eye_points(face_landmarks, image_w, image_h, DEFAULT_LEFT_EYE_IDX)
        right_pts = _extract_eye_points(face_landmarks, image_w, image_h, DEFAULT_RIGHT_EYE_IDX)

        left: EarResult = compute_ear(left_pts)
        right: EarResult = compute_ear(right_pts)
        if not (left.valid and right.valid):
            self._below_since_s = None
            self._last_state = EyeState.NO_FACE
            return DetectionResult(
                time_s=time_s,
                ear=float(self._ear_smooth or 0.0),
                left_ear=float(left.ear),
                right_ear=float(right.ear),
                state=EyeState.NO_FACE,
                face_count=face_count,
                threshold=self.ear_threshold,
                calibrated=self._calibration is not None,
                blink_ms=0.0,
            )

        ear_raw = float((left.ear + right.ear) / 2.0)

        if self._ear_smooth is None:
            ear = ear_raw
        else:
            a = self.smooth_alpha
            ear = a * ear_raw + (1.0 - a) * float(self._ear_smooth)
        self._ear_smooth = ear

        if self._open_ema is not None and self._open_var_ema is not None:
            if ear > (self.ear_threshold + self.adapt_margin):
                r = self.adapt_rate
                delta = ear - self._open_ema
                self._open_ema = (1.0 - r) * float(self._open_ema) + r * ear
                self._open_var_ema = (1.0 - r) * float(self._open_var_ema) + r * float(delta * delta)

                std = float(np.sqrt(max(1e-9, self._open_var_ema)))
                thr = float(self._open_ema - 2.5 * std)
                thr = float(np.clip(thr, 0.12, 0.40))
                self.ear_threshold = thr

        thr = self.ear_threshold
        below = ear < thr
        if below:
            if self._below_since_s is None:
                self._below_since_s = time_s
        else:
            self._below_since_s = None

        blink_ms = 0.0
        state = EyeState.OPEN
        if below and self._below_since_s is not None:
            duration_ms = (time_s - self._below_since_s) * 1000.0
            blink_ms = float(duration_ms)
            if duration_ms >= self.min_closed_ms:
                state = EyeState.CLOSED
            elif duration_ms >= self.min_blink_ms:
                state = EyeState.BLINK
            else:
                state = EyeState.OPEN

        if face_count > 1 and state != EyeState.NO_FACE:
            if state in (EyeState.OPEN, EyeState.BLINK, EyeState.CLOSED):
                pass
        self._last_state = state

        return DetectionResult(
            time_s=time_s,
            ear=float(ear),
            left_ear=float(left.ear),
            right_ear=float(right.ear),
            state=state,
            face_count=face_count,
            threshold=float(thr),
            calibrated=self._calibration is not None,
            blink_ms=float(blink_ms),
        )

