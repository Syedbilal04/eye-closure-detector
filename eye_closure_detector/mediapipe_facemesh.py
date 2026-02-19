from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from .model_assets import ModelAsset, ensure_model


@dataclass(frozen=True)
class FaceMeshConfig:
    max_num_faces: int = 3
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    model_path: Optional[Path] = None
    download_model: bool = True


@dataclass(frozen=True)
class _Landmark:
    x: float
    y: float


@dataclass(frozen=True)
class _FaceLandmarks:
    landmark: List[_Landmark]


@dataclass(frozen=True)
class FaceMeshResult:
    multi_face_landmarks: List[_FaceLandmarks]


class FaceMeshRunner:
    def __init__(self, config: FaceMeshConfig = FaceMeshConfig()):
        model_path = config.model_path or (Path("models") / "face_landmarker.task")
        if config.download_model:
            model_path = ensure_model(ModelAsset(path=model_path))

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=int(config.max_num_faces),
            min_face_detection_confidence=float(config.min_detection_confidence),
            min_face_presence_confidence=float(config.min_detection_confidence),
            min_tracking_confidence=float(config.min_tracking_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, *, timestamp_ms: int) -> FaceMeshResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        faces: List[_FaceLandmarks] = []
        for face in (res.face_landmarks or []):
            faces.append(_FaceLandmarks(landmark=[_Landmark(x=float(p.x), y=float(p.y)) for p in face]))
        return FaceMeshResult(multi_face_landmarks=faces)

