from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import urllib.request


DEFAULT_FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass(frozen=True)
class ModelAsset:
    path: Path
    url: str = DEFAULT_FACE_LANDMARKER_URL


def ensure_model(asset: ModelAsset, *, timeout_s: float = 30.0) -> Path:
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    if asset.path.exists() and asset.path.stat().st_size > 0:
        return asset.path

    tmp = asset.path.with_suffix(asset.path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with urllib.request.urlopen(asset.url, timeout=timeout_s) as r:
        data = r.read()
    tmp.write_bytes(data)
    tmp.replace(asset.path)
    return asset.path

