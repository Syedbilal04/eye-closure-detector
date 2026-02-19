from __future__ import annotations

import argparse
import time

import cv2

from .detector import EyeClosureDetector
from .mediapipe_facemesh import FaceMeshConfig, FaceMeshRunner


def main():
    p = argparse.ArgumentParser(description="Benchmark FaceMesh+EAR pipeline FPS")
    p.add_argument("--source", default="0")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--max-num-faces", type=int, default=1)
    p.add_argument("--process-scale", type=float, default=0.5)
    args = p.parse_args()

    if str(args.source).isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(str(args.source))

    mesh = FaceMeshRunner(FaceMeshConfig(max_num_faces=int(args.max_num_faces)))
    det = EyeClosureDetector()
    t0 = time.perf_counter()
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            now = time.perf_counter()
            if now - t0 >= float(args.seconds):
                break
            proc = frame
            s = float(args.process_scale)
            if 0.05 < s < 1.0:
                proc = cv2.resize(frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            h, w = proc.shape[:2]
            res = mesh.process(proc, timestamp_ms=int((now - t0) * 1000.0))
            det.update(time_s=now - t0, face_mesh_result=res, image_size=(w, h))
    finally:
        cap.release()
        mesh.close()

    elapsed = max(1e-9, time.perf_counter() - t0)
    fps = frames / elapsed
    print(f"frames={frames} elapsed_s={elapsed:.3f} fps={fps:.2f}")


if __name__ == "__main__":
    main()

