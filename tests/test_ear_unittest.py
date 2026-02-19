import os
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from eye_closure_detector.detector import EyeClosureDetector
from eye_closure_detector.ear import compute_ear
from eye_closure_detector.mediapipe_facemesh import FaceMeshRunner


class TestEar(unittest.TestCase):
    def test_ear_basic_geometry(self):
        pts = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [4.0, 0.0],
                [2.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=np.float64,
        )
        res = compute_ear(pts)
        self.assertTrue(res.valid)
        self.assertLess(abs(res.ear - 0.5), 1e-9)

    def test_ear_invalid_when_horizontal_zero(self):
        pts = np.zeros((6, 2), dtype=np.float64)
        res = compute_ear(pts)
        self.assertFalse(res.valid)


class TestIntegration(unittest.TestCase):
    def test_pipeline_runs_on_video_if_provided(self):
        video = os.environ.get("EYE_VIDEO_PATH")
        if not video:
            self.skipTest("Set EYE_VIDEO_PATH to run this integration test")

        path = Path(video)
        self.assertTrue(path.exists())

        cap = cv2.VideoCapture(str(path))
        mesh = FaceMeshRunner()
        det = EyeClosureDetector()
        frames = 0
        try:
            while frames < 60:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                h, w = frame.shape[:2]
                res = mesh.process(frame)
                out = det.update(time_s=frames / 30.0, face_mesh_result=res, image_size=(w, h))
                self.assertGreater(out.threshold, 0)
        finally:
            cap.release()
            mesh.close()

        self.assertGreater(frames, 0)

    def test_minimum_fps_smoke(self):
        video = os.environ.get("EYE_VIDEO_PATH")
        if not video:
            self.skipTest("Set EYE_VIDEO_PATH to run this performance smoke test")

        cap = cv2.VideoCapture(str(video))
        mesh = FaceMeshRunner()
        det = EyeClosureDetector()

        frames = 0
        t0 = time.perf_counter()
        try:
            while frames < 120:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                h, w = frame.shape[:2]
                res = mesh.process(frame)
                det.update(time_s=(time.perf_counter() - t0), face_mesh_result=res, image_size=(w, h))
        finally:
            cap.release()
            mesh.close()

        elapsed = max(1e-9, time.perf_counter() - t0)
        fps = frames / elapsed
        self.assertGreater(fps, 5.0)


if __name__ == "__main__":
    unittest.main()

