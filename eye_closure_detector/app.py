from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional, Union

import cv2

from .calibration import EarCalibrator
from .detector import EyeClosureDetector, EyeState
from .mediapipe_facemesh import FaceMeshConfig, FaceMeshRunner


def _open_capture(source: str):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def _draw_text(frame, text: str, y: int, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _equalize_luma(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def run(
    *,
    source: str,
    calibrate_seconds: float,
    no_display: bool,
    csv_path: Optional[Path],
    width: Optional[int],
    height: Optional[int],
    min_det_conf: float,
    min_track_conf: float,
    equalize: bool,
    max_seconds: Optional[float],
    max_num_faces: int,
    process_scale: float,
):
    cap = _open_capture(source)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    mesh = FaceMeshRunner(
        FaceMeshConfig(
            max_num_faces=int(max_num_faces),
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
    )
    detector = EyeClosureDetector()
    calibrator = EarCalibrator()

    csv_file = None
    writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "time_s",
                "ear",
                "left_ear",
                "right_ear",
                "state",
                "face_count",
                "threshold",
                "calibrated",
                "blink_ms",
            ],
        )
        writer.writeheader()

    t0 = time.perf_counter()
    last_report = t0
    frames = 0
    calibrated = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if equalize:
                frame = _equalize_luma(frame)

            proc = frame
            if process_scale and float(process_scale) != 1.0:
                s = float(process_scale)
                if 0.05 < s < 1.0:
                    proc = cv2.resize(frame, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            frames += 1
            now = time.perf_counter()
            time_s = now - t0

            if max_seconds is not None and time_s >= float(max_seconds):
                break
            h, w = proc.shape[:2]

            result = mesh.process(proc, timestamp_ms=int(time_s * 1000.0))
            det = detector.update(time_s=time_s, face_mesh_result=result, image_size=(w, h))

            if not calibrated and calibrate_seconds > 0 and time_s <= calibrate_seconds:
                if det.state != EyeState.NO_FACE and det.ear > 0:
                    calibrator.add(det.ear)
                cal = calibrator.compute()
                if cal is not None and time_s >= min(1.0, calibrate_seconds):
                    detector.set_calibration(cal)
                    calibrated = True

            if writer is not None:
                writer.writerow(
                    {
                        "time_s": f"{det.time_s:.6f}",
                        "ear": f"{det.ear:.6f}",
                        "left_ear": f"{det.left_ear:.6f}",
                        "right_ear": f"{det.right_ear:.6f}",
                        "state": det.state.value,
                        "face_count": det.face_count,
                        "threshold": f"{det.threshold:.6f}",
                        "calibrated": int(det.calibrated),
                        "blink_ms": f"{det.blink_ms:.1f}",
                    }
                )

            if not no_display:
                state_color = (0, 255, 0)
                if det.state == EyeState.CLOSED:
                    state_color = (0, 0, 255)
                elif det.state == EyeState.BLINK:
                    state_color = (0, 255, 255)
                elif det.state == EyeState.NO_FACE:
                    state_color = (180, 180, 180)

                _draw_text(frame, f"STATE: {det.state.value}", 30, state_color)
                _draw_text(frame, f"EAR: {det.ear:.3f} (thr {det.threshold:.3f})", 60)
                if calibrate_seconds > 0 and not calibrated:
                    _draw_text(frame, f"CALIBRATING... {time_s:.1f}/{calibrate_seconds:.1f}s", 90)
                elif calibrated:
                    _draw_text(frame, "CALIBRATED", 90)
                _draw_text(frame, f"faces: {det.face_count}", 120)
                cv2.imshow("Eye Closure Detector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if now - last_report >= 2.0:
                fps = frames / (now - t0)
                last_report = now
                if no_display:
                    print(f"time={time_s:.1f}s fps={fps:.1f} state={det.state.value} ear={det.ear:.3f}")
    finally:
        cap.release()
        mesh.close()
        if csv_file is not None:
            csv_file.close()
        if not no_display:
            cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-time eye closure detection using MediaPipe Face Mesh")
    p.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video path")
    p.add_argument("--calibrate-seconds", type=float, default=3.0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--csv", type=str, default=None, help="Write predictions to CSV")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--min-det-conf", type=float, default=0.6)
    p.add_argument("--min-track-conf", type=float, default=0.6)
    p.add_argument("--equalize", action="store_true", help="Histogram-equalize luma to improve low-light stability")
    p.add_argument("--max-seconds", type=float, default=None, help="Stop after N seconds (useful for smoke tests)")
    p.add_argument("--max-num-faces", type=int, default=3)
    p.add_argument("--process-scale", type=float, default=1.0, help="Downscale frames for faster processing (e.g. 0.5)")
    return p


def main():
    args = build_arg_parser().parse_args()
    run(
        source=str(args.source),
        calibrate_seconds=float(args.calibrate_seconds),
        no_display=bool(args.no_display),
        csv_path=Path(args.csv) if args.csv else None,
        width=args.width,
        height=args.height,
        min_det_conf=float(args.min_det_conf),
        min_track_conf=float(args.min_track_conf),
        equalize=bool(args.equalize),
        max_seconds=args.max_seconds,
        max_num_faces=int(args.max_num_faces),
        process_scale=float(args.process_scale),
    )


if __name__ == "__main__":
    main()

