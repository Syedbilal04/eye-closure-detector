import argparse
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eye_closure_detector.app import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else str(args.source))
    opened = bool(cap.isOpened())
    ok, frame = cap.read() if opened else (False, None)
    shape = getattr(frame, "shape", None)
    cap.release()

    print(f"source={args.source} opened={opened} first_frame={bool(ok)} shape={shape}")
    if not ok:
        return

    run(
        source=str(args.source),
        calibrate_seconds=0.0,
        no_display=True,
        csv_path=Path(args.csv) if args.csv else None,
        width=None,
        height=None,
        min_det_conf=0.6,
        min_track_conf=0.6,
        equalize=False,
        max_seconds=float(args.seconds),
        max_num_faces=1,
        process_scale=0.5,
    )


if __name__ == "__main__":
    main()
