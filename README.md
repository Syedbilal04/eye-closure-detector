# Eye Closure Detector

> Real-time eye-closure detection with **MediaPipe Face Landmarker** and **Eye Aspect Ratio (EAR)**. It tells normal blinks apart from prolonged eye closure and adapts its threshold to each user through calibration.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14+-0097A7?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

6th-semester mini project.

## ✨ Features

- Per-frame eye states: `OPEN`, `BLINK`, `CLOSED`, `NO_FACE`
- **EAR** per eye from 6 landmarks each, averaged across both eyes
- **Blink vs. prolonged closure** decided by duration-based state tracking
- **Adaptive calibration:** measures your open-eye EAR (mean/std) and sets `threshold = clamp(mean − k·std, min, max)`, then keeps adapting it during the session from an exponential moving average of open-eye EAR
- **Multi-face handling:** when several faces are detected, follows the closest one (largest inter-ocular distance)
- Handles lost faces gracefully: emits `NO_FACE` and never latches a false closure
- Works with webcams or video files. Headless mode and CSV export are available
- Low-light option (`--equalize`) and a downscale option (`--process-scale`) for higher FPS
- Benchmark tool, an annotation-based accuracy evaluator (accuracy / precision / recall / F1), and unit tests for the EAR math

## 🧠 How it works

```
EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)
```

| Eye | Face Mesh landmark indices |
| --- | --- |
| Left | `[33, 160, 158, 133, 153, 144]` |
| Right | `[362, 385, 387, 263, 373, 380]` |

`p1`/`p4` are the horizontal eye corners; `(p2, p6)` and `(p3, p5)` are the vertical pairs. See `docs/algorithm.md` for details.

## 🛠️ Tech Stack

Python · MediaPipe Tasks (Face Landmarker) · OpenCV · NumPy · `unittest`

## 🚀 Getting Started

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt   # mediapipe>=0.10.14, opencv-python>=4.9, numpy>=1.26
```

### Run on a webcam

```bash
python -m eye_closure_detector.app --source 0 --calibrate-seconds 3
```

The face landmarker model ships in `models/face_landmarker.task`. If it's missing, the app downloads it on first run.

### Useful options

| Flag | Purpose |
| --- | --- |
| `--source` | Webcam index (e.g. `0`) or a video path |
| `--calibrate-seconds` | Open-eye calibration time (default 3) |
| `--no-display` | Run headless |
| `--csv out.csv` | Write per-frame predictions |
| `--process-scale 0.5` | Downscale frames for speed |
| `--max-num-faces` | Max faces to track (default 3) |
| `--equalize` | Histogram-equalise luma for low light |
| `--width` / `--height` | Capture resolution |
| `--min-det-conf` / `--min-track-conf` | Detection/tracking confidence (default 0.6) |
| `--max-seconds` | Stop after N seconds (smoke tests) |

### Video file → CSV

```bash
python -m eye_closure_detector.app --source path/to/video.mp4 --no-display --csv out.csv
```

### Tests, benchmark and evaluation

```bash
python -m unittest discover -s tests -p "test_*.py" -q
# optional integration/performance tests: set EYE_VIDEO_PATH=path/to/video.mp4 first
python -m eye_closure_detector.benchmark --source 0 --seconds 10
python tools/evaluate_annotations.py --pred out.csv --ann docs/sample_annotations.csv
```

The performance target is **≥30 FPS** on typical 720p webcam input (hardware-dependent). Measured numbers are not recorded yet; see `docs/performance.md` for how to benchmark.

## 📁 Project Structure

```
eye_closure_detector/
  app.py                 # CLI entry point (webcam/video, overlay, CSV)
  detector.py            # EyeClosureDetector state machine
  ear.py                 # EAR computation
  calibration.py         # adaptive threshold calibration
  landmarks.py           # eye landmark indices
  mediapipe_facemesh.py  # MediaPipe Tasks wrapper
  model_assets.py        # model download/locate
  benchmark.py           # FPS benchmark
models/face_landmarker.task
tools/                   # evaluation + inspection + smoke-run scripts
tests/test_ear_unittest.py
docs/                    # algorithm, annotation format, performance, sample annotations
```

## 🗺️ Possible extensions

- Drowsiness alerting built on top of `CLOSED` duration
- Federated / on-device training

## 📄 License

No licence file has been added yet, so all rights are reserved by default.
