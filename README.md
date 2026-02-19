# Eye Closure Detection (MediaPipe Face Mesh)

This repository provides a complete, production-oriented Python module for real-time eye-closure detection using **MediaPipe Face Mesh** (468 landmarks) and **Eye Aspect Ratio (EAR)**.

## How It Works

1. **Face Landmarker (MediaPipe Tasks)** detects 468 facial landmarks per face.
2. For each eye, the system extracts **6 key landmarks** (12 total points).
3. It computes **EAR** per eye:

   `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`

4. It fuses both eyes (average EAR) and applies **state tracking** to distinguish:
   - **Blink** (typical duration ~100–400ms)
   - **Prolonged closure** (longer than blink threshold)

## Landmark Indices

The module uses widely adopted Face Mesh landmark indices (per MediaPipe topology):

- **Left eye (6 points)**: `[33, 160, 158, 133, 153, 144]`
- **Right eye (6 points)**: `[362, 385, 387, 263, 373, 380]`

Point ordering matches the EAR formula where `p1` and `p4` are the horizontal corners, and `(p2,p6)` and `(p3,p5)` form the two vertical pairs.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (Webcam)

```bash
python -m eye_closure_detector.app --source 0 --calibrate-seconds 3
```

On first run, the app downloads the face landmarker model to `models/face_landmarker.task`.

Smoke-test for a few seconds (no UI):

```bash
python -m eye_closure_detector.app --source 0 --no-display --calibrate-seconds 0 --max-seconds 5
```

If FPS is low, try:

```bash
python -m eye_closure_detector.app --source 0 --process-scale 0.5 --max-num-faces 1
```

Low-light option:

```bash
python -m eye_closure_detector.app --source 0 --equalize
```

## Run (Video File)

```bash
python -m eye_closure_detector.app --source path\\to\\video.mp4 --no-display --csv out.csv
```

## Calibration and Adaptive Threshold

Calibration measures your **open-eye** EAR for a few seconds and computes:

- `baseline_ear_mean`
- `baseline_ear_std`
- `closure_threshold = clamp(baseline_mean - k * baseline_std, min, max)`

This adapts to individual eye geometry and reduces false detections.

## Tests

Unit tests (EAR math):

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

Optional integration test (requires a video path):

```bash
set EYE_VIDEO_PATH=C:\\path\\to\\video.mp4
python -m unittest discover -s tests -p "test_*.py" -q
```

## Accuracy Validation

For accuracy validation against manual annotations:

1. Run predictions to CSV (`--csv`).
2. Create an annotation CSV (see `docs/annotation_format.md`).
3. Evaluate:

```bash
python tools/evaluate_annotations.py --pred out.csv --ann docs/sample_annotations.csv
```

## Performance

Benchmark on a file/webcam:

```bash
python -m eye_closure_detector.benchmark --source 0 --seconds 10
```

The target is **≥30 FPS** on typical 720p webcam input; performance depends on CPU/GPU and resolution.

## More Documentation

- `docs/algorithm.md`
- `docs/performance.md`
- `docs/annotation_format.md`

## Notes on Robustness

- If detection is temporarily lost, the module emits `NO_FACE` and avoids latching false closures.
- If multiple faces are present, it automatically chooses the face with the **largest inter-ocular distance** (closest face).
- If landmarks are unstable under extreme lighting/occlusion, the detector increases smoothing and requires more evidence before declaring prolonged closure.

