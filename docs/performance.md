# Performance & Validation

## Benchmark FPS

Run:

```bash
python -m eye_closure_detector.benchmark --source 0 --seconds 10
```

Expected output:

```text
frames=XXX elapsed_s=10.000 fps=YY.YY
```

Notes:

- FPS depends strongly on resolution and CPU.
- For higher FPS, reduce capture resolution via `--width/--height` in the app.

## Accuracy Validation (Manual Annotations)

1. Generate prediction CSV:

```bash
python -m eye_closure_detector.app --source video.mp4 --no-display --csv out.csv
```

2. Create manual annotation CSV (see `annotation_format.md`).
3. Evaluate:

```bash
python tools/evaluate_annotations.py --pred out.csv --ann your_annotations.csv
```

The script prints accuracy/precision/recall/F1 on a frame-aligned basis.

## Achieving >95% Accuracy

In practice, accuracy is driven by:

- Correct calibration (user looking forward, eyes open)
- Camera stability and sufficient lighting
- Choosing reasonable duration thresholds for the target use case

If accuracy is below target for a specific environment, recommended adjustments:

- Increase `--calibrate-seconds` to improve threshold estimation
- Enable `--equalize` in low-light
- Tune `min_closed_ms` and EAR threshold behavior in `EyeClosureDetector`

