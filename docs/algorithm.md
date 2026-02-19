# Algorithm Details

## MediaPipe Face Mesh

This module runs MediaPipe **Face Mesh** in streaming mode (`static_image_mode=False`) with:

- `refine_landmarks=True` for better eye-region stability
- `min_detection_confidence` and `min_tracking_confidence` exposed via CLI
- `max_num_faces=3` with automatic “closest face” selection

## Eye Landmark Extraction

For each detected face, 6 landmarks are sampled around each eye.

These indices are chosen so:

- `p1` and `p4` are eye corners (horizontal width)
- `(p2,p6)` and `(p3,p5)` are upper/lower eyelid pairs (vertical openings)

## EAR Computation

Let `p1..p6` be the 2D coordinates (pixel units) of one eye’s 6 points.

EAR is computed as:

`EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`

Properties:

- EAR decreases when the eyelids close.
- The normalization by eye width makes EAR more stable across distance changes.

Implementation details:

- Uses `float64` for numerical stability.
- Guards against division by ~0 (invalid geometry when points collapse).

## Real-Time State Logic

The detector tracks whether EAR stays below a threshold over time:

- If `EAR < threshold` for at least `min_blink_ms`: state becomes `BLINK`.
- If it remains below for at least `min_closed_ms`: state becomes `CLOSED`.
- Very short dips below threshold (e.g., noisy landmarks) are ignored.

This prevents false positives from momentary tracking jitter.

## Calibration + Adaptive Threshold

Calibration observes several seconds of **open-eye** EAR values and estimates:

- `baseline_mean`
- `baseline_std`
- `closure_threshold = clamp(baseline_mean - k * baseline_std, min, max)`

During runtime, the system can slowly adapt the threshold using only samples confidently above threshold (open-eye region). This compensates for:

- mild lighting changes
- the user moving closer/farther
- camera exposure drift

