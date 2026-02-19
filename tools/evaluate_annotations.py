from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    label: str


def load_annotations(path: Path) -> List[Interval]:
    intervals: List[Interval] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            intervals.append(
                Interval(
                    start=float(row["start_time_s"]),
                    end=float(row["end_time_s"]),
                    label=str(row["label"]).strip().upper(),
                )
            )
    return intervals


def label_at_time(intervals: List[Interval], t: float) -> str:
    for it in intervals:
        if it.start <= t < it.end:
            return it.label
    return "OPEN"


def load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    times: List[float] = []
    labels: List[int] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["time_s"])
            state = str(row["state"]).strip().upper()
            is_closed = 1 if state == "CLOSED" else 0
            times.append(t)
            labels.append(is_closed)
    return np.asarray(times, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate eye closure predictions against manual annotations")
    p.add_argument("--pred", required=True, type=str)
    p.add_argument("--ann", required=True, type=str)
    args = p.parse_args()

    pred_path = Path(args.pred)
    ann_path = Path(args.ann)

    intervals = load_annotations(ann_path)
    t, y_pred = load_predictions(pred_path)
    y_true = np.asarray([1 if label_at_time(intervals, float(ts)) == "CLOSED" else 0 for ts in t], dtype=np.int64)

    m = metrics(y_true, y_pred)
    for k in ("tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"):
        v = m[k]
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

