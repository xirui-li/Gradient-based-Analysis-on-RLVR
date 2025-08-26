#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-step evolution of:
  1) Total Shapley (sum over ALL params), averaged across ranks
  2) Accuracy reward ("rewards/accuracy_reward/mean"), averaged across ranks

Includes optional smoothing (EMA or moving average), point-decimation, and marker toggle.

Expected layout:
stats_root/
  step_00000000/metrics.json
  step_00000050/metrics.json
  ...

Each metrics.json:
  {
    "entries": [
      {
        "mode": "train" | "eval",
        "metrics": {
          "shapley_stats/params/<name>/dot_product": <number or [..]>,
          "rewards/accuracy_reward/mean": <number or [..]>,
          ...
        }
      },
      ...
    ]
  }

Outputs:
- <outdir>/total_shapley__<mode>.png
- <outdir>/accuracy_reward__<mode>.png
- Optional CSV: step,total_shapley,accuracy_reward,mode
"""

import argparse, json, os, re, glob, math, statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- helpers ---------------------------------

def read_metrics_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to read {path}: {e}")
        return None

def list_steps(stats_root: str) -> List[Tuple[int, str]]:
    """Return list of (step_int, metrics_json_path) sorted by step directory name."""
    step_dirs = sorted(glob.glob(os.path.join(stats_root, "step_*")))
    out = []
    for d in step_dirs:
        base = os.path.basename(d)
        m = re.match(r"step_(\d+)$", base)
        if not m:
            continue
        step = int(m.group(1))
        metrics_path = os.path.join(d, "metrics.json")
        if os.path.isfile(metrics_path):
            out.append((step, metrics_path))
    return out

def _flatten_last_numeric(met: Dict[str, object], key: str) -> Optional[float]:
    """Return last numeric value for `key` from metrics dict (handles lists/str)."""
    if key not in met:
        return None
    v = met[key]
    if isinstance(v, list):
        if not v:
            return None
        v = v[-1]
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)  # numeric strings
    except Exception:
        return None

def _avg_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return None
    return sum(xs) / len(xs)

def _nan_stats(label: str, arr: List[Optional[float]]) -> None:
    vals = [v for v in arr if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if vals:
        print(f"[info] {label}: {len(vals)} points | min={min(vals):.6g} max={max(vals):.6g} mean={statistics.fmean(vals):.6g}")
    else:
        print(f"[warn] {label}: no usable points")

# ----------------------------- smoothing --------------------------------

def _as_float_array(y: List[Optional[float]]) -> np.ndarray:
    return np.array([
        (np.nan if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v))
        for v in y
    ], dtype=float)

def _moving_average(y: List[Optional[float]], window: int = 21) -> List[float]:
    if window < 1:
        return _as_float_array(y).tolist()
    y_arr = _as_float_array(y)
    w = np.ones(int(window), dtype=float)
    # numerator: treat NaNs as 0
    num = np.convolve(np.nan_to_num(y_arr, nan=0.0), w, mode="same")
    # denominator: count of non-NaN in window
    den = np.convolve(~np.isnan(y_arr), w, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
    return out.tolist()

def _ema(y: List[Optional[float]], alpha: float = 0.2) -> List[float]:
    y_arr = _as_float_array(y)
    out = np.full_like(y_arr, np.nan, dtype=float)
    s = np.nan
    for i, v in enumerate(y_arr):
        if np.isnan(v):
            out[i] = s
            continue
        s = v if np.isnan(s) else alpha * v + (1.0 - alpha) * s
        out[i] = s
    return out.tolist()

def _apply_smoothing(y: List[Optional[float]], kind: str, ma_window: int, ema_alpha: float) -> List[Optional[float]]:
    if kind == "ma":
        return _moving_average(y, window=ma_window)
    if kind == "ema":
        return _ema(y, alpha=ema_alpha)
    # none
    return [float("nan") if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in y]

# ------------------------ aggregation per step -------------------------

# Accuracy key preference (your exact key first)
ACCURACY_KEYS = [
    "rewards/accuracy_reward/mean",
    "reward/accuracy",
    "accuracy_reward",
    "acc_reward",
    "accuracy",
    "eval/accuracy",
    "metrics/accuracy",
]

def extract_step_aggregates(metrics_blob: dict, mode: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (total_shapley_sum_avg_over_ranks, accuracy_reward_avg_over_ranks)
    restricted to entries with matching `mode`.

    Per rank entry:
      - total_shapley_rank = sum of all metrics whose keys match:
            "shapley_stats/params/<name>/dot_product"
      - acc_rank = first found among ACCURACY_KEYS
    Then average across ranks (ignoring Nones).
    """
    if not metrics_blob:
        return (None, None)

    entries = metrics_blob.get("entries", [])
    per_rank_shapley_sums: List[Optional[float]] = []
    per_rank_acc_vals: List[Optional[float]] = []

    for e in entries:
        if e.get("mode") != mode:
            continue
        met = e.get("metrics", {})
        if not isinstance(met, dict):
            continue

        # Sum Shapley over all params for this rank
        s_sum = 0.0
        s_any = False
        for k in met.keys():
            if isinstance(k, str) and k.startswith("shapley_stats/params/") and k.endswith("/dot_product"):
                val = _flatten_last_numeric(met, k)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    s_sum += float(val)
                    s_any = True
        per_rank_shapley_sums.append(s_sum if s_any else None)

        # Accuracy reward
        acc_val = None
        for ak in ACCURACY_KEYS:
            acc_val = _flatten_last_numeric(met, ak)
            if acc_val is not None:
                break
        per_rank_acc_vals.append(acc_val)

    total_shapley = _avg_ignore_none(per_rank_shapley_sums)
    acc_reward = _avg_ignore_none(per_rank_acc_vals)
    return (total_shapley, acc_reward)

# ----------------------------- CSV writing ------------------------------

def write_csv(csv_path: str, rows: List[Tuple[int, Optional[float], Optional[float], str]]):
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "total_shapley", "accuracy_reward", "mode"])
        for step, s, a, mode in rows:
            w.writerow([step, s, a, mode])
    print(f"[ok] Wrote CSV to {csv_path}")

# ------------------------------- plotting --------------------------------

def _plot_one(x, y, title, ylabel, fname, *, smooth="ema", ma_window=21, ema_alpha=0.2, markers=True, stride=1):
    y_plot = _apply_smoothing(y, kind=smooth, ma_window=ma_window, ema_alpha=ema_alpha)

    # optional decimation to reduce clutter
    x_plot = x[::stride]
    y_plot = y_plot[::stride]

    plt.figure()
    plt.plot(x_plot, y_plot, marker=("o" if markers else None), linewidth=1)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[ok] Saved {fname}")

# ------------------------------- main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot TOTAL Shapley (sum over params) and Accuracy reward over steps.")
    parser.add_argument("--stats_root", required=True, help="Path like stats/<run_name>")
    parser.add_argument("--mode", default="train", choices=["train", "eval"], help="Which mode to read")
    parser.add_argument("--outdir", default="plots", help="Where to save PNGs and CSV")
    parser.add_argument("--csv", default=None, help="Optional CSV path for the aggregated series")

    # smoothing & plotting options
    parser.add_argument("--smooth", choices=["none", "ema", "ma"], default="ema",
                        help="Smoothing: ema (exp. moving avg), ma (moving avg), or none.")
    parser.add_argument("--ema_alpha", type=float, default=0.2,
                        help="EMA alpha in (0,1]. Higher follows data faster.")
    parser.add_argument("--ma_window", type=int, default=21,
                        help="Window size for moving average (odd recommended).")
    parser.add_argument("--no_markers", action="store_true", help="Plot lines without point markers.")
    parser.add_argument("--stride", type=int, default=1, help="Plot every Nth point to reduce clutter.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    steps = list_steps(args.stats_root)
    if not steps:
        print(f"[error] No step_* folders under {args.stats_root}")
        return

    xs = [s for s, _ in steps]
    ys_total_shapley: List[Optional[float]] = []
    ys_accuracy: List[Optional[float]] = []
    csv_rows = []

    for step, path in steps:
        blob = read_metrics_json(path)
        total_s, acc = extract_step_aggregates(blob, args.mode)
        ys_total_shapley.append(total_s)
        ys_accuracy.append(acc)
        if args.csv:
            csv_rows.append((step, total_s, acc, args.mode))

    print("\n[info] Aggregated series (averaged across ranks per step):")
    _nan_stats("Total Shapley (sum over params)", ys_total_shapley)
    _nan_stats("Accuracy Reward", ys_accuracy)

    # plots
    _plot_one(
        xs, ys_total_shapley,
        f"Total Shapley (sum over params) over steps — mode: {args.mode}",
        "Total Shapley (dot-product sum)",
        os.path.join(args.outdir, f"total_shapley__{args.mode}.png"),
        smooth=args.smooth, ma_window=args.ma_window, ema_alpha=args.ema_alpha,
        markers=not args.no_markers, stride=args.stride
    )

    _plot_one(
        xs, ys_accuracy,
        f"Accuracy Reward over steps — mode: {args.mode}",
        "Accuracy Reward",
        os.path.join(args.outdir, f"accuracy_reward__{args.mode}.png"),
        smooth=args.smooth, ma_window=args.ma_window, ema_alpha=args.ema_alpha,
        markers=not args.no_markers, stride=args.stride
    )

    if args.csv:
        write_csv(args.csv, csv_rows)

if __name__ == "__main__":
    main()

