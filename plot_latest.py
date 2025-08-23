#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-step evolution of Shapley, weight (or proxy), and reward for 9 sampled
parameters spread across early/middle/late layers.

Sampling logic:
- Discover param names with Shapley keys by scanning early steps.
- Parse layer index from names (expects "...layers.<idx>....").
- Pick 3 distinct-layer params from each band: early, middle, late (fill as needed).

Existing behavior preserved:
- Per-param, save 3 separate charts: Shapley, Weight/Proxy, Reward.
- Averages over ranks per step; handles missing values.
- Optional CSV export will include ALL sampled params in long format.

Usage (example):
  python plot_param_evolution_9.py \
    --stats_root stats/<run_name> \
    --mode train \
    --param_regex "mlp\\.up_proj\\.weight$" \
    --scan_steps 12 \
    --outdir plots \
    --csv plots/series_9params.csv
"""

import argparse, json, os, re, glob, math, statistics
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

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
    """Return list of (step_int, metrics_json_path) sorted by step."""
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

def _flatten_dict_lists(d: Dict[str, List], key: str):
    """Return last numeric value from metrics dict list if present and numeric."""
    if key not in d:
        return None
    v = d[key]
    if isinstance(v, list) and len(v) > 0:
        v = v[-1]
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return None

def _avg_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v)))]
    if not xs:
        return None
    return sum(xs) / len(xs)

def extract_param_names_from_one(metrics_blob: dict, mode: str) -> List[str]:
    """Find all param names that have shapley values in this blob->entries[*]->metrics."""
    names = set()
    if not metrics_blob:
        return []
    entries = metrics_blob.get("entries", [])
    for e in entries:
        if e.get("mode") != mode:
            continue
        met = e.get("metrics", {})
        for k in met.keys():
            # Expect "shapley_stats/params/{name}/dot_product"
            if k.startswith("shapley_stats/params/") and k.endswith("/dot_product"):
                middle = k[len("shapley_stats/params/") : -len("/dot_product")]
                names.add(middle)
    return sorted(names)

def extract_all_param_names(stats_root: str, mode: str, scan_steps: int) -> List[str]:
    """Union of param names across the first `scan_steps` step folders with valid JSON."""
    steps = list_steps(stats_root)
    if not steps:
        return []
    pool = set()
    for _, p in steps[:max(1, scan_steps)]:
        blob = read_metrics_json(p)
        pool.update(extract_param_names_from_one(blob, mode))
    return sorted(pool)

_LAYER_RE = re.compile(r"\blayers\.(\d+)\b")

def parse_layer_idx(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def get_weight_keys_for(name: str) -> List[str]:
    """Ordered preference for weight value keys."""
    return [
        f"weights/params/{name}/value",        # actual weight value (mean) if saved
        f"model/params/{name}/norm",           # param norm from model snapshot
        f"param_stats/{name}/norm",            # any custom norm path
        f"grad_stats/params/{name}/frobenius_norm",  # fallback proxy (grad norm)
    ]

def extract_step_values(metrics_blob: dict, mode: str, name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (shapley, weight_or_proxy, reward) averaged over ranks for this step."""
    if not metrics_blob:
        return (None, None, None)
    entries = metrics_blob.get("entries", [])
    shapley_vals, weight_vals, reward_vals = [], [], []

    shapley_key = f"shapley_stats/params/{name}/dot_product"
    weight_keys = get_weight_keys_for(name)

    for e in entries:
        if e.get("mode") != mode:
            continue
        met = e.get("metrics", {})

        shapley_vals.append(_flatten_dict_lists(met, shapley_key))

        w_val = None
        for wk in weight_keys:
            w_val = _flatten_dict_lists(met, wk)
            if w_val is not None:
                break
        weight_vals.append(w_val)

        reward_vals.append(_flatten_dict_lists(met, "reward"))

    return (
        _avg_ignore_none(shapley_vals),
        _avg_ignore_none(weight_vals),
        _avg_ignore_none(reward_vals),
    )

def sanitize(s: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", s)[:140]

# ----------------------- Sampling 9 params by bands ---------------------

def sample_params_by_bands(all_names: List[str],
                           per_band: int = 3,
                           bands: int = 3) -> List[str]:
    """
    Select `per_band` names from each of `bands` bands along the layer index.
    Falls back to even sampling if not enough layer-indexed names.
    """
    # Separate names with/without layer index
    with_idx = []
    no_idx = []
    for n in all_names:
        li = parse_layer_idx(n)
        if li is None:
            no_idx.append(n)
        else:
            with_idx.append((li, n))

    if not with_idx:
        # Fallback: no layer indices; take first per_band*bands names evenly spaced
        target = per_band * bands
        if len(all_names) <= target:
            print("[warn] No layer indices found; taking all available names.")
            return all_names
        stride = max(1, len(all_names) // target)
        picked = all_names[::stride][:target]
        print(f"[warn] No layer indices; evenly sampled {len(picked)} params.")
        return picked

    # Sort by layer, then name for stability
    with_idx.sort(key=lambda t: (t[0], t[1]))
    layer_vals = [li for li, _ in with_idx]
    Lmin, Lmax = min(layer_vals), max(layer_vals)
    if Lmin == Lmax:
        # All params from same layer; just take first target
        target = per_band * bands
        uniq = [n for _, n in with_idx]
        return uniq[:target]

    # Define bands by tertiles (or general equal-range bands)
    # Compute band edges as floats, then map each li to band in [0, bands-1].
    def band_of(li: int) -> int:
        # Normalize to [0,1] then scale
        alpha = (li - Lmin) / max(1e-9, (Lmax - Lmin))
        b = int(alpha * bands)
        return min(b, bands - 1)

    buckets: List[List[Tuple[int, str]]] = [[] for _ in range(bands)]
    for li, n in with_idx:
        buckets[band_of(li)].append((li, n))

    # Pick up to per_band from each bucket, preferring distinct layers first
    picked: List[str] = []
    for b, bucket in enumerate(buckets):
        if not bucket:
            continue
        # stable order: by layer, then name
        bucket.sort(key=lambda t: (t[0], t[1]))
        chosen, seen_layers = [], set()
        # First pass: ensure distinct layers if possible
        for li, n in bucket:
            if li in seen_layers:
                continue
            chosen.append(n)
            seen_layers.add(li)
            if len(chosen) >= per_band:
                break
        # Second pass: fill if still short
        if len(chosen) < per_band:
            for _, n in bucket:
                if n in chosen:
                    continue
                chosen.append(n)
                if len(chosen) >= per_band:
                    break
        picked.extend(chosen[:per_band])

    # If total < 9, fill from remaining (with index first, then no_idx)
    target = per_band * bands
    if len(picked) < target:
        remaining_with_idx = [n for _, n in with_idx if n not in picked]
        for n in remaining_with_idx:
            if len(picked) >= target:
                break
            picked.append(n)
    if len(picked) < target:
        for n in no_idx:
            if len(picked) >= target:
                break
            picked.append(n)

    return picked[:target]

# ----------------------------- CSV writing ------------------------------

def write_csv_multi(csv_path: str, records: List[Tuple[str, int, Optional[float], Optional[float], Optional[float], str]]):
    """
    Write long-form CSV with columns:
      param_name, step, shapley, weight_or_proxy, reward, mode
    """
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param_name", "step", "shapley", "weight_or_proxy", "reward", "mode"])
        for param_name, step, s, we, r, mode in records:
            w.writerow([param_name, step, s, we, r, mode])
    print(f"[ok] Wrote CSV to {csv_path}")

# ----------------------------- plotting --------------------------------

def _nan_stats(label, arr):
    vals = [v for v in arr if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if vals:
        print(f"[info] {label}: have {len(vals)} points, "
              f"min={min(vals):.6g}, max={max(vals):.6g}, mean={statistics.fmean(vals):.6g}")
    else:
        print(f"[warn] {label}: no usable points")

def _plot_one(x, y, title, ylabel, fname):
    plt.figure()
    y2 = [float("nan") if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in y]
    plt.plot(x, y2, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"[ok] Saved {fname}")

# ------------------------------- main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Shapley, Weight, Reward over steps for 9 sampled parameters.")
    parser.add_argument("--stats_root", required=True, help="Path like stats/<run_name>")
    parser.add_argument("--mode", default="train", choices=["train", "eval"], help="Which mode to read")
    parser.add_argument("--param_regex", default=None, help="Regex to filter parameter names (matched against full name)")
    parser.add_argument("--scan_steps", type=int, default=10, help="How many initial steps to scan for parameter discovery")
    parser.add_argument("--n_per_band", type=int, default=1, help="Parameters per band (early/middle/late)")
    parser.add_argument("--bands", type=int, default=3, help="Number of bands across depth (default=3 for early/mid/late)")
    parser.add_argument("--outdir", default="plots", help="Where to save PNGs and CSV")
    parser.add_argument("--csv", default=None, help="Optional CSV path for ALL sampled series (long-form)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    steps = list_steps(args.stats_root)
    if not steps:
        print(f"[error] No step_* folders under {args.stats_root}")
        return

    # Discover candidate param names with Shapley presence
    all_names = extract_all_param_names(args.stats_root, args.mode, args.scan_steps)
    if not all_names:
        print("[error] No parameters with Shapley stats found in scanned steps.")
        return

    if args.param_regex:
        rx = re.compile(args.param_regex)
        filtered = [n for n in all_names if rx.search(n)]
        if filtered:
            all_names = filtered
            print(f"[info] Filtered by /{args.param_regex}/ => {len(all_names)} candidates")
        else:
            print(f"[warn] No names matched /{args.param_regex}/; proceeding with unfiltered pool of {len(all_names)}.")

    sampled = sample_params_by_bands(all_names, per_band=args.n_per_band, bands=args.bands)
    if not sampled:
        print("[error] Sampling failed to pick any parameters.")
        return

    # Log selection and save a manifest
    manifest_path = os.path.join(args.outdir, "sampled_params.txt")
    with open(manifest_path, "w") as f:
        for n in sampled:
            li = parse_layer_idx(n)
            f.write(f"{n}\tlayer_idx={li if li is not None else 'NA'}\n")
    print(f"[ok] Saved sampled param list to {manifest_path}")
    print("[info] Sampled parameters:")
    for n in sampled:
        print(f"    - {n} (layer_idx={parse_layer_idx(n)})")

    # Prepare per-step arrays once
    xs = [s for s, _ in steps]

    # For CSV aggregation
    csv_records = []

    # Iterate each sampled param and produce 3 plots
    for name in sampled:
        ys_shapley, ys_weight, ys_reward = [], [], []
        for step, path in steps:
            blob = read_metrics_json(path)
            s, w, r = extract_step_values(blob, args.mode, name)
            ys_shapley.append(s)
            ys_weight.append(w)
            ys_reward.append(r)
            if args.csv is not None:
                csv_records.append((name, step, s, w, r, args.mode))

        # Basic stats per param
        print(f"\n[info] Param: {name}")
        _nan_stats("Shapley", ys_shapley)
        _nan_stats("Weight/Proxy", ys_weight)
        _nan_stats("Reward", ys_reward)

        base = f"{sanitize(name)}__{args.mode}"
        # 3 separate charts
        _plot_one(xs, ys_shapley,
                  f"Shapley over steps\nparam: {name}  |  mode: {args.mode}",
                  "Shapley (dot product)",
                  os.path.join(args.outdir, f"{base}__shapley.png"))
        _plot_one(xs, ys_weight,
                  f"Weight (or proxy) over steps\nparam: {name}  |  mode: {args.mode}",
                  "Weight / Proxy (fallback rules)",
                  os.path.join(args.outdir, f"{base}__weight_or_proxy.png"))
        _plot_one(xs, ys_reward,
                  f"Reward over steps\nmode: {args.mode}",
                  "Reward",
                  os.path.join(args.outdir, f"{base}__reward.png"))

    if args.csv:
        write_csv_multi(args.csv, csv_records)

if __name__ == "__main__":
    main()
