#!/usr/bin/env python3
"""
Benchmark inferred results against the GreenMIR dataset.

- Reads ground truth from a CSV (e.g., static/dataset.csv).
- Reads inference outputs from paper-*.json (structure compatible with src/infer.py).
- For each paper and each column, keeps the model candidate with the smallest distance.
- Plots one pair of charts per column (missingness + histogram of distances).
- Writes a CSV with mean absolute distance-to-zero per column and a global weighted average.
"""
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler


# ---------------------------------------------------------------------------
# Config

BAR_COLORS = ["#16a34a", "#f59e0b", "#ef4444"]
HIST_COLOR = "#3b82f6"

DATASET_RENAME = {
    "model name": "model",
}

COLUMN_META = {
    "model": ("text", "Model (JW)"),
    "abstract": ("text", "Abstract (JW)"),
    "country": ("text", "Country (JW)"),
    "year": ("numeric", "Year (rel)"),
    "parameters": ("numeric", "Parameters (rel)"),
    "hardware": ("text", "Hardware (JW)"),
    "hardware_compute": ("numeric", "Hardware compute (rel)"),
    "hardware_number": ("numeric", "Hardware number (rel)"),
    "hardware_power": ("numeric", "Hardware power (rel)"),
    "training_compute": ("numeric", "Training compute (rel)"),
    "training_time": ("numeric", "Training time (rel)"),
    "power_draw": ("numeric", "Power draw (rel)"),
    "co2eq": ("numeric", "CO₂eq (rel)"),
    # Map inference aliases to dataset columns (handled at load time)
    "h_number": ("numeric", "Hardware number (rel)"),
    "h_power": ("numeric", "Hardware power (rel)"),
    "h_compute": ("numeric", "Hardware compute (rel)"),
}


# ---------------------------------------------------------------------------
# Helpers

def _jw(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    score = JaroWinkler.normalized_similarity(str(a), str(b))
    return score / 100.0 if score > 1.0 else score


def _signed_rel(infer_val: float | None, base_val: float | None) -> float | None:
    if infer_val is None or base_val is None:
        return None
    denom = max(abs(infer_val), abs(base_val), 1.0)
    if denom <= 0:
        return None
    return (infer_val - base_val) / denom


def _extract_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value):
                return None
        except Exception:
            pass
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped:
            return None
        match = _NUMERIC_PATTERN.search(stripped)
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None
    return None


def _is_missing_text(val: Any) -> bool:
    if val is None:
        return True
    # Handle pandas NaN values
    if isinstance(val, float):
        try:
            if math.isnan(val):
                return True
        except Exception:
            pass
    if isinstance(val, str):
        return not val.strip()
    return False


# ---------------------------------------------------------------------------
# Data loading

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = df.rename(columns=DATASET_RENAME)
    df.insert(0, "id_paper", df.index + 1)
    keep_cols = ["id_paper"] + [col for col in COLUMN_META if col in df.columns]
    return df[keep_cols]


def load_inference(outputs_path: Path, tracked_fields: list[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with possibly several rows per paper (one per model in the JSON).
    Downstream logic will pick the best model per metric (smallest distance).
    
    Supports two JSON formats:
    1. Nested format: [{id_paper: X, models: [{model: "...", h_number: ..., ...}]}]
    2. Flat format: [{id_paper: X, hardware_number: ..., parameters: ..., ...}]
    """
    records: list[dict[str, Any]] = []
    paths: list[Path]
    if outputs_path.is_file():
        paths = [outputs_path]
    else:
        paths = sorted(outputs_path.glob("paper-*.json"))
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[benchmark] skip {path}: {exc}")
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            pid = entry.get("id_paper")
            if pid is None:
                continue
            models = entry.get("models", [])
            
            # Handle flat format (no models array, fields directly on entry)
            if not models:
                # Check if this is flat format (has tracked fields directly)
                has_direct_fields = any(
                    field in entry or field.replace("hardware_", "h_") in entry
                    for field in tracked_fields
                )
                if has_direct_fields:
                    # Flat format: extract fields directly from entry
                    cleaned = {k: _extract_scalar(v) for k, v in entry.items()}
                    record = {"id_paper": int(pid)}
                    for field in tracked_fields:
                        # Try direct field first, then h_* alias
                        if field in cleaned:
                            record[field] = cleaned.get(field)
                        elif field.startswith("hardware_") and field.replace("hardware_", "h_") in cleaned:
                            record[field] = cleaned.get(field.replace("hardware_", "h_"))
                        else:
                            record[field] = None
                    records.append(record)
                else:
                    # No models and no direct fields - create empty record
                    records.append({"id_paper": int(pid)})
                continue
            
            # Handle nested format (models array)
            for model in models:
                if not isinstance(model, dict):
                    continue
                cleaned = {k: _extract_scalar(v) for k, v in model.items()}
                record = {"id_paper": int(pid)}
                for field in tracked_fields:
                    # Allow inference alias mapping (e.g., h_number -> hardware_number)
                    if field.startswith("hardware_") and field.replace("hardware_", "h_") in cleaned:
                        record[field] = cleaned.get(field.replace("hardware_", "h_"))
                    else:
                        record[field] = cleaned.get(field)
                records.append(record)
    if not records:
        return pd.DataFrame(columns=["id_paper"] + tracked_fields)
    df = pd.DataFrame(records)
    return df.sort_values(["id_paper"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics + plotting

@dataclass
class Metric:
    field: str
    kind: str  # "text" | "numeric"
    display_name: str

    @property
    def bins(self) -> Iterable[float]:
        if self.kind == "text":
            return np.linspace(0.0, 1.0, 26)
        return np.linspace(-1.0, 1.0, 41)

    @property
    def range_limits(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.kind == "text" else (-1.0, 1.0)


def build_metrics(common_fields: list[str]) -> list[Metric]:
    return [Metric(field=f, kind=COLUMN_META[f][0], display_name=COLUMN_META[f][1]) for f in common_fields]


def _draw_pair(ax_left, ax_right, distances: list[float], missing: tuple[int, int, int], *, bins, vmin: float, vmax: float, title: str) -> None:
    categories = ["both null", "infer null vs base", "infer vs base null"]
    ax_left.bar(categories, list(missing), color=BAR_COLORS, edgecolor="black", width=0.6)
    ax_left.tick_params(axis="x", rotation=20, labelsize=8)
    ax_left.tick_params(axis="y", labelsize=8)
    ax_right.tick_params(axis="x", labelsize=8)
    ax_right.tick_params(axis="y", labelsize=8)
    ax_right.set_xlim(vmin, vmax)
    ax_left.grid(True, linestyle=":", alpha=0.3)
    ax_right.grid(True, linestyle=":", alpha=0.3)

    if distances:
        hist_vals, _, _ = ax_right.hist(distances, bins=bins, color=HIST_COLOR, edgecolor="black")
        right_max = max(hist_vals) if len(hist_vals) else 0
    else:
        right_max = 0
    left_max = max(missing) if missing else 0
    y_max = max(left_max, right_max, 1)
    ax_left.set_ylim(0, y_max * 1.10)
    ax_right.set_ylim(0, y_max * 1.10)
    ax_left.set_ylabel("count", fontsize=8)
    ax_left.set_title(title, fontsize=10)
    ax_right.set_xlabel("distance", fontsize=8)


def _plot_metrics_grid(out_path: Path, metrics: list[Metric], distances: dict[str, list[float]], missing: dict[str, tuple[int, int, int]]) -> None:
    if not metrics:
        print("[benchmark] no overlapping columns; skip plot")
        return

    cols = 3
    rows = math.ceil(len(metrics) / cols)
    fig = plt.figure(figsize=(cols * 5.5, rows * 4.0))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.30, hspace=0.45)

    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, cols)
        slot = gs[r, c]
        subgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=slot, width_ratios=[1, 2])
        ax_left = fig.add_subplot(subgs[0, 0])
        ax_right = fig.add_subplot(subgs[0, 1], sharey=ax_left)
        vals = [float(v) for v in distances.get(metric.field, []) if v is not None]
        miss = missing.get(metric.field, (0, 0, 0))
        bins = metric.bins
        vmin, vmax = metric.range_limits
        _draw_pair(ax_left, ax_right, vals, miss, bins=bins, vmin=vmin, vmax=vmax, title=metric.display_name)

    legend_handles = [
        plt.matplotlib.patches.Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(BAR_COLORS, ["both null", "infer null vs base", "infer vs base null"], strict=True)
    ]
    legend_handles.append(plt.matplotlib.patches.Patch(facecolor=HIST_COLOR, edgecolor="black", label="distance"))
    legend_y = 0.06 if rows == 1 else 0.03
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=9, frameon=True, bbox_to_anchor=(0.5, legend_y))
    bottom_margin = 0.28 if rows == 1 else (0.24 if rows == 2 else 0.20)
    fig.subplots_adjust(bottom=bottom_margin, top=0.90)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gradient_stacked(out_path: Path, metric: Metric, distances: list[float], missing: tuple[int, int, int], n_bins: int = 10) -> None:
    """
    Create a gradient stacked bar plot for a single metric.
    Bottom: distance distribution as gradient (viridis), Top: null categories stacked.
    """
    total_rows = len(distances) + sum(missing)
    if total_rows == 0:
        return
    
    fig, ax = plt.subplots(figsize=(4, 6))
    
    # Compute fractions
    frac_distances = len(distances) / total_rows if total_rows else 0
    frac_both_null = missing[0] / total_rows if total_rows else 0
    frac_infer_null = missing[1] / total_rows if total_rows else 0
    frac_base_null = missing[2] / total_rows if total_rows else 0
    
    # For the distance portion, bin the values and show as gradient
    cmap = plt.get_cmap('viridis')
    
    if distances:
        # Normalize distances to [0, 1] for coloring
        abs_distances = [abs(d) for d in distances]
        # Bin the distances
        counts, edges = np.histogram(abs_distances, bins=n_bins, range=(0.0, 1.0))
        
        # Draw gradient bars from bottom
        bottom = 0.0
        for i, count in enumerate(counts):
            if count == 0:
                continue
            height = (count / total_rows)
            # Color based on bin position (low distance = light, high distance = dark)
            color = cmap(i / max(1, n_bins - 1))
            ax.bar([0], [height], bottom=bottom, color=color, edgecolor='none', width=0.6)
            bottom += height
    else:
        bottom = 0.0
    
    # Stack null categories on top
    if frac_both_null > 0:
        ax.bar([0], [frac_both_null], bottom=bottom, color=BAR_COLORS[0], edgecolor='black', width=0.6, label='both null')
        bottom += frac_both_null
    if frac_infer_null > 0:
        ax.bar([0], [frac_infer_null], bottom=bottom, color=BAR_COLORS[1], edgecolor='black', width=0.6, label='infer null vs base')
        bottom += frac_infer_null
    if frac_base_null > 0:
        ax.bar([0], [frac_base_null], bottom=bottom, color=BAR_COLORS[2], edgecolor='black', width=0.6, label='infer vs base null')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of rows')
    ax.set_xticks([0])
    ax.set_xticklabels([metric.display_name], rotation=45, ha='right')
    ax.set_title(f'{metric.display_name}: gradient stack')
    
    # Add colorbar for distance scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('normalized distance (low→high), scale=1.0', fontsize=8)
    
    # Legend
    handles = [
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null vs base'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='infer vs base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_all_gradient_stacked(out_path: Path, metrics: list[Metric], distances: dict[str, list[float]], missing: dict[str, tuple[int, int, int]], n_bins: int = 10) -> None:
    """
    Create a combined gradient stacked bar plot for all metrics side by side.
    """
    if not metrics:
        return
    
    # Filter metrics that have data
    valid_metrics = [m for m in metrics if distances.get(m.field) or sum(missing.get(m.field, (0,0,0))) > 0]
    if not valid_metrics:
        return
    
    n_metrics = len(valid_metrics)
    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.5), 6))
    
    cmap = plt.get_cmap('viridis')
    x_positions = np.arange(n_metrics)
    bar_width = 0.7
    
    for idx, metric in enumerate(valid_metrics):
        vals = [abs(v) for v in distances.get(metric.field, [])]
        miss = missing.get(metric.field, (0, 0, 0))
        total_rows = len(vals) + sum(miss)
        
        if total_rows == 0:
            continue
        
        # Compute fractions
        frac_both_null = miss[0] / total_rows
        frac_infer_null = miss[1] / total_rows
        frac_base_null = miss[2] / total_rows
        
        # Bin the distances
        bottom = 0.0
        if vals:
            counts, _ = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                height = count / total_rows
                color = cmap(i / max(1, n_bins - 1))
                ax.bar([idx], [height], bottom=bottom, color=color, edgecolor='none', width=bar_width)
                bottom += height
        
        # Stack null categories
        if frac_both_null > 0:
            ax.bar([idx], [frac_both_null], bottom=bottom, color=BAR_COLORS[0], edgecolor='black', width=bar_width)
            bottom += frac_both_null
        if frac_infer_null > 0:
            ax.bar([idx], [frac_infer_null], bottom=bottom, color=BAR_COLORS[1], edgecolor='black', width=bar_width)
            bottom += frac_infer_null
        if frac_base_null > 0:
            ax.bar([idx], [frac_base_null], bottom=bottom, color=BAR_COLORS[2], edgecolor='black', width=bar_width)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of rows')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m.field for m in valid_metrics], rotation=45, ha='right')
    ax.set_title('All metrics: gradient stack (bottom: distance, top: null categories)')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('normalized distance (low→high)', fontsize=8)
    
    # Legend
    handles = [
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null vs base'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='infer vs base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distance computation

def compute_distances(
    merged: pd.DataFrame,
    metrics: list[Metric],
    infer_models: pd.DataFrame,
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]], list[dict[str, Any]]]:
    """
    Compute distances per paper by selecting, for each metric, the model candidate
    that minimizes the distance. Returns distances, missing counts, and a merged
    (one row per paper) summary containing best inference values.
    """
    distances: dict[str, list[float]] = {m.field: [] for m in metrics}
    missing: dict[str, list[int]] = {m.field: [0, 0, 0] for m in metrics}
    merged_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        out_row: dict[str, Any] = {"id_paper": row["id_paper"]}
        for metric in metrics:
            field = metric.field
            base_val = row.get(f"{field}_base")
            miss = missing[field]

            # Get all model candidates for this paper from the full inference DataFrame
            models = infer_models[infer_models["id_paper"] == row["id_paper"]]
            if models.empty:
                # Fall back to the merged row's inference value
                inf_candidates: list[Any] = [row.get(f"{field}_inf")]
            else:
                inf_candidates = models.get(field, pd.Series(dtype=object)).tolist()

            if not inf_candidates:
                inf_candidates = [None]

            best_val = None

            if metric.kind == "numeric":
                base_num = _parse_numeric(base_val)
                if base_num is None:
                    all_missing = all(_parse_numeric(val) is None for val in inf_candidates)
                    if all_missing:
                        miss[0] += 1
                    else:
                        miss[2] += 1
                    continue

                best_rel = None
                for candidate in inf_candidates:
                    inf_num = _parse_numeric(candidate)
                    if inf_num is None:
                        continue
                    rel = _signed_rel(inf_num, base_num)
                    if rel is None or math.isnan(rel):
                        continue
                    rel_abs = abs(rel)
                    if best_rel is None or rel_abs < abs(best_rel):
                        best_rel = rel
                        best_val = candidate

                if best_rel is None:
                    miss[1] += 1
                else:
                    # For year, keep absolute distance (relative errors can be negative).
                    to_store = abs(best_rel) if field == "year" else float(best_rel)
                    distances[field].append(float(to_store))
            else:
                base_txt = None if _is_missing_text(base_val) else str(base_val).strip()
                if base_txt is None:
                    all_missing = all(_is_missing_text(val) for val in inf_candidates)
                    if all_missing:
                        miss[0] += 1
                    else:
                        miss[2] += 1
                    continue

                best_jw = None
                for candidate in inf_candidates:
                    inf_txt = None if _is_missing_text(candidate) else str(candidate).strip()
                    if inf_txt is None:
                        continue
                    jw_dist = 1.0 - _jw(base_txt, inf_txt)
                    if best_jw is None or jw_dist < best_jw:
                        best_jw = jw_dist
                        best_val = inf_txt
                if best_jw is None:
                    miss[1] += 1
                else:
                    distances[field].append(float(best_jw))

        merged_rows.append(out_row)

    missing = {k: tuple(v) for k, v in missing.items()}  # type: ignore[assignment]
    return distances, missing, merged_rows  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Outputs

def save_combined_json(df: pd.DataFrame, out_path: Path) -> None:
    records = df.sort_values("id_paper").replace({np.nan: None}).to_dict(orient="records")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def save_distance_summary(distances: dict[str, list[float]], out_path: Path) -> float:
    """
    Saves mean absolute distance per column and returns the global weighted average
    (sum(|d|)/count over all metrics).
    """
    rows = []
    total_abs = 0.0
    total_count = 0
    for field, vals in distances.items():
        if not vals:
            mean_abs = None
            count = 0
        else:
            abs_vals = [abs(v) for v in vals]
            mean_abs = float(np.mean(abs_vals))
            count = len(vals)
            total_abs += sum(abs_vals)
            total_count += count
        rows.append({"field": field, "mean_abs_distance": mean_abs, "count": count})

    global_score = (total_abs / total_count) if total_count else float("nan")
    rows.append({"field": "GLOBAL_WEIGHTED_AVG", "mean_abs_distance": global_score, "count": total_count})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return global_score


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs GreenMIR dataset.")
    parser.add_argument("--dataset", required=True, help="CSV with ground-truth values (e.g., static/dataset.csv)")
    parser.add_argument("--outputs-dir", required=True, help="Directory containing paper-XXXX.json files or a single aggregated JSON file")
    parser.add_argument("--out-dir", default="results/benchmark_greenmir", help="Where to store merged data and plots")
    parser.add_argument(
        "--columns",
        nargs="*",
        choices=sorted(COLUMN_META.keys()),
        help="Optional subset of columns to benchmark (defaults to every shared column).",
    )
    parser.add_argument(
        "--all-papers",
        action="store_true",
        help="Include dataset rows even if no inference output exists (default: only keep papers present in outputs).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_dataset(dataset_path)
    requested_columns = args.columns if args.columns else list(COLUMN_META.keys())
    tracked_fields = [field for field in requested_columns if field in base_df.columns]
    if not tracked_fields:
        raise SystemExit("[benchmark] None of the requested columns exist in dataset.")
    infer_df = load_inference(outputs_dir, tracked_fields)

    combined_json_path = out_dir / "combined_inference.json"
    # Save collapsed (one row per paper) for convenience; plotting still uses per-model candidates.
    collapse_for_json = infer_df.groupby("id_paper").tail(1).reset_index(drop=True)
    save_combined_json(collapse_for_json, combined_json_path)
    print(f"[benchmark] combined inference written to {combined_json_path}")

    if not args.all_papers:
        inferred_ids = set(infer_df["id_paper"].tolist())
        base_df = base_df[base_df["id_paper"].isin(inferred_ids)]

    if base_df.empty:
        print("[benchmark] No overlapping papers between dataset and inference; nothing to plot.")
        return

    # Merge like v4/benchmark.py for proper column alignment
    merged = base_df.merge(infer_df, on="id_paper", how="left", suffixes=("_base", "_inf"))
    # Keep only one row per paper to avoid counting duplicates
    merged = merged.drop_duplicates(subset=["id_paper"], keep="first")

    # Only include fields that exist in both base and infer
    common_fields = [field for field in tracked_fields if f"{field}_inf" in merged.columns]
    metrics = build_metrics(common_fields)
    if not metrics:
        print("[benchmark] no common fields between dataset and inference; aborting plot.")
        return

    distances, missing, merged_rows = compute_distances(merged, metrics, infer_df)

    merged_path = out_dir / "merged.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[benchmark] merged comparison CSV written to {merged_path}")
    plot_path = out_dir / "benchmark.png"
    _plot_metrics_grid(plot_path, metrics, distances, missing)
    print(f"[benchmark] figure saved to {plot_path}")

    # Generate gradient stacked plots
    overview_dir = out_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual gradient stacked plots per metric
    for metric in metrics:
        vals = [float(v) for v in distances.get(metric.field, []) if v is not None]
        miss = missing.get(metric.field, (0, 0, 0))
        grad_path = overview_dir / f"grad_stacked_{metric.field}.png"
        _plot_gradient_stacked(grad_path, metric, vals, miss)
    
    # Combined gradient stacked plot for all metrics
    all_grad_path = overview_dir / "grad_stacked_all.png"
    _plot_all_gradient_stacked(all_grad_path, metrics, distances, missing)
    print(f"[benchmark] gradient stacked plots saved to {overview_dir}")

    summary_path = out_dir / "distance_summary.csv"
    global_score = save_distance_summary(distances, summary_path)
    print(f"[benchmark] distance summary saved to {summary_path}")
    print(f"[benchmark] global weighted average distance-to-zero: {global_score}")


if __name__ == "__main__":
    main()
