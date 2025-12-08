#!/usr/bin/env python3
"""
Generate filtered gradient stacked plot and percentage table for epoch benchmark results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Colors for null categories
BAR_COLORS = ["#16a34a", "#f59e0b", "#ef4444"]


def _is_missing(val) -> bool:
    """Check if a value is missing/null."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "n/a", "na")


def _compute_distance(base_val, inf_val, field: str) -> float | None:
    """Compute distance between base and inferred value."""
    if _is_missing(base_val) or _is_missing(inf_val):
        return None
    
    # ID fields (country, hardware) - exact match
    if field in ("country", "hardware"):
        try:
            return 0.0 if int(float(base_val)) == int(float(inf_val)) else 1.0
        except (ValueError, TypeError):
            return 0.0 if str(base_val).strip() == str(inf_val).strip() else 1.0
    
    # Year field - absolute difference / 5 (capped at 1.0)
    if field == "year":
        try:
            b = float(base_val)
            i = float(inf_val)
            return min(1.0, abs(b - i) / 5.0)
        except (ValueError, TypeError):
            return None
    
    # Other numeric fields - relative difference
    try:
        b = float(base_val)
        i = float(inf_val)
        if b == 0 and i == 0:
            return 0.0
        denom = max(abs(b), abs(i))
        if denom == 0:
            return None
        return min(1.0, abs(b - i) / denom)
    except (ValueError, TypeError):
        return None


def load_epoch_data(result_dir: Path) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]]]:
    """
    Load distances and missing counts from epoch benchmark results.
    Returns (distances, missing) dictionaries.
    """
    merged_path = result_dir / "matched_models.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"matched_models.csv not found in {result_dir}")
    
    df = pd.read_csv(merged_path)
    
    # Fields to analyze (aligned with epoch.ai documentation - no co2eq)
    fields = ["country", "year", "parameters", "hardware", "hardware_number", 
              "training_time", "training_compute", "power_draw"]
    
    distances: dict[str, list[float]] = {}
    missing: dict[str, tuple[int, int, int]] = {}
    
    for field in fields:
        # Determine column names
        if field in ("country", "hardware"):
            base_col = f"id_{field}_base"
            inf_col = f"id_{field}_inf"
            if base_col not in df.columns:
                base_col = f"{field}_base"
            if inf_col not in df.columns:
                inf_col = f"{field}_inf"
        else:
            base_col = f"{field}_base"
            inf_col = f"{field}_inf"
        
        if base_col not in df.columns or inf_col not in df.columns:
            continue
        
        # Count missing categories and compute distances
        both_null = 0
        infer_null = 0
        base_null = 0
        dist_vals = []
        
        for _, row in df.iterrows():
            base_val = row.get(base_col)
            inf_val = row.get(inf_col)
            
            base_missing = _is_missing(base_val)
            inf_missing = _is_missing(inf_val)
            
            if base_missing and inf_missing:
                both_null += 1
            elif inf_missing:
                infer_null += 1
            elif base_missing:
                base_null += 1
            else:
                dist = _compute_distance(base_val, inf_val, field)
                if dist is not None:
                    dist_vals.append(abs(dist))
        
        distances[field] = dist_vals
        missing[field] = (both_null, infer_null, base_null)
    
    return distances, missing


def plot_grad_stacked_filtered(
    out_path: Path,
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]],
    n_bins: int = 10,
) -> None:
    """
    Create filtered gradient stacked bar plot with:
    - Transparent background
    - "precision" colorbar
    """
    # Order fields
    field_order = ["country", "year", "parameters", "hardware", "hardware_number", 
                   "training_time", "training_compute", "power_draw"]
    valid_fields = [f for f in field_order if f in distances and 
                    (distances.get(f) or sum(missing.get(f, (0, 0, 0))) > 0)]
    
    if not valid_fields:
        print(f"No valid fields to plot for {out_path}")
        return
    
    n_metrics = len(valid_fields)
    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.0), 5))
    
    cmap = plt.get_cmap('viridis')
    x_positions = np.arange(n_metrics)
    bar_width = 0.7
    
    for idx, field in enumerate(valid_fields):
        vals = distances.get(field, [])
        miss = missing.get(field, (0, 0, 0))
        total_rows = len(vals) + sum(miss)
        
        if total_rows == 0:
            continue
        
        frac_both_null = miss[0] / total_rows
        frac_infer_null = miss[1] / total_rows
        frac_base_null = miss[2] / total_rows
        
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
        
        if frac_both_null > 0:
            ax.bar([idx], [frac_both_null], bottom=bottom, color=BAR_COLORS[0], edgecolor='black', width=bar_width)
            bottom += frac_both_null
        if frac_infer_null > 0:
            ax.bar([idx], [frac_infer_null], bottom=bottom, color=BAR_COLORS[1], edgecolor='black', width=bar_width)
            bottom += frac_infer_null
        if frac_base_null > 0:
            ax.bar([idx], [frac_base_null], bottom=bottom, color=BAR_COLORS[2], edgecolor='black', width=bar_width)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of rows', fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(valid_fields, rotation=45, ha='right', fontsize=10)
    ax.set_title('Metrics: gradient stack (bottom: precision, top: missing data)', fontsize=12, fontweight='bold')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('precision (high→low)', fontsize=9)
    
    handles = [
        mpatches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        mpatches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null vs base'),
        mpatches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='infer vs base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def create_percentage_table(
    out_path: Path,
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """
    Create a clean percentage table with metrics:
    - Precision: 1 - mean_distance (accuracy of extracted values)
    - Recall: n / (n + infer_null) (% of base values that were extracted)
    - F1: 2 * precision * recall / (precision + recall)
    - n: number of valid comparisons
    
    Missing tuple format: (both_null, infer_null, base_null)
    """
    # Order fields
    field_order = ["country", "year", "parameters", "hardware", "hardware_number", 
                   "training_time", "training_compute", "power_draw"]
    
    results = []
    for field in field_order:
        if field not in distances or not distances[field]:
            continue
        
        vals = distances[field]
        n = len(vals)
        mean_dist = np.mean(vals)
        
        # Precision: accuracy of values when both exist
        precision = 1.0 - mean_dist
        
        # Calculate recall if missing data is available
        # Recall = n / (n + infer_null) = % of base values that were successfully extracted
        if missing and field in missing:
            both_null, infer_null, base_null = missing[field]
            # Total base values = n + infer_null (cases where base exists)
            total_base = n + infer_null
            recall = n / total_base if total_base > 0 else 0.0
            # Coverage = n / total_rows
            total_rows = n + both_null + infer_null + base_null
            coverage = n / total_rows if total_rows > 0 else 0.0
        else:
            recall = 1.0  # Assume 100% if no missing data
            coverage = 1.0
        
        # F1 Score: harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        results.append({
            "field": field,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n": n
        })
    
    # Sort by F1 descending
    results.sort(key=lambda x: x["f1"], reverse=True)
    
    if not results:
        print(f"No data for {out_path}")
        return
    
    # Create the table
    n_rows = len(results)
    fig, ax = plt.subplots(figsize=(8, 0.4 + n_rows * 0.35))
    ax.axis("off")
    
    table_data = []
    for r in results:
        table_data.append([
            r["field"],
            f"{r['precision']*100:.1f}%",
            f"{r['recall']*100:.1f}%",
            f"{r['f1']*100:.1f}%",
            str(r["n"])
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["field", "precision", "recall", "F1", "n"],
        loc="center",
        cellLoc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    
    # Color by F1 score
    for row_idx, r in enumerate(results):
        rgba = plt.cm.RdYlGn(r["f1"])
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        for col_idx in range(5):
            table[row_idx + 1, col_idx].set_facecolor(color)
    
    for j in range(5):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", fontweight="bold")
    
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate epoch plots")
    parser.add_argument("--result-dir", type=str, default="results_epoch/arxiv_only",
                        help="Result directory")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    print(f"Processing: {result_dir}")
    
    # Load data
    distances, missing = load_epoch_data(result_dir)
    
    # Create overview directory
    overview_dir = result_dir / "overview"
    overview_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plot_grad_stacked_filtered(overview_dir / "grad_stacked_filtered.png", distances, missing)
    create_percentage_table(overview_dir / "percentage_table_clean.png", distances, missing)


if __name__ == "__main__":
    main()

