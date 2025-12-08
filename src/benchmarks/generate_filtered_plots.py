#!/usr/bin/env python3
"""
Generate filtered gradient stacked plots from existing benchmark results.
Removes specified columns and uses transparent background.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# Colors for null categories
BAR_COLORS = ["#16a34a", "#f59e0b", "#ef4444"]

# Columns to exclude
EXCLUDE_COLUMNS = {"model", "abstract", "hardware_compute", "training_compute", "power_draw", "co2eq"}


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
    
    # ID fields (country, hardware) - exact match (compare as integers)
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
        pass
    
    # Text fields - Jaro-Winkler (but we're excluding these anyway)
    return None


def load_benchmark_data(result_dir: Path) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]]]:
    """
    Load distances and missing counts from benchmark results.
    Returns (distances, missing) dictionaries.
    """
    # Load matched models CSV
    merged_path = result_dir / "merged.csv"
    if not merged_path.exists():
        raise FileNotFoundError(f"merged.csv not found in {result_dir}")
    
    df = pd.read_csv(merged_path)
    
    # Fields to process (all possible fields)
    all_fields = ["model", "abstract", "country", "year", "parameters", "hardware", 
                  "hardware_compute", "hardware_number", "hardware_power", 
                  "training_compute", "training_time", "power_draw", "co2eq"]
    
    # Build distances and missing dicts
    distances: dict[str, list[float]] = {}
    missing: dict[str, tuple[int, int, int]] = {}
    
    for field in all_fields:
        # Skip excluded columns
        if field in EXCLUDE_COLUMNS:
            continue
        
        # Get base and infer columns
        # For ID fields, use id_* columns if available
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


def plot_filtered_gradient_stacked(
    out_path: Path,
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]],
    n_bins: int = 10,
) -> None:
    """
    Create filtered gradient stacked bar plot with:
    - Excluded columns removed
    - Transparent background
    - "precision" instead of "normalized distance"
    """
    # Get fields that have data (excluding the ones we don't want)
    valid_fields = [
        f for f in distances.keys()
        if f not in EXCLUDE_COLUMNS and (distances.get(f) or sum(missing.get(f, (0, 0, 0))) > 0)
    ]
    
    if not valid_fields:
        print(f"No valid fields to plot for {out_path}")
        return
    
    # Order fields nicely
    field_order = ["country", "year", "parameters", "hardware", "hardware_number", "hardware_power", "training_time"]
    valid_fields = sorted(valid_fields, key=lambda x: field_order.index(x) if x in field_order else 999)
    
    n_metrics = len(valid_fields)
    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.2), 5))
    
    # Viridis: low distance (i=0) = purple (high precision), high distance (i=9) = yellow (low precision)
    cmap = plt.get_cmap('viridis')
    x_positions = np.arange(n_metrics)
    bar_width = 0.7
    
    for idx, field in enumerate(valid_fields):
        vals = distances.get(field, [])
        miss = missing.get(field, (0, 0, 0))
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
                # Low distance (i=0) = purple (high precision)
                # High distance (i=9) = yellow (low precision)
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
    ax.set_ylabel('fraction of rows', fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(valid_fields, rotation=45, ha='right', fontsize=10)
    ax.set_title('Metrics: gradient stack (bottom: precision, top: missing data)', fontsize=12, fontweight='bold')
    
    # Colorbar: purple (0) = high precision, yellow (1) = low precision
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('precision (high→low)', fontsize=9)
    
    # Legend
    handles = [
        mpatches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        mpatches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null vs base'),
        mpatches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='infer vs base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate filtered gradient stacked plots")
    parser.add_argument("--results-dir", type=str, default="results_id",
                        help="Directory containing benchmark results")
    parser.add_argument("--output-name", type=str, default="grad_stacked_filtered.png",
                        help="Output filename for the new plot")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Find all greenmir result directories
    greenmir_dirs = sorted(results_dir.glob("greenmir_*"))
    
    if not greenmir_dirs:
        print(f"No greenmir_* directories found in {results_dir}")
        return
    
    for result_dir in greenmir_dirs:
        print(f"\nProcessing: {result_dir.name}")
        
        try:
            distances, missing = load_benchmark_data(result_dir)
            
            # Create output path
            overview_dir = result_dir / "overview"
            overview_dir.mkdir(exist_ok=True)
            out_path = overview_dir / args.output_name
            
            plot_filtered_gradient_stacked(out_path, distances, missing)
            
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()

