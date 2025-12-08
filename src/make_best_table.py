#!/usr/bin/env python3
"""
Generate tables showing precision, recall, and F1 metrics for greenmir benchmarks.

Metrics explained:
- Precision: 1 - mean_distance (how accurate are the extracted values)
- Recall: n / (n + infer_null) (what % of ground truth values did we extract)
- F1: 2 * precision * recall / (precision + recall) (harmonic mean)
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _is_missing(val) -> bool:
    """Check if a value is missing/null."""
    if val is None:
        return True
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "null", "n/a", "na")


def _compute_distance(base_val, inf_val, field: str) -> float | None:
    """Compute distance using the same logic as grad_stacked_filtered."""
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


def _get_field_stats(result_dir: Path, field: str) -> dict | None:
    """Get precision, recall, F1 stats for a field."""
    merged_path = result_dir / "merged.csv"
    if not merged_path.exists():
        return None
    
    df = pd.read_csv(merged_path)
    
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
        return None
    
    # Count categories
    n_both = 0  # Both have values
    n_infer_null = 0  # Base has value, inf is null (missed extraction)
    n_base_null = 0  # Inf has value, base is null
    n_both_null = 0  # Both null
    
    distances = []
    
    for _, row in df.iterrows():
        base_val = row.get(base_col)
        inf_val = row.get(inf_col)
        
        base_missing = _is_missing(base_val)
        inf_missing = _is_missing(inf_val)
        
        if base_missing and inf_missing:
            n_both_null += 1
        elif base_missing:
            n_base_null += 1
        elif inf_missing:
            n_infer_null += 1
        else:
            n_both += 1
            dist = _compute_distance(base_val, inf_val, field)
            if dist is not None:
                distances.append(dist)
    
    if not distances:
        return None
    
    n = len(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Precision: accuracy of extracted values
    precision = 1.0 - mean_dist
    
    # Recall: % of ground truth values that were extracted
    total_base = n + n_infer_null  # Cases where base has a value
    recall = n / total_base if total_base > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "field": field,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "std": std_dist,
        "n": n
    }


def _create_table(results: list, out_path: Path):
    """Create a clean table PNG with precision, recall, F1."""
    if not results:
        print(f"No data for {out_path}")
        return
    
    # Sort by F1 descending
    results.sort(key=lambda x: x["f1"], reverse=True)
    
    # Create the table
    n_rows = len(results)
    fig, ax = plt.subplots(figsize=(8, 0.4 + n_rows * 0.35))
    ax.axis("off")
    
    # Table data
    table_data = []
    for r in results:
        table_data.append([
            r["field"],
            f"{r['precision']*100:.1f}%",
            f"{r['recall']*100:.1f}%",
            f"{r['f1']*100:.1f}%",
            str(r["n"])
        ])
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["field", "precision", "recall", "F1", "n"],
        loc="center",
        cellLoc="center"
    )
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)
    
    # Set colors for each row based on F1
    for row_idx, r in enumerate(results):
        rgba = plt.cm.RdYlGn(r["f1"])
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        for col_idx in range(5):
            table[row_idx + 1, col_idx].set_facecolor(color)
    
    # Set header style
    for j in range(5):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", fontweight="bold")
    
    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Save
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_single_table(result_dir: Path):
    """Generate a percentage table for a single result directory."""
    merged_path = result_dir / "merged.csv"
    if not merged_path.exists():
        print(f"No merged.csv in {result_dir}")
        return
    
    # Fields to include (same as grad_stacked_filtered)
    fields = ["country", "year", "parameters", "hardware", "hardware_number", "hardware_power", "training_time"]
    
    results = []
    for field in fields:
        stats = _get_field_stats(result_dir, field)
        if stats:
            results.append(stats)
    
    out_path = result_dir / "overview" / "percentage_table_clean.png"
    out_path.parent.mkdir(exist_ok=True)
    _create_table(results, out_path)


def make_best_table():
    """Generate a table showing the best F1 for each field across all prompt types."""
    # Models to compare
    models = ["greenmir_strict_high", "greenmir_derived_high", "greenmir_estimated_high"]
    
    # Fields in the order of grad_stacked_filtered.png
    fields_order = ["country", "year", "parameters", "hardware", "hardware_number", "hardware_power", "training_time"]
    
    # Find best for each field
    best_results = []
    
    for field in fields_order:
        best_f1 = -1
        best_stats = None
        
        for model in models:
            result_dir = Path(f"results_id/{model}")
            stats = _get_field_stats(result_dir, field)
            
            if stats and stats["f1"] > best_f1:
                best_f1 = stats["f1"]
                best_stats = stats
        
        if best_stats:
            best_results.append(best_stats)
    
    _create_table(best_results, Path("results_id/best_precision_table.png"))


def main():
    parser = argparse.ArgumentParser(description="Generate precision/recall/F1 tables")
    parser.add_argument("--result-dir", type=str, help="Generate table for specific result directory")
    parser.add_argument("--all-greenmir", action="store_true", help="Generate tables for all greenmir results")
    parser.add_argument("--best", action="store_true", help="Generate best table across models")
    args = parser.parse_args()
    
    if args.result_dir:
        make_single_table(Path(args.result_dir))
    elif args.all_greenmir:
        models = ["greenmir_strict", "greenmir_derived", "greenmir_estimated",
                  "greenmir_strict_high", "greenmir_derived_high", "greenmir_estimated_high"]
        for model in models:
            result_dir = Path(f"results_id/{model}")
            if result_dir.exists():
                print(f"Processing {model}...")
                make_single_table(result_dir)
        make_best_table()
    elif args.best:
        make_best_table()
    else:
        # Default: generate all greenmir tables
        models = ["greenmir_strict", "greenmir_derived", "greenmir_estimated",
                  "greenmir_strict_high", "greenmir_derived_high", "greenmir_estimated_high"]
        for model in models:
            result_dir = Path(f"results_id/{model}")
            if result_dir.exists():
                print(f"Processing {model}...")
                make_single_table(result_dir)
        make_best_table()


if __name__ == "__main__":
    main()
