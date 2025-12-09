import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
import numpy as np

from .metrics import Metric


BAR_COLORS = ["#16a34a", "#f59e0b", "#ef4444"]
HIST_COLOR = "#3b82f6"


def draw_pair(
    ax_left,
    ax_right,
    distances: list[float],
    missing: tuple[int, int, int],
    bins,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    categories = ["both null", "infer null", "base null"]
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


def plot_metrics_grid(
    out_path: Path,
    metrics: list[Metric],
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]],
) -> None:
    if not metrics:
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
        draw_pair(ax_left, ax_right, vals, miss, bins=metric.bins, vmin=metric.range_limits[0], vmax=metric.range_limits[1], title=metric.display_name)

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(BAR_COLORS, ["both null", "infer null", "base null"], strict=True)
    ]
    legend_handles.append(mpatches.Patch(facecolor=HIST_COLOR, edgecolor="black", label="distance"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradient_stacked(
    out_path: Path,
    metrics: list[Metric],
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]],
    n_bins: int = 10,
) -> None:
    valid_metrics = [m for m in metrics if distances.get(m.field) or sum(missing.get(m.field, (0, 0, 0))) > 0]
    if not valid_metrics:
        return

    n_metrics = len(valid_metrics)
    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.5), 6))

    cmap = plt.get_cmap('viridis')
    x_positions = np.arange(n_metrics)
    bar_width = 0.7

    for idx, metric in enumerate(valid_metrics):
        dists = distances.get(metric.field, [])
        miss = missing.get(metric.field, (0, 0, 0))
        total_rows = len(dists) + sum(miss)
        if total_rows == 0:
            continue

        bottom = 0.0

        if dists:
            abs_distances = [abs(d) for d in dists]
            counts, _ = np.histogram(abs_distances, bins=n_bins, range=(0.0, 1.0))
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                height = count / total_rows
                color = cmap(i / max(1, n_bins - 1))
                ax.bar([idx], [height], bottom=bottom, color=color, edgecolor='none', width=bar_width)
                bottom += height

        frac_both_null = miss[0] / total_rows
        frac_infer_null = miss[1] / total_rows
        frac_base_null = miss[2] / total_rows

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
    ax.set_xticklabels([m.display_name for m in valid_metrics], rotation=45, ha='right')
    ax.set_title('Gradient Stacked: Distance Distribution + Nulls')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('normalized distance', fontsize=8)

    handles = [
        mpatches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        mpatches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null'),
        mpatches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def create_percentage_table(distances: dict[str, list[float]], out_path: Path) -> None:
    table_data = []
    for field, vals in distances.items():
        if vals:
            abs_vals = [abs(v) for v in vals]
            mean_dist = float(np.mean(abs_vals))
            std_dist = float(np.std(abs_vals))
            accuracy_pct = (1.0 - mean_dist) * 100.0
            table_data.append({
                "field": field,
                "accuracy_pct": accuracy_pct,
                "distance_pct": mean_dist * 100.0,
                "std_pct": std_dist * 100.0,
                "count": len(vals)
            })

    if not table_data:
        return

    table_data.sort(key=lambda x: x["accuracy_pct"], reverse=True)

    fig, ax = plt.subplots(figsize=(12, max(3, len(table_data) * 0.5 + 1.5)))
    ax.axis('off')

    col_labels = ["Field", "Accuracy (%)", "Distance (%)", "Std (%)", "N"]
    cell_text = []
    cell_colors = []

    for row in table_data:
        acc = row["accuracy_pct"]
        if acc >= 90:
            color = "#d4edda"
        elif acc >= 70:
            color = "#fff3cd"
        elif acc >= 50:
            color = "#ffeeba"
        else:
            color = "#f8d7da"
        cell_text.append([
            row["field"],
            f"{acc:.1f}",
            f"{row['distance_pct']:.1f}",
            f"{row['std_pct']:.1f}",
            str(row["count"])
        ])
        cell_colors.append([color] * 5)

    table = ax.table(cellText=cell_text, colLabels=col_labels, cellColours=cell_colors, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_f1_table(
    stats: list[dict],
    out_path: Path,
    transparent: bool = True
) -> None:
    if not stats:
        return

    fig, ax = plt.subplots(figsize=(8, 0.4 + len(stats) * 0.35))
    ax.axis("off")

    table_data = [
        [r["field"], f"{r['precision']*100:.1f}%", f"{r['recall']*100:.1f}%", f"{r['f1']*100:.1f}%", str(r["n"])]
        for r in stats
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["field", "precision", "recall", "F1", "n"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.4)

    for row_idx, r in enumerate(stats):
        rgba = plt.cm.RdYlGn(r["f1"])
        color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
        for col_idx in range(5):
            table[row_idx + 1, col_idx].set_facecolor(color)

    for j in range(5):
        table[0, j].set_facecolor("#404040")
        table[0, j].set_text_props(color="white", fontweight="bold")

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=transparent, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_gradient_stacked_filtered(
    out_path: Path,
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]],
    field_order: list[str] | None = None,
    n_bins: int = 10,
    transparent: bool = True
) -> None:
    if field_order is None:
        field_order = ["country", "year", "parameters", "hardware", "hardware_number",
                       "hardware_power", "training_time", "training_compute", "power_draw"]

    valid_fields = [f for f in field_order if f in distances and (distances.get(f) or sum(missing.get(f, (0, 0, 0))) > 0)]
    if not valid_fields:
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

        for frac, c in zip([miss[0]/total_rows, miss[1]/total_rows, miss[2]/total_rows], BAR_COLORS, strict=True):
            if frac > 0:
                ax.bar([idx], [frac], bottom=bottom, color=c, edgecolor='black', width=bar_width)
                bottom += frac

    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction of rows', fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(valid_fields, rotation=45, ha='right', fontsize=10)
    ax.set_title('Metrics: gradient stack (bottom: precision, top: missing)', fontsize=12, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('precision (high→low)', fontsize=9)

    handles = [mpatches.Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(BAR_COLORS, ['both null', 'infer null', 'base null'], strict=True)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=transparent)
    plt.close(fig)

