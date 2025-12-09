#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from core.lookups import CountryLookup, HardwareLookup
from core.metrics import (
    COLUMN_META, Metric, build_metrics, extract_scalar, parse_numeric,
    jw_similarity, signed_rel, is_missing, save_distance_summary
)
from core.plotting import plot_metrics_grid, plot_gradient_stacked, create_percentage_table


DATASET_RENAME = {"model name": "model"}


def load_dataset(
    dataset_path: Path,
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup
) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = df.rename(columns=DATASET_RENAME)
    df.insert(0, "id_paper", df.index + 1)

    if "country" in df.columns:
        df["id_country"] = df["country"].apply(country_lookup.get_id)
    if "hardware" in df.columns:
        df["id_hardware"] = df["hardware"].apply(hardware_lookup.get_id)

    keep_cols = ["id_paper"] + [col for col in COLUMN_META if col in df.columns]
    if "id_country" in df.columns:
        keep_cols.append("id_country")
    if "id_hardware" in df.columns:
        keep_cols.append("id_hardware")
    return df[keep_cols]


def load_inference(
    outputs_path: Path,
    tracked_fields: list[str],
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    if outputs_path.is_file():
        paths = [outputs_path]
    else:
        paths = sorted(outputs_path.glob("paper-*.json"))

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        for entry in data:
            pid = entry.get("id_paper")
            if pid is None:
                continue

            models = entry.get("models", [])
            if not models:
                has_direct = any(f in entry or f.replace("hardware_", "h_") in entry for f in tracked_fields)
                if has_direct:
                    models = [entry]
                else:
                    records.append({"id_paper": int(pid)})
                    continue

            for model in models:
                if not isinstance(model, dict):
                    continue
                cleaned = {k: extract_scalar(v) for k, v in model.items()}
                record: dict[str, Any] = {"id_paper": int(pid)}

                for field in tracked_fields:
                    if field.startswith("hardware_") and field.replace("hardware_", "h_") in cleaned:
                        record[field] = cleaned.get(field.replace("hardware_", "h_"))
                    else:
                        record[field] = cleaned.get(field)

                if "country" in record and record["country"]:
                    record["id_country"] = country_lookup.get_id(record["country"])

                if "hardware" in record and record["hardware"]:
                    record["id_hardware"] = hardware_lookup.get_id(record["hardware"])
                    if record.get("hardware_power") is None and record["id_hardware"]:
                        hw_power = hardware_lookup.get_power(record["id_hardware"])
                        if hw_power:
                            record["hardware_power"] = hw_power

                if record.get("power_draw") is None:
                    tt, hn, hp = record.get("training_time"), record.get("hardware_number"), record.get("hardware_power")
                    if tt and hn and hp:
                        try:
                            record["power_draw"] = float(tt) * float(hn) * float(hp) / 1000.0
                        except (ValueError, TypeError):
                            pass

                records.append(record)

    if not records:
        return pd.DataFrame(columns=["id_paper"] + tracked_fields + ["id_country", "id_hardware"])
    return pd.DataFrame(records).sort_values(["id_paper"]).reset_index(drop=True)


def compute_distances(
    merged: pd.DataFrame,
    metrics: list[Metric],
    infer_models: pd.DataFrame,
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]], list[dict[str, Any]]]:
    distances: dict[str, list[float]] = {m.field: [] for m in metrics}
    missing: dict[str, list[int]] = {m.field: [0, 0, 0] for m in metrics}
    merged_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        out_row: dict[str, Any] = {"id_paper": row["id_paper"]}

        for metric in metrics:
            field = metric.field
            miss = missing[field]

            if metric.kind == "id":
                id_col = "id_country" if field == "country" else "id_hardware" if field == "hardware" else None
                if not id_col:
                    continue

                base_id = row.get(f"{id_col}_base")
                if base_id is not None and not (isinstance(base_id, float) and math.isnan(base_id)):
                    base_id = int(base_id)
                else:
                    base_id = None

                models = infer_models[infer_models["id_paper"] == row["id_paper"]]
                inf_ids = models.get(id_col, pd.Series(dtype=object)).tolist() if not models.empty else [row.get(f"{id_col}_inf")]
                clean_inf_ids = [int(i) if i is not None and not (isinstance(i, float) and math.isnan(i)) else None for i in inf_ids]

                if base_id is None:
                    miss[0 if all(i is None for i in clean_inf_ids) else 2] += 1
                    continue

                has_any = any(i is not None for i in clean_inf_ids)
                if not has_any:
                    miss[1] += 1
                else:
                    dist = 0.0 if any(i == base_id for i in clean_inf_ids if i is not None) else 1.0
                    distances[field].append(dist)
                    out_row[f"{field}_dist"] = dist
                continue

            base_val = row.get(f"{field}_base")
            models = infer_models[infer_models["id_paper"] == row["id_paper"]]
            inf_candidates = models.get(field, pd.Series(dtype=object)).tolist() if not models.empty else [row.get(f"{field}_inf")]
            if not inf_candidates:
                inf_candidates = [None]

            if metric.kind == "numeric":
                base_num = parse_numeric(base_val)
                if base_num is None:
                    miss[0 if all(parse_numeric(v) is None for v in inf_candidates) else 2] += 1
                    continue

                best_dist = float("inf")
                for candidate in inf_candidates:
                    inf_num = parse_numeric(candidate)
                    if inf_num is not None:
                        rel = signed_rel(inf_num, base_num)
                        if rel is not None and abs(rel) < best_dist:
                            best_dist = abs(rel)

                if best_dist == float("inf"):
                    miss[1] += 1
                else:
                    distances[field].append(best_dist)
                    out_row[f"{field}_dist"] = best_dist

            else:
                base_txt = None if is_missing(base_val) else str(base_val).strip()
                if base_txt is None:
                    miss[0 if all(is_missing(v) for v in inf_candidates) else 2] += 1
                    continue

                best_dist = float("inf")
                for candidate in inf_candidates:
                    inf_txt = None if is_missing(candidate) else str(candidate).strip()
                    if inf_txt:
                        d = 1.0 - jw_similarity(base_txt, inf_txt)
                        if d < best_dist:
                            best_dist = d

                if best_dist == float("inf"):
                    miss[1] += 1
                else:
                    distances[field].append(best_dist)
                    out_row[f"{field}_dist"] = best_dist

        merged_rows.append(out_row)

    return distances, {k: tuple(v) for k, v in missing.items()}, merged_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs GreenMIR dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outputs-dir", required=True)
    parser.add_argument("--out-dir", default="results_id/benchmark_greenmir")
    parser.add_argument("--db", default="data/greenmir.db")
    parser.add_argument("--columns", nargs="*", choices=sorted(COLUMN_META.keys()))
    parser.add_argument("--all-papers", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    db_path = Path(args.db)
    out_dir.mkdir(parents=True, exist_ok=True)

    country_lookup = CountryLookup(db_path)
    hardware_lookup = HardwareLookup(db_path)
    print(f"[benchmark_greenmir] Loaded {len(country_lookup.countries)} countries, {len(hardware_lookup.hardware)} hardware")

    base_df = load_dataset(dataset_path, country_lookup, hardware_lookup)
    requested_columns = args.columns if args.columns else list(COLUMN_META.keys())
    tracked_fields = [f for f in requested_columns if f in base_df.columns]
    if not tracked_fields:
        raise SystemExit("[benchmark_greenmir] None of the requested columns exist in dataset.")

    infer_df = load_inference(outputs_dir, tracked_fields, country_lookup, hardware_lookup)

    if not args.all_papers:
        inferred_ids = set(infer_df["id_paper"].tolist())
        base_df = base_df[base_df["id_paper"].isin(inferred_ids)]

    if base_df.empty:
        print("[benchmark_greenmir] No overlapping papers between dataset and inference.")
        return

    merged = base_df.merge(infer_df, on="id_paper", how="left", suffixes=("_base", "_inf"))
    merged = merged.drop_duplicates(subset=["id_paper"], keep="first")

    common_fields = []
    for field in tracked_fields:
        if field == "country" and ("id_country_base" in merged.columns or "id_country_inf" in merged.columns):
            common_fields.append(field)
        elif field == "hardware" and ("id_hardware_base" in merged.columns or "id_hardware_inf" in merged.columns):
            common_fields.append(field)
        elif f"{field}_inf" in merged.columns:
            common_fields.append(field)

    metrics = build_metrics(common_fields)
    if not metrics:
        print("[benchmark_greenmir] No common fields between dataset and inference.")
        return

    distances, missing, merged_rows = compute_distances(merged, metrics, infer_df)

    merged.to_csv(out_dir / "merged.csv", index=False)
    plot_metrics_grid(out_dir / "benchmark.png", metrics, distances, missing)
    global_score = save_distance_summary(distances, out_dir / "distance_summary.csv")
    create_percentage_table(distances, out_dir / "percentage_table.png")

    overview_dir = out_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    plot_gradient_stacked(overview_dir / "grad_stacked_all.png", metrics, distances, missing)

    print(f"[benchmark_greenmir] Global distance: {global_score:.4f}")

    for field in ["country", "hardware"]:
        vals = distances.get(field, [])
        if vals:
            acc = sum(1 for v in vals if v == 0.0) / len(vals) * 100
            print(f"[benchmark_greenmir] {field.capitalize()} ID match: {acc:.1f}%")


if __name__ == "__main__":
    main()
