#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from core.lookups import CountryLookup, HardwareLookup
from core.matching import ModelMatch, match_models_for_paper
from core.metrics import (
    COLUMN_META, Metric, build_metrics, extract_scalar,
    jw_similarity, parse_numeric, is_missing, signed_rel, save_distance_summary
)
from core.plotting import plot_metrics_grid, create_percentage_table


def load_ground_truth(
    db_path: Path,
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import sqlite3
    conn = sqlite3.connect(str(db_path))

    try:
        paper_df = pd.read_sql_query("""
            SELECT id_paper, link, abstract, country, id_country, year, split
            FROM paper_info
        """, conn)
    except Exception:
        paper_df = pd.DataFrame(columns=["id_paper", "link", "abstract", "country", "id_country", "year", "split"])

    try:
        model_df = pd.read_sql_query("""
            SELECT id_model, id_paper, model, architecture, parameters,
                   id_hardware, hardware, hardware_compute, hardware_number,
                   hardware_power, training_compute, training_time, power_draw, co2eq
            FROM model_info
        """, conn)
    except Exception:
        model_df = pd.DataFrame(columns=["id_model", "id_paper", "model", "parameters", "hardware"])

    conn.close()

    if "id_country" in paper_df.columns:
        def ensure_country_id(row: pd.Series) -> int | None:
            if pd.notna(row.get("id_country")):
                return int(row["id_country"])
            return country_lookup.get_id(row.get("country"))
        paper_df["id_country"] = paper_df.apply(ensure_country_id, axis=1)

    if "id_hardware" in model_df.columns:
        def ensure_hardware_id(row: pd.Series) -> int | None:
            if pd.notna(row.get("id_hardware")):
                return int(row["id_hardware"])
            return hardware_lookup.get_id(row.get("hardware"))
        model_df["id_hardware"] = model_df.apply(ensure_hardware_id, axis=1)

    return paper_df, model_df


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
        paths = sorted(outputs_path.glob("*.json"))

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
                records.append({"id_paper": int(pid)})
                continue

            for model in models:
                if not isinstance(model, dict):
                    continue

                cleaned = {k: extract_scalar(v) for k, v in model.items()}
                record: dict[str, Any] = {"id_paper": int(pid), "model": cleaned.get("model")}

                for field in tracked_fields:
                    if field.startswith("hardware_") and field.replace("hardware_", "h_") in cleaned:
                        record[field] = cleaned.get(field.replace("hardware_", "h_"))
                    else:
                        record[field] = cleaned.get(field)

                if "country" in record and record["country"]:
                    record["id_country"] = country_lookup.get_id(record["country"])

                if "hardware" in record and record["hardware"]:
                    record["id_hardware"] = hardware_lookup.get_id(record["hardware"])

                records.append(record)

    if not records:
        return pd.DataFrame(columns=["id_paper", "model"] + tracked_fields)

    return pd.DataFrame(records).sort_values(["id_paper"]).reset_index(drop=True)


def compute_distances(
    matches: list[ModelMatch],
    paper_df: pd.DataFrame,
    metrics: list[Metric],
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]], list[dict[str, Any]]]:
    distances: dict[str, list[float]] = {m.field: [] for m in metrics}
    missing: dict[str, list[int]] = {m.field: [0, 0, 0] for m in metrics}
    result_rows: list[dict[str, Any]] = []

    paper_info = paper_df.set_index("id_paper").to_dict("index") if len(paper_df) > 0 else {}

    for match in matches:
        base = match.base_model
        inf = match.inf_model
        pid = base.get("id_paper") or inf.get("id_paper")
        pinfo = paper_info.get(pid, {})

        out_row: dict[str, Any] = {
            "id_paper": pid,
            "model_base": base.get("model"),
            "model_inf": inf.get("model"),
            "match_score": match.score,
        }

        for metric in metrics:
            field = metric.field
            miss = missing[field]

            if field in ("country", "year") and field not in base:
                base_val = pinfo.get(field)
            else:
                base_val = base.get(field)

            inf_val = inf.get(field)

            if metric.kind == "id":
                if field == "country":
                    base_id = pinfo.get("id_country")
                    inf_id = inf.get("id_country")
                elif field == "hardware":
                    base_id = base.get("id_hardware")
                    inf_id = inf.get("id_hardware")
                else:
                    continue

                if base_id is not None and not (isinstance(base_id, float) and math.isnan(base_id)):
                    base_id = int(base_id)
                else:
                    base_id = None

                if inf_id is not None and not (isinstance(inf_id, float) and math.isnan(inf_id)):
                    inf_id = int(inf_id)
                else:
                    inf_id = None

                out_row[f"{field}_base"] = base_val
                out_row[f"{field}_inf"] = inf_val

                if base_id is None:
                    miss[0 if inf_id is None else 2] += 1
                    continue
                if inf_id is None:
                    miss[1] += 1
                    continue

                dist = 0.0 if base_id == inf_id else 1.0
                distances[field].append(dist)
                out_row[f"{field}_dist"] = dist

            elif metric.kind == "numeric":
                base_num = parse_numeric(base_val)
                inf_num = parse_numeric(inf_val)

                out_row[f"{field}_base"] = base_num
                out_row[f"{field}_inf"] = inf_num

                if base_num is None:
                    miss[0 if inf_num is None else 2] += 1
                    continue
                if inf_num is None:
                    miss[1] += 1
                    continue

                rel = signed_rel(inf_num, base_num)
                if rel is None:
                    miss[1] += 1
                    continue

                distances[field].append(abs(rel))
                out_row[f"{field}_dist"] = abs(rel)

            else:
                base_txt = None if is_missing(base_val) else str(base_val).strip()
                inf_txt = None if is_missing(inf_val) else str(inf_val).strip()

                out_row[f"{field}_base"] = base_txt
                out_row[f"{field}_inf"] = inf_txt

                if base_txt is None:
                    miss[0 if inf_txt is None else 2] += 1
                    continue
                if inf_txt is None:
                    miss[1] += 1
                    continue

                jw_dist = 1.0 - jw_similarity(base_txt, inf_txt)
                distances[field].append(jw_dist)
                out_row[f"{field}_dist"] = jw_dist

        result_rows.append(out_row)

    return distances, {k: tuple(v) for k, v in missing.items()}, result_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs folder-based dataset.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out-dir", default="results_folder/benchmark")
    parser.add_argument("--columns", nargs="*", choices=sorted(COLUMN_META.keys()))
    args = parser.parse_args()

    db_path = Path(args.database)
    outputs_path = Path(args.outputs)
    out_dir = Path(args.out_dir)

    country_lookup = CountryLookup(db_path)
    hardware_lookup = HardwareLookup(db_path)
    print(f"[benchmark_folder] Loaded {len(country_lookup.countries)} countries, {len(hardware_lookup.hardware)} hardware")

    all_fields = ["model", "parameters", "hardware", "hardware_number", "training_time", "training_compute", "power_draw", "co2eq", "country", "year"]
    tracked_fields = [f for f in args.columns if f in all_fields] if args.columns else all_fields

    paper_df, model_df = load_ground_truth(db_path, country_lookup, hardware_lookup)

    if len(paper_df) == 0 and len(model_df) == 0:
        print("[benchmark_folder] No ground truth data found")
        return

    print(f"[benchmark_folder] Loaded {len(paper_df)} papers, {len(model_df)} models")

    infer_df = load_inference(outputs_path, tracked_fields, country_lookup, hardware_lookup)
    if len(infer_df) == 0:
        print("[benchmark_folder] No inference data found")
        return

    all_matches: list[ModelMatch] = []
    matched_papers = 0

    for pid in set(infer_df["id_paper"].unique()):
        base_models = model_df[model_df["id_paper"] == pid].to_dict("records") if len(model_df) > 0 else []
        if not base_models:
            continue

        inf_models = infer_df[infer_df["id_paper"] == pid].to_dict("records")
        if not inf_models:
            continue

        matches = match_models_for_paper(base_models, inf_models)
        if matches:
            all_matches.extend(matches)
            matched_papers += 1

    print(f"[benchmark_folder] Matched {len(all_matches)} model pairs from {matched_papers} papers")

    if not all_matches:
        print("[benchmark_folder] No matches found")
        return

    metrics = build_metrics(tracked_fields)
    distances, missing, result_rows = compute_distances(all_matches, paper_df, metrics)

    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result_rows).to_csv(out_dir / "matched_models.csv", index=False)
    plot_metrics_grid(out_dir / "benchmark.png", metrics, distances, missing)
    global_score = save_distance_summary(distances, out_dir / "distance_summary.csv")
    create_percentage_table(distances, out_dir / "percentage_table.png")

    print(f"[benchmark_folder] Global distance: {global_score:.4f}")


if __name__ == "__main__":
    main()
