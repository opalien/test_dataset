#!/usr/bin/env python3
import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from core.lookups import CountryLookup, HardwareLookup
from core.matching import ModelMatch, match_models_for_paper
from core.metrics import (
    COLUMN_META, Metric, build_metrics, extract_scalar, parse_numeric,
    jw_similarity, signed_rel, is_missing, save_distance_summary
)
from core.plotting import plot_metrics_grid, plot_gradient_stacked, create_percentage_table


def load_ground_truth(
    db_path: Path,
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup,
    arxiv_only: bool = False,
    confidence_filter: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(str(db_path))

    confidence_df = pd.read_sql_query("SELECT Link as link, Confidence as confidence FROM epoch WHERE Link IS NOT NULL", conn)

    query = "SELECT id_paper, link, abstract, country, id_country, year FROM paper_info"
    if arxiv_only:
        query += " WHERE link LIKE '%arxiv%'"
    paper_df = pd.read_sql_query(query, conn)

    confidence_df = confidence_df.drop_duplicates(subset=["link"], keep="first")
    paper_df = paper_df.merge(confidence_df, on="link", how="left")
    paper_df["confidence"] = paper_df["confidence"].fillna("Unknown")
    paper_df = paper_df.drop_duplicates(subset=["id_paper"], keep="first")

    if confidence_filter:
        paper_df = paper_df[paper_df["confidence"] == confidence_filter]

    valid_ids = set(paper_df["id_paper"].tolist())

    model_df = pd.read_sql_query("""
        SELECT id_model, id_paper, model, architecture, parameters,
               id_hardware, hardware, hardware_number, training_compute, training_time, power_draw
        FROM model_info
    """, conn)
    conn.close()

    model_df = model_df[model_df["id_paper"].isin(valid_ids)]

    if "id_country" in paper_df.columns:
        paper_df["id_country"] = paper_df.apply(
            lambda r: int(r["id_country"]) if pd.notna(r.get("id_country")) else country_lookup.get_id(r.get("country")), axis=1
        )

    if "id_hardware" in model_df.columns:
        model_df["id_hardware"] = model_df.apply(
            lambda r: int(r["id_hardware"]) if pd.notna(r.get("id_hardware")) else hardware_lookup.get_id(r.get("hardware")), axis=1
        )

    def calc_power_draw(row: pd.Series) -> float | None:
        if pd.notna(row.get("power_draw")) and row.get("power_draw") > 0:
            return row["power_draw"]
        hn, hid = row.get("hardware_number"), row.get("id_hardware")
        if pd.isna(hn) or pd.isna(hid):
            return None
        hp = hardware_lookup.get_power(int(hid))
        if hp is None:
            return None
        year = paper_df[paper_df["id_paper"] == row["id_paper"]]["year"].iloc[0] if row["id_paper"] in paper_df["id_paper"].values else 2024
        if pd.isna(year):
            year = 2024
        pue = 1.23 * math.exp((year - 2008) * math.log(1.08 / 1.23) / 16) if year >= 2009 else 1.23
        server_overhead = 1.82 if float(hn) > 1 else 1.0
        return pue * server_overhead * hp * float(hn)

    model_df["power_draw"] = model_df.apply(calc_power_draw, axis=1)
    return paper_df, model_df


def load_inference(
    outputs_path: Path,
    tracked_fields: list[str],
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    paths = [outputs_path] if outputs_path.is_file() else sorted(outputs_path.glob("*.json"))

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
    metrics: list[Metric]
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]], list[dict[str, Any]]]:
    distances: dict[str, list[float]] = {m.field: [] for m in metrics}
    missing: dict[str, list[int]] = {m.field: [0, 0, 0] for m in metrics}
    result_rows: list[dict[str, Any]] = []
    paper_info = paper_df.set_index("id_paper").to_dict("index") if len(paper_df) > 0 else {}

    for match in matches:
        base, inf = match.base_model, match.inf_model
        pid = base.get("id_paper") or inf.get("id_paper")
        pinfo = paper_info.get(pid, {})
        out_row: dict[str, Any] = {"id_paper": pid, "model_base": base.get("model"), "model_inf": inf.get("model"), "match_score": match.score}

        for metric in metrics:
            field = metric.field
            miss = missing[field]
            base_val = pinfo.get(field) if field in ("country", "year") and field not in base else base.get(field)
            inf_val = inf.get(field)

            if metric.kind == "id":
                id_col = "id_country" if field == "country" else "id_hardware" if field == "hardware" else None
                if not id_col:
                    continue
                base_id = pinfo.get(id_col) if field == "country" else base.get(id_col)
                inf_id = inf.get(id_col)

                base_id = int(base_id) if base_id is not None and not (isinstance(base_id, float) and math.isnan(base_id)) else None
                inf_id = int(inf_id) if inf_id is not None and not (isinstance(inf_id, float) and math.isnan(inf_id)) else None

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
                base_num, inf_num = parse_numeric(base_val), parse_numeric(inf_val)
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


def run_benchmark(
    db_path: Path, outputs_path: Path, out_dir: Path, tracked_fields: list[str],
    country_lookup: CountryLookup, hardware_lookup: HardwareLookup,
    arxiv_only: bool, confidence_filter: str | None, label: str
) -> dict[str, Any] | None:
    paper_df, model_df = load_ground_truth(db_path, country_lookup, hardware_lookup, arxiv_only, confidence_filter)
    if paper_df.empty or model_df.empty:
        print(f"[benchmark_epoch/{label}] No data; skipping.")
        return None

    infer_df = load_inference(outputs_path, tracked_fields, country_lookup, hardware_lookup)
    infer_df = infer_df[infer_df["id_paper"].isin(set(paper_df["id_paper"].tolist()))]
    if infer_df.empty:
        print(f"[benchmark_epoch/{label}] No inference data; skipping.")
        return None

    all_matches: list[ModelMatch] = []
    matched_papers = 0

    for pid in set(infer_df["id_paper"].unique()):
        base_models = model_df[model_df["id_paper"] == pid].to_dict("records")
        if not base_models:
            continue
        inf_models = infer_df[infer_df["id_paper"] == pid].to_dict("records")
        if not inf_models:
            continue
        matches = match_models_for_paper(base_models, inf_models)
        if matches:
            all_matches.extend(matches)
            matched_papers += 1

    print(f"[benchmark_epoch/{label}] Matched {len(all_matches)} pairs from {matched_papers} papers")
    if not all_matches:
        return None

    metrics = build_metrics(tracked_fields)
    distances, missing, result_rows = compute_distances(all_matches, paper_df, metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(out_dir / "matched_models.csv", index=False)
    plot_metrics_grid(out_dir / "benchmark.png", metrics, distances, missing)
    global_score = save_distance_summary(distances, out_dir / "distance_summary.csv")
    create_percentage_table(distances, out_dir / "percentage_table.png")

    overview_dir = out_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    plot_gradient_stacked(overview_dir / "grad_stacked_all.png", metrics, distances, missing)

    print(f"[benchmark_epoch/{label}] Global distance: {global_score:.4f}")
    return {"label": label, "papers": len(paper_df), "models": len(model_df), "matched_pairs": len(all_matches), "matched_papers": matched_papers, "global_distance": global_score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs Epoch AI dataset.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out-dir", default="results_epoch/benchmark")
    parser.add_argument("--columns", nargs="*", choices=sorted(COLUMN_META.keys()))
    parser.add_argument("--arxiv-only", action="store_true")
    parser.add_argument("--no-confidence-split", action="store_true")
    args = parser.parse_args()

    db_path, outputs_path, out_dir = Path(args.database), Path(args.outputs), Path(args.out_dir)
    country_lookup, hardware_lookup = CountryLookup(db_path), HardwareLookup(db_path)
    print(f"[benchmark_epoch] Loaded {len(country_lookup.countries)} countries, {len(hardware_lookup.hardware)} hardware")

    all_fields = ["model", "parameters", "hardware", "hardware_number", "training_time", "training_compute", "power_draw", "country", "year"]
    tracked_fields = [f for f in args.columns if f in all_fields] if args.columns else all_fields

    configs = [{"confidence_filter": None, "label": "all", "subdir": "all"}]
    if not args.no_confidence_split:
        configs.extend([
            {"confidence_filter": "Confident", "label": "confident", "subdir": "confident"},
            {"confidence_filter": "Likely", "label": "likely", "subdir": "likely"},
            {"confidence_filter": "Speculative", "label": "speculative", "subdir": "speculative"},
        ])

    summaries = []
    for cfg in configs:
        result = run_benchmark(db_path, outputs_path, out_dir / cfg["subdir"], tracked_fields, country_lookup, hardware_lookup, args.arxiv_only, cfg["confidence_filter"], cfg["label"])
        if result:
            summaries.append(result)

    if summaries:
        print(f"\n{'='*55}")
        print(f"{'Config':<15} {'Papers':>8} {'Models':>8} {'Matched':>8} {'Distance':>10}")
        for s in summaries:
            print(f"{s['label']:<15} {s['papers']:>8} {s['models']:>8} {s['matched_pairs']:>8} {s['global_distance']:>10.4f}")
        pd.DataFrame(summaries).to_csv(out_dir / "confidence_summary.csv", index=False)


if __name__ == "__main__":
    main()
