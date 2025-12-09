import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler


COLUMN_META: dict[str, tuple[str, str]] = {
    "model": ("text", "Model (JW)"),
    "country": ("id", "Country (ID match)"),
    "year": ("numeric", "Year (rel)"),
    "parameters": ("numeric", "Parameters (rel)"),
    "hardware": ("id", "Hardware (ID match)"),
    "hardware_number": ("numeric", "Hardware number (rel)"),
    "hardware_compute": ("numeric", "Hardware compute (rel)"),
    "hardware_power": ("numeric", "Hardware power (rel)"),
    "training_compute": ("numeric", "Training compute (rel)"),
    "training_time": ("numeric", "Training time (rel)"),
    "power_draw": ("numeric", "Power draw (rel)"),
    "co2eq": ("numeric", "CO₂eq (rel)"),
    "h_number": ("numeric", "Hardware number (rel)"),
    "h_power": ("numeric", "Hardware power (rel)"),
    "h_compute": ("numeric", "Hardware compute (rel)"),
}


@dataclass
class Metric:
    field: str
    kind: str
    display_name: str

    @property
    def bins(self) -> Iterable[float]:
        if self.kind == "text":
            return np.linspace(0.0, 1.0, 26)
        if self.kind == "id":
            return [0, 1]
        return np.linspace(-1.0, 1.0, 41)

    @property
    def range_limits(self) -> tuple[float, float]:
        if self.kind == "id":
            return (0.0, 1.0)
        return (0.0, 1.0) if self.kind == "text" else (-1.0, 1.0)


def build_metrics(fields: list[str]) -> list[Metric]:
    return [
        Metric(field=f, kind=COLUMN_META[f][0], display_name=COLUMN_META[f][1])
        for f in fields if f in COLUMN_META
    ]


def jw_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    score = JaroWinkler.normalized_similarity(str(a), str(b))
    return score / 100.0 if score > 1.0 else score


def signed_rel(infer_val: float | None, base_val: float | None) -> float | None:
    if infer_val is None or base_val is None:
        return None
    denom = max(abs(infer_val), abs(base_val), 1.0)
    if denom <= 0:
        return None
    return (infer_val - base_val) / denom


_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_numeric(value: Any) -> float | None:
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


def is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float):
        try:
            if math.isnan(val):
                return True
        except Exception:
            pass
    if isinstance(val, str):
        return not val.strip()
    return False


def extract_scalar(value: Any) -> Any:
    if isinstance(value, list):
        val = value[0] if value else None
        return None if val == "" else val
    return value


def save_distance_summary(distances: dict[str, list[float]], out_path) -> float:
    rows = []
    total_abs = 0.0
    total_count = 0
    all_abs_vals = []

    for field, vals in distances.items():
        if not vals:
            mean_abs = None
            std_abs = None
            count = 0
        else:
            abs_vals = [abs(v) for v in vals]
            mean_abs = float(np.mean(abs_vals))
            std_abs = float(np.std(abs_vals))
            count = len(vals)
            total_abs += sum(abs_vals)
            total_count += count
            all_abs_vals.extend(abs_vals)
        rows.append({"field": field, "mean_abs_distance": mean_abs, "std_abs_distance": std_abs, "count": count})

    global_score = (total_abs / total_count) if total_count else float("nan")
    global_std = float(np.std(all_abs_vals)) if all_abs_vals else float("nan")
    rows.append({"field": "GLOBAL_WEIGHTED_AVG", "mean_abs_distance": global_score, "std_abs_distance": global_std, "count": total_count})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return global_score


def compute_distance(base_val: Any, inf_val: Any, field: str) -> float | None:
    if is_missing(base_val) or is_missing(inf_val):
        return None
    if field in ("country", "hardware"):
        try:
            return 0.0 if int(float(base_val)) == int(float(inf_val)) else 1.0
        except (ValueError, TypeError):
            return 0.0 if str(base_val).strip() == str(inf_val).strip() else 1.0
    if field == "year":
        try:
            return min(1.0, abs(float(base_val) - float(inf_val)) / 5.0)
        except (ValueError, TypeError):
            return None
    try:
        b, i = float(base_val), float(inf_val)
        if b == 0 and i == 0:
            return 0.0
        denom = max(abs(b), abs(i))
        return min(1.0, abs(b - i) / denom) if denom > 0 else None
    except (ValueError, TypeError):
        return None


def load_results_data(
    result_dir,
    csv_name: str = "matched_models.csv",
    exclude_fields: set[str] | None = None
) -> tuple[dict[str, list[float]], dict[str, tuple[int, int, int]]]:
    from pathlib import Path
    result_dir = Path(result_dir)
    csv_path = result_dir / csv_name
    if not csv_path.exists():
        csv_path = result_dir / "merged.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found in {result_dir}")

    df = pd.read_csv(csv_path)
    exclude = exclude_fields or set()

    all_fields = ["country", "year", "parameters", "hardware", "hardware_number",
                  "hardware_power", "training_time", "training_compute", "power_draw", "co2eq"]
    distances: dict[str, list[float]] = {}
    missing: dict[str, tuple[int, int, int]] = {}

    for field in all_fields:
        if field in exclude:
            continue
        if field in ("country", "hardware"):
            base_col = f"id_{field}_base" if f"id_{field}_base" in df.columns else f"{field}_base"
            inf_col = f"id_{field}_inf" if f"id_{field}_inf" in df.columns else f"{field}_inf"
        else:
            base_col, inf_col = f"{field}_base", f"{field}_inf"
        if base_col not in df.columns or inf_col not in df.columns:
            continue

        both_null, infer_null, base_null = 0, 0, 0
        dist_vals = []
        for _, row in df.iterrows():
            bv, iv = row.get(base_col), row.get(inf_col)
            bm, im = is_missing(bv), is_missing(iv)
            if bm and im:
                both_null += 1
            elif im:
                infer_null += 1
            elif bm:
                base_null += 1
            else:
                d = compute_distance(bv, iv, field)
                if d is not None:
                    dist_vals.append(abs(d))
        distances[field] = dist_vals
        missing[field] = (both_null, infer_null, base_null)

    return distances, missing


def compute_f1_stats(
    distances: dict[str, list[float]],
    missing: dict[str, tuple[int, int, int]]
) -> list[dict[str, Any]]:
    results = []
    for field, vals in distances.items():
        if not vals:
            continue
        n = len(vals)
        mean_dist = float(np.mean(vals))
        precision = 1.0 - mean_dist
        miss = missing.get(field, (0, 0, 0))
        total_base = n + miss[1]
        recall = n / total_base if total_base > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append({"field": field, "precision": precision, "recall": recall, "f1": f1, "n": n})
    results.sort(key=lambda x: x["f1"], reverse=True)
    return results

