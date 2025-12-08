#!/usr/bin/env python3
"""
Benchmark inferred results against a custom dataset created by make_table_from_folder.py.

- Reads ground truth from a SQLite DB (model_info + paper_info tables).
- Reads inference outputs from JSON files.
- Matches inferred models to ground truth models using a robust matching algorithm.
- For country field: maps text to ID using the country table via Jaro-Winkler.
- For hardware field: maps text to ID using the hardware table via token-based matching.
- Computes distances and generates plots.

This is a simplified version of benchmark_epoch_id.py for custom folder-based datasets.
"""
import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz as rf_fuzz


# ---------------------------------------------------------------------------
# Config

BAR_COLORS = ["#16a34a", "#f59e0b", "#ef4444"]
HIST_COLOR = "#3b82f6"

COLUMN_META = {
    "model": ("text", "Model (JW)"),
    "country": ("id", "Country (ID match)"),
    "year": ("numeric", "Year (rel)"),
    "parameters": ("numeric", "Parameters (rel)"),
    "hardware": ("id", "Hardware (ID match)"),
    "hardware_number": ("numeric", "Hardware number (rel)"),
    "training_compute": ("numeric", "Training compute (rel)"),
    "training_time": ("numeric", "Training time (rel)"),
    "power_draw": ("numeric", "Power draw (rel)"),
    "co2eq": ("numeric", "CO2eq (rel)"),
    # Aliases for inference JSON fields
    "h_number": ("numeric", "Hardware number (rel)"),
}


# ---------------------------------------------------------------------------
# Country ID Lookup (same as benchmark_epoch_id.py)

COUNTRY_ALIASES: dict[str, str] = {
    "usa": "United States of America",
    "u.s.a.": "United States of America",
    "u.s.": "United States of America",
    "us": "United States of America",
    "america": "United States of America",
    "united states": "United States of America",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "prc": "China",
    "people's republic of china": "China",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "uae": "United Arab Emirates",
    "deutschland": "Germany",
    "holland": "Netherlands",
    "russian federation": "Russia",
}


class CountryLookup:
    """Lookup country ID from name using Jaro-Winkler similarity with alias support."""
    
    def __init__(self, db_path: Path):
        self.countries: dict[int, str] = {}
        self.carbon_intensity: dict[int, float | None] = {}
        self._name_to_id: dict[str, int] = {}
        self._cache: dict[str, int | None] = {}
        
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            try:
                for row in cur.execute("SELECT id_country, name, carbon_intensity FROM country").fetchall():
                    cid, name = int(row[0]), str(row[1]) if row[1] else ""
                    carbon = float(row[2]) if row[2] is not None else None
                    self.countries[cid] = name
                    self.carbon_intensity[cid] = carbon
                    if name:
                        self._name_to_id[name.lower().strip()] = cid
            except Exception as e:
                print(f"[benchmark_folder] Warning: Could not load country table: {e}")
            conn.close()
    
    def get_id(self, name: str | None) -> int | None:
        if not name or not self.countries:
            return None
        
        name_clean = str(name).strip()
        if not name_clean or name_clean.lower() in ("nan", "none", "null", "n/a", "na", ""):
            return None
        
        cache_key = name_clean.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if cache_key in COUNTRY_ALIASES:
            canonical = COUNTRY_ALIASES[cache_key]
            if canonical.lower() in self._name_to_id:
                result = self._name_to_id[canonical.lower()]
                self._cache[cache_key] = result
                return result
        
        if cache_key in self._name_to_id:
            result = self._name_to_id[cache_key]
            self._cache[cache_key] = result
            return result
        
        best_id: int | None = None
        best_score = 0.0
        for cid, cname in self.countries.items():
            if not cname:
                continue
            score = JaroWinkler.normalized_similarity(cache_key, cname.lower())
            if score > 1.0:
                score = score / 100.0
            if score > best_score:
                best_score = score
                best_id = cid
        
        if best_score < 0.8:
            best_id = None
        
        self._cache[cache_key] = best_id
        return best_id
    
    def get_name(self, cid: int | None) -> str | None:
        if cid is None:
            return None
        return self.countries.get(cid)


# ---------------------------------------------------------------------------
# Hardware ID Lookup (simplified from benchmark_epoch_id.py)

def _tokenize_hardware(name: str) -> list[str]:
    if not name:
        return []
    s = name.lower().strip().replace("-", " ").replace("_", " ")
    for noise in ["(estimated)", "gpus", "gpu", "graphics card", "graphics", "accelerator"]:
        s = s.replace(noise, " ")
    tokens = [t.strip() for t in s.split() if t.strip()]
    return tokens


class HardwareLookup:
    """Lookup hardware ID from name using token-based Jaro-Winkler similarity."""
    
    def __init__(self, db_path: Path):
        self.hardware: dict[int, str] = {}
        self.hardware_power: dict[int, float | None] = {}
        self._name_to_id: dict[str, int] = {}
        self._tokens: dict[int, list[str]] = {}
        self._cache: dict[str, int | None] = {}
        
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            try:
                for row in cur.execute("SELECT id, name, power FROM hardware").fetchall():
                    hid = int(row[0])
                    name = str(row[1]) if row[1] else ""
                    power = float(row[2]) if row[2] is not None else None
                    
                    self.hardware[hid] = name
                    self.hardware_power[hid] = power
                    self._tokens[hid] = _tokenize_hardware(name)
                    if name:
                        self._name_to_id[name.lower().strip()] = hid
            except Exception as e:
                print(f"[benchmark_folder] Warning: Could not load hardware table: {e}")
            conn.close()
    
    def get_id(self, name: str | None) -> int | None:
        if not name or not self.hardware:
            return None
        
        name_clean = str(name).strip()
        if not name_clean or name_clean.lower() in ("nan", "none", "null", "n/a", "na", "", "gpu"):
            return None
        
        cache_key = name_clean.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if cache_key in self._name_to_id:
            result = self._name_to_id[cache_key]
            self._cache[cache_key] = result
            return result
        
        query_tokens = _tokenize_hardware(name_clean)
        if not query_tokens:
            self._cache[cache_key] = None
            return None
        
        best_id: int | None = None
        best_score = 0.0
        
        for hid, ref_tokens in self._tokens.items():
            if not ref_tokens:
                continue
            
            common = set(query_tokens) & set(ref_tokens)
            union = set(query_tokens) | set(ref_tokens)
            score = len(common) / len(union) if union else 0
            
            if score > best_score:
                best_score = score
                best_id = hid
        
        if best_score < 0.3:
            best_id = None
        
        self._cache[cache_key] = best_id
        return best_id
    
    def get_name(self, hid: int | None) -> str | None:
        if hid is None:
            return None
        return self.hardware.get(hid)
    
    def get_power(self, hid: int | None) -> float | None:
        if hid is None:
            return None
        power_kw = self.hardware_power.get(hid)
        if power_kw is not None:
            return power_kw * 1000.0
        return None


# ---------------------------------------------------------------------------
# Model Matching (simplified from benchmark_epoch_id.py)

def _normalize_model_name(name: str | None) -> str:
    if not name:
        return ""
    s = str(name).lower().strip()
    if s in ('nan', 'none', 'null', 'n/a', ''):
        return ""
    for h in ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2212', '‑', '–', '—']:
        s = s.replace(h, '-')
    for noise in ['(', ')', '[', ']', '"', "'", ',', '.', ':']:
        s = s.replace(noise, ' ')
    s = s.replace('-', ' ').replace('_', ' ')
    return ' '.join(s.split())


def _model_name_similarity(name1: str, name2: str) -> float:
    n1 = _normalize_model_name(name1)
    n2 = _normalize_model_name(name2)
    
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0
    
    return rf_fuzz.token_set_ratio(n1, n2) / 100.0


@dataclass
class ModelMatch:
    base_idx: int
    inf_idx: int
    score: float
    base_model: dict[str, Any]
    inf_model: dict[str, Any]


def _is_valid_inferred_model(model: dict[str, Any]) -> bool:
    model_name = model.get("model", "")
    if not model_name:
        return False
    name_str = str(model_name).strip().lower()
    if name_str in ("nan", "none", "null", "n/a", ""):
        return False
    return True


def match_models_for_paper(
    base_models: list[dict[str, Any]],
    inf_models: list[dict[str, Any]],
    *,
    min_threshold: float = 0.4,
) -> list[ModelMatch]:
    if not base_models:
        return []
    
    valid_inf_models = [(i, m) for i, m in enumerate(inf_models) if _is_valid_inferred_model(m)]
    if not valid_inf_models:
        return []
    
    scores: list[tuple[int, int, float]] = []
    for bi, base in enumerate(base_models):
        base_name = base.get("model", "")
        for orig_ii, inf in valid_inf_models:
            inf_name = inf.get("model", "")
            score = _model_name_similarity(base_name, inf_name)
            if score >= 0.15:
                scores.append((bi, orig_ii, score))
    
    scores.sort(key=lambda x: x[2], reverse=True)
    
    used_base: set[int] = set()
    used_inf: set[int] = set()
    matches: list[ModelMatch] = []
    
    for bi, ii, score in scores:
        if bi in used_base or ii in used_inf:
            continue
        if score < min_threshold:
            continue
        
        matches.append(ModelMatch(
            base_idx=bi,
            inf_idx=ii,
            score=score,
            base_model=base_models[bi],
            inf_model=inf_models[ii],
        ))
        used_base.add(bi)
        used_inf.add(ii)
    
    return matches


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
        val = value[0] if value else None
        if val == "":
            return None
        return val
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


def _is_missing(val: Any) -> bool:
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


# ---------------------------------------------------------------------------
# Data loading

def load_ground_truth(db_path: Path, country_lookup: CountryLookup, hardware_lookup: HardwareLookup) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ground truth from a folder-based DB (created by make_table_from_folder.py).
    Returns (paper_df, model_df).
    """
    conn = sqlite3.connect(str(db_path))
    
    # Load paper info
    try:
        paper_df = pd.read_sql_query("""
            SELECT id_paper, link, abstract, country, id_country, year, split
            FROM paper_info
        """, conn)
    except Exception:
        paper_df = pd.DataFrame(columns=["id_paper", "link", "abstract", "country", "id_country", "year", "split"])
    
    # Load model info
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
    
    # Enrich with IDs if missing
    if "id_country" in paper_df.columns:
        def ensure_country_id(row):
            if pd.notna(row.get("id_country")):
                return int(row["id_country"])
            return country_lookup.get_id(row.get("country"))
        paper_df["id_country"] = paper_df.apply(ensure_country_id, axis=1)
    
    if "id_hardware" in model_df.columns:
        def ensure_hardware_id(row):
            if pd.notna(row.get("id_hardware")):
                return int(row["id_hardware"])
            return hardware_lookup.get_id(row.get("hardware"))
        model_df["id_hardware"] = model_df.apply(ensure_hardware_id, axis=1)
    
    return paper_df, model_df


def load_inference(outputs_path: Path, tracked_fields: list[str], country_lookup: CountryLookup, hardware_lookup: HardwareLookup) -> pd.DataFrame:
    """
    Load inference from JSON files.
    Returns DataFrame with one row per (paper, model).
    """
    records: list[dict[str, Any]] = []
    paths: list[Path]
    
    if outputs_path.is_file():
        paths = [outputs_path]
    else:
        paths = sorted(outputs_path.glob("*.json"))
    
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[benchmark_folder] skip {path}: {exc}")
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
                
                cleaned = {k: _extract_scalar(v) for k, v in model.items()}
                record = {"id_paper": int(pid)}
                
                for field in tracked_fields:
                    if field.startswith("hardware_") and field.replace("hardware_", "h_") in cleaned:
                        record[field] = cleaned.get(field.replace("hardware_", "h_"))
                    else:
                        record[field] = cleaned.get(field)
                
                record["model"] = cleaned.get("model")
                
                if "country" in record and record["country"]:
                    record["id_country"] = country_lookup.get_id(record["country"])
                
                if "hardware" in record and record["hardware"]:
                    record["id_hardware"] = hardware_lookup.get_id(record["hardware"])
                
                records.append(record)
    
    if not records:
        return pd.DataFrame(columns=["id_paper", "model"] + tracked_fields)
    
    return pd.DataFrame(records).sort_values(["id_paper"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics

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
    return [Metric(field=f, kind=COLUMN_META[f][0], display_name=COLUMN_META[f][1]) for f in fields if f in COLUMN_META]


# ---------------------------------------------------------------------------
# Distance computation

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
        
        out_row = {
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
                out_row[f"id_{field}_base"] = base_id
                out_row[f"id_{field}_inf"] = inf_id
                
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
                base_num = _parse_numeric(base_val)
                inf_num = _parse_numeric(inf_val)
                
                out_row[f"{field}_base"] = base_num
                out_row[f"{field}_inf"] = inf_num
                
                if base_num is None:
                    miss[0 if inf_num is None else 2] += 1
                    continue
                if inf_num is None:
                    miss[1] += 1
                    continue
                
                rel = _signed_rel(inf_num, base_num)
                if rel is None:
                    miss[1] += 1
                    continue
                
                distances[field].append(abs(rel))
                out_row[f"{field}_dist"] = abs(rel)
                
            else:  # text
                base_txt = None if _is_missing(base_val) else str(base_val).strip()
                inf_txt = None if _is_missing(inf_val) else str(inf_val).strip()
                
                out_row[f"{field}_base"] = base_txt
                out_row[f"{field}_inf"] = inf_txt
                
                if base_txt is None:
                    miss[0 if inf_txt is None else 2] += 1
                    continue
                if inf_txt is None:
                    miss[1] += 1
                    continue
                
                jw_dist = 1.0 - _jw(base_txt, inf_txt)
                distances[field].append(jw_dist)
                out_row[f"{field}_dist"] = jw_dist
        
        result_rows.append(out_row)
    
    missing = {k: tuple(v) for k, v in missing.items()}
    return distances, missing, result_rows


# ---------------------------------------------------------------------------
# Plotting

def _draw_pair(ax_left, ax_right, distances: list[float], missing: tuple[int, int, int], *, bins, vmin: float, vmax: float, title: str) -> None:
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


def _plot_metrics_grid(out_path: Path, metrics: list[Metric], distances: dict[str, list[float]], missing: dict[str, tuple[int, int, int]]) -> None:
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
        bins = metric.bins
        vmin, vmax = metric.range_limits
        _draw_pair(ax_left, ax_right, vals, miss, bins=bins, vmin=vmin, vmax=vmax, title=metric.display_name)

    legend_handles = [
        plt.matplotlib.patches.Patch(facecolor=color, edgecolor="black", label=label)
        for color, label in zip(BAR_COLORS, ["both null", "infer null", "base null"], strict=True)
    ]
    legend_handles.append(plt.matplotlib.patches.Patch(facecolor=HIST_COLOR, edgecolor="black", label="distance"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.03))
    fig.subplots_adjust(bottom=0.20, top=0.90)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_distance_summary(distances: dict[str, list[float]], out_path: Path) -> float:
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


def _create_percentage_table(distances: dict[str, list[float]], out_path: Path) -> None:
    import matplotlib.patches as mpatches
    
    table_data = []
    for field, vals in distances.items():
        if vals:
            abs_vals = [abs(v) for v in vals]
            mean_dist = float(np.mean(abs_vals))
            std_dist = float(np.std(abs_vals))
            accuracy_pct = (1.0 - mean_dist) * 100.0
            count = len(vals)
            table_data.append({
                "field": field,
                "accuracy_pct": accuracy_pct,
                "distance_pct": mean_dist * 100.0,
                "std_pct": std_dist * 100.0,
                "count": count
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
            f"{acc:.1f}%",
            f"{row['distance_pct']:.1f}%",
            f"±{row['std_pct']:.1f}%",
            str(row["count"])
        ])
        cell_colors.append([color] * 5)
    
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e9ecef"] * 5,
        cellLoc='center',
        loc='center',
        colWidths=[0.30, 0.18, 0.18, 0.15, 0.10]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for j in range(5):
        cell = table[(0, j)]
        cell.set_text_props(weight='bold')
    
    ax.set_title("Metrics Summary (non-null columns)", fontsize=14, fontweight='bold', pad=20)
    
    legend_elements = [
        mpatches.Patch(facecolor="#d4edda", edgecolor='black', label='≥90%'),
        mpatches.Patch(facecolor="#fff3cd", edgecolor='black', label='70-90%'),
        mpatches.Patch(facecolor="#ffeeba", edgecolor='black', label='50-70%'),
        mpatches.Patch(facecolor="#f8d7da", edgecolor='black', label='<50%'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.15), ncol=4, fontsize=9)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI

def run_benchmark(
    db_path: Path,
    outputs_path: Path,
    out_dir: Path,
    tracked_fields: list[str],
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup,
) -> dict[str, Any] | None:
    """Run benchmark for folder-based dataset."""
    
    paper_df, model_df = load_ground_truth(db_path, country_lookup, hardware_lookup)
    
    if len(paper_df) == 0 and len(model_df) == 0:
        print(f"[benchmark_folder] No ground truth data found; skipping.")
        return None
    
    print(f"[benchmark_folder] Loaded {len(paper_df)} papers, {len(model_df)} models")
    
    infer_df = load_inference(outputs_path, tracked_fields, country_lookup, hardware_lookup)
    
    if len(infer_df) == 0:
        print(f"[benchmark_folder] No inference data found; skipping.")
        return None
    
    # Match models
    all_matches: list[ModelMatch] = []
    matched_papers = 0
    
    inferred_paper_ids = set(infer_df["id_paper"].unique())
    
    for pid in inferred_paper_ids:
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
        print(f"[benchmark_folder] No matches found; skipping.")
        return None
    
    metrics = build_metrics(tracked_fields)
    if not metrics:
        return None
    
    distances, missing, result_rows = compute_distances(all_matches, paper_df, metrics)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(out_dir / "matched_models.csv", index=False)
    
    _plot_metrics_grid(out_dir / "benchmark.png", metrics, distances, missing)
    
    global_score = save_distance_summary(distances, out_dir / "distance_summary.csv")
    print(f"[benchmark_folder] Global distance: {global_score:.4f}")
    
    _create_percentage_table(distances, out_dir / "percentage_table.png")
    
    return {
        "papers": len(paper_df),
        "models": len(model_df),
        "matched_pairs": len(all_matches),
        "matched_papers": matched_papers,
        "global_distance": global_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs folder-based dataset.")
    parser.add_argument("--database", required=True, help="Path to database (created by make_table_from_folder.py)")
    parser.add_argument("--outputs", required=True, help="JSON file or directory with inference results")
    parser.add_argument("--out-dir", default="results_folder/benchmark", help="Output directory")
    parser.add_argument(
        "--columns",
        nargs="*",
        choices=sorted(COLUMN_META.keys()),
        help="Columns to benchmark (default: all)",
    )
    args = parser.parse_args()

    db_path = Path(args.database)
    outputs_path = Path(args.outputs)
    out_dir = Path(args.out_dir)

    country_lookup = CountryLookup(db_path)
    hardware_lookup = HardwareLookup(db_path)
    print(f"[benchmark_folder] Loaded {len(country_lookup.countries)} countries, {len(hardware_lookup.hardware)} hardware")

    all_fields = ["model", "parameters", "hardware", "hardware_number", 
                  "training_time", "training_compute", "power_draw", "co2eq",
                  "country", "year"]
    if args.columns:
        tracked_fields = [f for f in args.columns if f in all_fields]
    else:
        tracked_fields = all_fields

    result = run_benchmark(
        db_path=db_path,
        outputs_path=outputs_path,
        out_dir=out_dir,
        tracked_fields=tracked_fields,
        country_lookup=country_lookup,
        hardware_lookup=hardware_lookup,
    )
    
    if result:
        print(f"\n{'='*60}")
        print("[benchmark_folder] SUMMARY")
        print(f"{'='*60}")
        print(f"Papers: {result['papers']}")
        print(f"Models: {result['models']}")
        print(f"Matched pairs: {result['matched_pairs']}")
        print(f"Global distance: {result['global_distance']:.4f}")


if __name__ == "__main__":
    main()
