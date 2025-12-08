#!/usr/bin/env python3
"""
Benchmark inferred results against the GreenMIR dataset with ID-based country and hardware matching.

- Reads ground truth from a CSV (e.g., static/dataset.csv).
- Reads inference outputs from paper-*.json (structure compatible with src/infer.py).
- For country field: maps text to ID using the country table in greenmir.db via Jaro-Winkler.
- For hardware field: maps text to ID using the hardware table in greenmir.db via Jaro-Winkler.
- Compares id_country_true == id_country_pred and id_hardware_true == id_hardware_pred (exact match).
- For each paper and each column, keeps the model candidate with the smallest distance.
- Plots one pair of charts per column (missingness + histogram of distances).
- Writes a CSV with mean absolute distance-to-zero per column and a global weighted average.
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
    "country": ("id", "Country (ID match)"),  # ID-based
    "year": ("numeric", "Year (rel)"),
    "parameters": ("numeric", "Parameters (rel)"),
    "hardware": ("id", "Hardware (ID match)"),  # ID-based
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
# Country ID Lookup

# Country aliases: common abbreviations/variants -> canonical name
COUNTRY_ALIASES: dict[str, str] = {
    # United States
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "america": "United States",
    "united states of america": "United States",
    # United Kingdom
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    # China
    "prc": "China",
    "people's republic of china": "China",
    "p.r.c.": "China",
    # South Korea
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "rok": "South Korea",
    # UAE
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    # Germany
    "deutschland": "Germany",
    # Netherlands
    "holland": "Netherlands",
    # Russia
    "russian federation": "Russia",
}


class CountryLookup:
    """Lookup country ID from name using Jaro-Winkler similarity with alias support."""
    
    def __init__(self, db_path: Path):
        self.countries: dict[int, str] = {}
        self._name_to_id: dict[str, int] = {}
        self._cache: dict[str, int | None] = {}
        
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            try:
                for row in cur.execute("SELECT id_country, name FROM country").fetchall():
                    cid, name = int(row[0]), str(row[1]) if row[1] else ""
                    self.countries[cid] = name
                    # Also store normalized name for exact match
                    if name:
                        self._name_to_id[name.lower().strip()] = cid
            except Exception as e:
                print(f"[benchmark_id] Warning: Could not load country table: {e}")
            conn.close()
    
    def get_id(self, name: str | None) -> int | None:
        """Find the country ID that best matches the given name using Jaro-Winkler."""
        if not name or not self.countries:
            return None
        
        name_clean = str(name).strip()
        if not name_clean:
            return None
        
        # Handle invalid values like "nan"
        if name_clean.lower() in ("nan", "none", "null", "n/a", "na", ""):
            return None
        
        # Check cache
        cache_key = name_clean.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check aliases first (exact match on alias)
        if cache_key in COUNTRY_ALIASES:
            canonical = COUNTRY_ALIASES[cache_key]
            if canonical.lower() in self._name_to_id:
                result = self._name_to_id[canonical.lower()]
                self._cache[cache_key] = result
                return result
        
        # Try exact match on database name
        if cache_key in self._name_to_id:
            result = self._name_to_id[cache_key]
            self._cache[cache_key] = result
            return result
        
        # Find best Jaro-Winkler match
        best_id: int | None = None
        best_score = 0.0
        
        for cid, cname in self.countries.items():
            if not cname:
                continue
            score = JaroWinkler.normalized_similarity(name_clean.lower(), cname.lower())
            if score > 1.0:
                score = score / 100.0
            if score > best_score:
                best_score = score
                best_id = cid
        
        # Only accept if similarity is reasonable (> 0.8 to avoid false matches)
        if best_score < 0.8:
            best_id = None
        
        self._cache[cache_key] = best_id
        return best_id
    
    def get_name(self, cid: int | None) -> str | None:
        """Get country name from ID."""
        if cid is None:
            return None
        return self.countries.get(cid)


# ---------------------------------------------------------------------------
# Hardware ID Lookup

def _tokenize_hardware(name: str) -> list[str]:
    """Tokenize hardware name, normalizing common patterns."""
    import re
    if not name:
        return []
    # Normalize: lowercase, replace hyphens/underscores with spaces
    s = name.lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    # Remove common suffixes/noise
    for noise in ["(estimated)", "(estimated", "estimated)", "gpu", "gpus", "graphics card", "graphics"]:
        s = s.replace(noise, " ")
    # Split and filter empty tokens
    tokens = [t.strip() for t in s.split() if t.strip()]
    
    # Split concatenated model numbers like "2080ti" -> ["2080", "ti"], "1080ti" -> ["1080", "ti"]
    expanded = []
    for t in tokens:
        match = re.match(r'^(\d{4})(ti|super|xt)$', t)
        if match:
            expanded.append(match.group(1))
            expanded.append(match.group(2))
        else:
            expanded.append(t)
    
    return expanded


def _extract_model_numbers(tokens: list[str]) -> set[str]:
    """Extract model numbers like '1080', '2080', 'a100', 'v100', 'ti', etc."""
    import re
    model_nums = set()
    for t in tokens:
        # Look for patterns like 1080, 2080, 3090, a100, v100, h100, etc.
        if re.match(r'^[a-z]?\d{2,4}[a-z]?$', t):
            model_nums.add(t)
        # Also capture "ti", "super", "xt", etc.
        if t in ("ti", "super", "xt", "sxm", "sxm2", "pcie"):
            model_nums.add(t)
    return model_nums


def _token_match_score(query_tokens: list[str], ref_tokens: list[str]) -> float:
    """
    Calculate a token-based matching score that is order-invariant.
    For each query token, find the best matching reference token.
    Returns average of best matches (0-1 scale).
    Critical model numbers must match exactly.
    """
    if not query_tokens or not ref_tokens:
        return 0.0
    
    # Extract model numbers - these must match exactly
    query_models = _extract_model_numbers(query_tokens)
    ref_models = _extract_model_numbers(ref_tokens)
    
    # If query has specific model numbers, they must be present in reference
    if query_models:
        # Check if at least one key model number matches
        if not query_models.intersection(ref_models):
            # No matching model numbers - heavily penalize
            return 0.1
    
    total_score = 0.0
    matched_ref = set()
    
    for qt in query_tokens:
        best_score = 0.0
        best_idx = -1
        for i, rt in enumerate(ref_tokens):
            if i in matched_ref:
                continue
            # Exact match gets 1.0
            if qt == rt:
                score = 1.0
            else:
                # For model numbers, require exact match or very high similarity
                if qt in query_models or rt in ref_models:
                    # Model number - must be exact or very close
                    if qt == rt:
                        score = 1.0
                    elif qt in rt or rt in qt:
                        score = 0.9  # One contains the other (e.g., "1080ti" contains "1080")
                    else:
                        score = JaroWinkler.normalized_similarity(qt, rt)
                        if score > 1.0:
                            score = score / 100.0
                        # Penalize non-exact model number matches
                        if score < 0.95:
                            score *= 0.5
                else:
                    # Regular token - use JW similarity
                    score = JaroWinkler.normalized_similarity(qt, rt)
                    if score > 1.0:
                        score = score / 100.0
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0 and best_score > 0.7:
            matched_ref.add(best_idx)
        total_score += best_score
    
    # Average score
    avg_score = total_score / len(query_tokens)
    
    # Bonus for matching more reference tokens (coverage)
    coverage = len(matched_ref) / len(ref_tokens) if ref_tokens else 0
    
    return 0.7 * avg_score + 0.3 * coverage


class HardwareLookup:
    """Lookup hardware ID from name using token-based Jaro-Winkler similarity, also provides power info."""
    
    def __init__(self, db_path: Path):
        self.hardware: dict[int, str] = {}
        self.hardware_power: dict[int, float | None] = {}  # power in kW
        self.hardware_compute: dict[int, float | None] = {}  # compute in FLOP/s
        self._name_to_id: dict[str, int] = {}
        self._tokens: dict[int, list[str]] = {}  # Pre-tokenized hardware names
        self._cache: dict[str, int | None] = {}
        
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            try:
                for row in cur.execute("SELECT id, name, compute, power FROM hardware").fetchall():
                    hid = int(row[0])
                    name = str(row[1]) if row[1] else ""
                    compute = float(row[2]) if row[2] is not None else None
                    power = float(row[3]) if row[3] is not None else None
                    
                    self.hardware[hid] = name
                    self.hardware_power[hid] = power
                    self.hardware_compute[hid] = compute
                    self._tokens[hid] = _tokenize_hardware(name)
                    # Also store normalized name for exact match
                    if name:
                        self._name_to_id[name.lower().strip()] = hid
            except Exception as e:
                print(f"[benchmark_id] Warning: Could not load hardware table: {e}")
            conn.close()
    
    def get_id(self, name: str | None) -> int | None:
        """Find the hardware ID that best matches the given name using token-based matching."""
        if not name or not self.hardware:
            return None
        
        name_clean = str(name).strip()
        if not name_clean:
            return None
        
        # Handle invalid values
        if name_clean.lower() in ("nan", "none", "null", "n/a", "na", "", "gpu", "gpus"):
            return None
        
        # Check cache
        cache_key = name_clean.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try exact match first
        if cache_key in self._name_to_id:
            result = self._name_to_id[cache_key]
            self._cache[cache_key] = result
            return result
        
        # Tokenize query
        query_tokens = _tokenize_hardware(name_clean)
        if not query_tokens:
            self._cache[cache_key] = None
            return None
        
        # Find best token-based match
        best_id: int | None = None
        best_score = 0.0
        
        for hid, ref_tokens in self._tokens.items():
            if not ref_tokens:
                continue
            
            # Calculate token match score
            score = _token_match_score(query_tokens, ref_tokens)
            
            # Also calculate reverse score (ref -> query) to catch asymmetric matches
            reverse_score = _token_match_score(ref_tokens, query_tokens)
            
            # Use the average of both directions
            final_score = (score + reverse_score) / 2
            
            if final_score > best_score:
                best_score = final_score
                best_id = hid
        
        # Only accept if similarity is reasonable (> 0.6)
        if best_score < 0.6:
            best_id = None
        
        self._cache[cache_key] = best_id
        return best_id
    
    def get_name(self, hid: int | None) -> str | None:
        """Get hardware name from ID."""
        if hid is None:
            return None
        return self.hardware.get(hid)
    
    def get_power(self, hid: int | None) -> float | None:
        """Get hardware power in Watts from ID (database stores kW, we convert to W)."""
        if hid is None:
            return None
        power_kw = self.hardware_power.get(hid)
        if power_kw is not None:
            return power_kw * 1000.0  # Convert kW to W
        return None
    
    def get_compute(self, hid: int | None) -> float | None:
        """Get hardware compute (in FLOP/s) from ID."""
        if hid is None:
            return None
        return self.hardware_compute.get(hid)


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

def load_dataset(dataset_path: Path, country_lookup: CountryLookup, hardware_lookup: HardwareLookup) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df = df.rename(columns=DATASET_RENAME)
    df.insert(0, "id_paper", df.index + 1)
    
    # Convert country names to IDs
    if "country" in df.columns:
        df["id_country"] = df["country"].apply(country_lookup.get_id)
    
    # Convert hardware names to IDs
    if "hardware" in df.columns:
        df["id_hardware"] = df["hardware"].apply(hardware_lookup.get_id)
    
    keep_cols = ["id_paper"] + [col for col in COLUMN_META if col in df.columns]
    if "id_country" in df.columns:
        keep_cols.append("id_country")
    if "id_hardware" in df.columns:
        keep_cols.append("id_hardware")
    return df[keep_cols]


def load_inference(outputs_path: Path, tracked_fields: list[str], country_lookup: CountryLookup, hardware_lookup: HardwareLookup) -> pd.DataFrame:
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
            print(f"[benchmark_id] skip {path}: {exc}")
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
                    # Convert country to ID
                    if "country" in record and record["country"]:
                        record["id_country"] = country_lookup.get_id(record["country"])
                    # Convert hardware to ID and infer hardware_power if missing
                    if "hardware" in record and record["hardware"]:
                        record["id_hardware"] = hardware_lookup.get_id(record["hardware"])
                        # Infer hardware_power from hardware ID if not already set
                        existing_hw_power = record.get("hardware_power")
                        if (existing_hw_power is None or existing_hw_power == "") and record["id_hardware"] is not None:
                            hw_power = hardware_lookup.get_power(record["id_hardware"])
                            if hw_power is not None:
                                record["hardware_power"] = hw_power
                    # Calculate power_draw = training_time * hardware_number * hardware_power / 1000 (kWh)
                    # Formula: power_draw (kWh) = training_time (h) * hardware_number * hardware_power (W) / 1000
                    existing_power_draw = record.get("power_draw")
                    if existing_power_draw is None or existing_power_draw == "":
                        training_time = record.get("training_time")
                        h_number = record.get("hardware_number")
                        hw_pwr = record.get("hardware_power")
                        if training_time is not None and training_time != "" and h_number is not None and h_number != "" and hw_pwr is not None and hw_pwr != "":
                            try:
                                record["power_draw"] = float(training_time) * float(h_number) * float(hw_pwr) / 1000.0
                            except (ValueError, TypeError):
                                pass
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
                # Convert country to ID
                if "country" in record and record["country"]:
                    record["id_country"] = country_lookup.get_id(record["country"])
                # Convert hardware to ID and infer hardware_power if missing
                if "hardware" in record and record["hardware"]:
                    record["id_hardware"] = hardware_lookup.get_id(record["hardware"])
                    # Infer hardware_power from hardware ID if not already set
                    existing_hw_power = record.get("hardware_power")
                    if (existing_hw_power is None or existing_hw_power == "") and record["id_hardware"] is not None:
                        hw_power = hardware_lookup.get_power(record["id_hardware"])
                        if hw_power is not None:
                            record["hardware_power"] = hw_power
                # Calculate power_draw = training_time * hardware_number * hardware_power / 1000 (kWh)
                # Formula: power_draw (kWh) = training_time (h) * hardware_number * hardware_power (W) / 1000
                existing_power_draw = record.get("power_draw")
                if existing_power_draw is None or existing_power_draw == "":
                    training_time = record.get("training_time")
                    h_number = record.get("hardware_number")
                    hw_pwr = record.get("hardware_power")
                    if training_time is not None and training_time != "" and h_number is not None and h_number != "" and hw_pwr is not None and hw_pwr != "":
                        try:
                            record["power_draw"] = float(training_time) * float(h_number) * float(hw_pwr) / 1000.0
                        except (ValueError, TypeError):
                            pass
                records.append(record)
    if not records:
        return pd.DataFrame(columns=["id_paper"] + tracked_fields + ["id_country", "id_hardware"])
    df = pd.DataFrame(records)
    return df.sort_values(["id_paper"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics + plotting

@dataclass
class Metric:
    field: str
    kind: str  # "text" | "numeric" | "id"
    display_name: str

    @property
    def bins(self) -> Iterable[float]:
        if self.kind == "text":
            return np.linspace(0.0, 1.0, 26)
        if self.kind == "id":
            return [0, 1]  # For ID: 0 = match, 1 = no match
        return np.linspace(-1.0, 1.0, 41)

    @property
    def range_limits(self) -> tuple[float, float]:
        if self.kind == "id":
            return (0.0, 1.0)
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
    ax_right.set_xlabel("distance (0=match, 1=mismatch)" if "ID" in title else "distance", fontsize=8)


def _plot_metrics_grid(out_path: Path, metrics: list[Metric], distances: dict[str, list[float]], missing: dict[str, tuple[int, int, int]]) -> None:
    if not metrics:
        print("[benchmark_id] no overlapping columns; skip plot")
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
    
    For country (kind="id"): uses id_country comparison (0 if match, 1 if mismatch).
    For hardware (kind="id"): uses id_hardware comparison (0 if match, 1 if mismatch).
    """
    distances: dict[str, list[float]] = {m.field: [] for m in metrics}
    missing: dict[str, list[int]] = {m.field: [0, 0, 0] for m in metrics}
    merged_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        out_row: dict[str, Any] = {"id_paper": row["id_paper"]}
        for metric in metrics:
            field = metric.field
            miss = missing[field]

            # Special handling for ID-based fields (country, hardware)
            if metric.kind == "id":
                # Determine which ID column to use
                if field == "country":
                    id_col = "id_country"
                elif field == "hardware":
                    id_col = "id_hardware"
                else:
                    continue  # Unknown ID field
                
                base_id = row.get(f"{id_col}_base")
                if base_id is None or (isinstance(base_id, float) and math.isnan(base_id)):
                    base_id = None
                else:
                    base_id = int(base_id)
                
                # Get all model candidates
                models = infer_models[infer_models["id_paper"] == row["id_paper"]]
                if models.empty:
                    inf_ids: list[Any] = [row.get(f"{id_col}_inf")]
                else:
                    inf_ids = models.get(id_col, pd.Series(dtype=object)).tolist()
                
                if not inf_ids:
                    inf_ids = [None]
                
                # Clean up inference IDs
                clean_inf_ids = []
                for iid in inf_ids:
                    if iid is None or (isinstance(iid, float) and math.isnan(iid)):
                        clean_inf_ids.append(None)
                    else:
                        clean_inf_ids.append(int(iid))
                
                if base_id is None:
                    all_missing = all(iid is None for iid in clean_inf_ids)
                    if all_missing:
                        miss[0] += 1
                    else:
                        miss[2] += 1
                    continue
                
                # Find best match (prefer exact ID match)
                has_match = any(iid == base_id for iid in clean_inf_ids if iid is not None)
                has_any_id = any(iid is not None for iid in clean_inf_ids)
                
                if not has_any_id:
                    miss[1] += 1
                else:
                    # Distance: 0 if match, 1 if mismatch
                    dist = 0.0 if has_match else 1.0
                    distances[field].append(dist)
                    out_row[f"{field}_dist"] = dist
                continue

            # Standard handling for other fields
            base_val = row.get(f"{field}_base")

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

                # Special handling for training_time: try BOTH sum and best match, keep best
                if field == "training_time":
                    # Collect all valid training times
                    valid_times = []
                    for candidate in inf_candidates:
                        inf_num = _parse_numeric(candidate)
                        if inf_num is not None and inf_num > 0:
                            valid_times.append(inf_num)
                    
                    if not valid_times:
                        miss[1] += 1
                        continue
                    
                    # Option 1: Sum all times
                    total_time = sum(valid_times)
                    rel_sum = _signed_rel(total_time, base_num)
                    
                    # Option 2: Best individual match
                    best_individual = min(valid_times, key=lambda x: abs(_signed_rel(x, base_num) or float('inf')))
                    rel_best = _signed_rel(best_individual, base_num)
                    
                    # Choose the one with smaller distance
                    if rel_sum is not None and rel_best is not None:
                        if abs(rel_sum) < abs(rel_best):
                            chosen_rel = rel_sum
                            out_row[f"{field}_method"] = "sum"
                        else:
                            chosen_rel = rel_best
                            out_row[f"{field}_method"] = "best"
                    elif rel_sum is not None:
                        chosen_rel = rel_sum
                        out_row[f"{field}_method"] = "sum"
                    elif rel_best is not None:
                        chosen_rel = rel_best
                        out_row[f"{field}_method"] = "best"
                    else:
                        miss[1] += 1
                        continue
                    
                    distances[field].append(float(abs(chosen_rel)))
                    out_row[f"{field}_dist"] = float(abs(chosen_rel))
                    continue

                # Standard handling: pick best candidate (minimizes distance)
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
                    out_row[f"{field}_dist"] = float(to_store)
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
                    out_row[f"{field}_dist"] = float(best_jw)

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
    Saves mean absolute distance per column with std and returns the global weighted average
    (sum(|d|)/count over all metrics).
    """
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
    """
    Creates a PNG table showing accuracy percentages (100 - distance%) for each non-null column.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Collect data for non-null columns
    table_data = []
    for field, vals in distances.items():
        if vals:  # Only include non-null columns
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
    
    # Sort by accuracy (descending)
    table_data.sort(key=lambda x: x["accuracy_pct"], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(3, len(table_data) * 0.5 + 1.5)))
    ax.axis('off')
    
    # Table headers
    col_labels = ["Champ", "Précision (%)", "Distance (%)", "Std (%)", "N"]
    
    # Table cell data
    cell_text = []
    cell_colors = []
    
    for row in table_data:
        acc = row["accuracy_pct"]
        dist = row["distance_pct"]
        std = row["std_pct"]
        
        # Color based on accuracy
        if acc >= 90:
            color = "#d4edda"  # Green
        elif acc >= 70:
            color = "#fff3cd"  # Yellow
        elif acc >= 50:
            color = "#ffeeba"  # Light orange
        else:
            color = "#f8d7da"  # Red
        
        cell_text.append([
            row["field"],
            f"{acc:.1f}%",
            f"{dist:.1f}%",
            f"±{std:.1f}%",
            str(row["count"])
        ])
        cell_colors.append([color, color, color, color, color])
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e9ecef"] * 5,
        cellLoc='center',
        loc='center',
        colWidths=[0.30, 0.18, 0.18, 0.15, 0.10]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Bold headers
    for j in range(5):
        cell = table[(0, j)]
        cell.set_text_props(weight='bold')
    
    # Title
    ax.set_title("Résumé des métriques (colonnes non-nulles)", fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#d4edda", edgecolor='black', label='≥90%'),
        mpatches.Patch(facecolor="#fff3cd", edgecolor='black', label='70-90%'),
        mpatches.Patch(facecolor="#ffeeba", edgecolor='black', label='50-70%'),
        mpatches.Patch(facecolor="#f8d7da", edgecolor='black', label='<50%'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.15), 
              ncol=4, fontsize=9, title="Précision")
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_papers_per_year(infer_df: pd.DataFrame, out_path: Path, metrics: list[Metric]) -> None:
    """
    Plot 1: Number of papers per year that have non-null values for each column.
    Uses only inferred data (from JSON).
    """
    import matplotlib.pyplot as plt
    
    # Get year column from inference
    if "year" not in infer_df.columns:
        return
    
    # Filter to valid years
    df = infer_df.copy()
    df["year_parsed"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year_parsed"].notna()]
    if df.empty:
        return
    
    df["year_int"] = df["year_parsed"].astype(int)
    years = sorted(df["year_int"].unique())
    
    if len(years) < 2:
        return
    
    # Count non-null values per year for each column (from inference)
    field_counts: dict[str, dict[int, int]] = {}
    
    for metric in metrics:
        field = metric.field
        
        # For ID-based fields, use id_* columns
        if metric.kind == "id":
            if field == "country":
                check_col = "id_country"
            elif field == "hardware":
                check_col = "id_hardware"
            else:
                check_col = field
        else:
            check_col = field
        
        if check_col not in df.columns:
            continue
        
        counts_by_year: dict[int, int] = {}
        for year in years:
            year_df = df[df["year_int"] == year]
            # Count non-null and non-empty values
            non_null = 0
            for val in year_df[check_col]:
                if val is not None and val != "" and not (isinstance(val, float) and np.isnan(val)):
                    non_null += 1
            counts_by_year[year] = non_null
        
        if sum(counts_by_year.values()) > 0:
            field_counts[field] = counts_by_year
    
    if not field_counts:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10.colors
    x = np.array(years)
    
    for i, (field, counts) in enumerate(field_counts.items()):
        y = [counts.get(yr, 0) for yr in years]
        ax.plot(x, y, marker='o', label=field, color=colors[i % len(colors)], linewidth=2, markersize=5)
    
    ax.set_xlabel("Année (inférée)", fontsize=12)
    ax.set_ylabel("Nombre de papiers", fontsize=12)
    ax.set_title("Évolution du nombre de papiers par année (données inférées)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_distance_by_year(infer_df: pd.DataFrame, merged_rows: list[dict[str, Any]], 
                            metrics: list[Metric], out_path: Path) -> None:
    """
    Plot 2: Mean distance per column by year with std as error bars.
    Uses inferred year from JSON.
    """
    import matplotlib.pyplot as plt
    
    # Get year from infer_df (each row has distances computed)
    if not merged_rows:
        return
    
    # Build year lookup from inference data
    year_by_paper: dict[int, int] = {}
    for _, row in infer_df.iterrows():
        pid = row.get("id_paper")
        year_val = row.get("year")
        if pid is None or year_val is None:
            continue
        try:
            year_int = int(float(year_val))
            year_by_paper[int(pid)] = year_int
        except (ValueError, TypeError):
            continue
    
    # Build a dataframe from merged_rows with year info from inference
    rows_with_year = []
    for row in merged_rows:
        pid = row.get("id_paper")
        if pid is None:
            continue
        
        # Get year from inference data
        year = year_by_paper.get(int(pid))
        if year is None:
            continue
        
        row_copy = dict(row)
        row_copy["year"] = year
        rows_with_year.append(row_copy)
    
    if not rows_with_year:
        return
    
    df = pd.DataFrame(rows_with_year)
    years = sorted(df["year"].unique())
    
    if len(years) < 2:
        return
    
    # Compute mean and std of distances per year for each column
    field_stats: dict[str, dict[int, tuple[float, float]]] = {}  # field -> year -> (mean, std)
    
    for metric in metrics:
        field = metric.field
        dist_col = f"{field}_dist"
        
        if dist_col not in df.columns:
            continue
        
        stats_by_year: dict[int, tuple[float, float]] = {}
        for year in years:
            year_df = df[df["year"] == year]
            vals = year_df[dist_col].dropna()
            if len(vals) > 0:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                stats_by_year[year] = (mean_val, std_val)
        
        if stats_by_year:
            field_stats[field] = stats_by_year
    
    if not field_stats:
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10.colors
    x = np.array(years)
    
    for i, (field, stats) in enumerate(field_stats.items()):
        y_mean = []
        y_std = []
        x_valid = []
        
        for year in years:
            if year in stats:
                x_valid.append(year)
                y_mean.append(stats[year][0])
                y_std.append(stats[year][1])
        
        if not x_valid:
            continue
        
        x_arr = np.array(x_valid)
        y_mean_arr = np.array(y_mean)
        y_std_arr = np.array(y_std)
        
        color = colors[i % len(colors)]
        ax.plot(x_arr, y_mean_arr, marker='o', label=field, color=color, linewidth=2, markersize=5)
        ax.fill_between(x_arr, y_mean_arr - y_std_arr, y_mean_arr + y_std_arr, 
                        color=color, alpha=0.2)
    
    ax.set_xlabel("Année (inférée)", fontsize=12)
    ax.set_ylabel("Distance moyenne", fontsize=12)
    ax.set_title("Évolution de la distance moyenne par année (±std, données inférées)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs GreenMIR dataset with ID-based country and hardware matching.")
    parser.add_argument("--dataset", required=True, help="CSV with ground-truth values (e.g., static/dataset.csv)")
    parser.add_argument("--outputs-dir", required=True, help="Directory containing paper-XXXX.json files or a single aggregated JSON file")
    parser.add_argument("--out-dir", default="results_id/benchmark_greenmir", help="Where to store merged data and plots")
    parser.add_argument("--db", default="data/greenmir.db", help="Path to greenmir.db containing country and hardware tables")
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
    db_path = Path(args.db)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load lookup tables
    country_lookup = CountryLookup(db_path)
    hardware_lookup = HardwareLookup(db_path)
    print(f"[benchmark_id] Loaded {len(country_lookup.countries)} countries and {len(hardware_lookup.hardware)} hardware from {db_path}")

    base_df = load_dataset(dataset_path, country_lookup, hardware_lookup)
    requested_columns = args.columns if args.columns else list(COLUMN_META.keys())
    tracked_fields = [field for field in requested_columns if field in base_df.columns]
    if not tracked_fields:
        raise SystemExit("[benchmark_id] None of the requested columns exist in dataset.")
    infer_df = load_inference(outputs_dir, tracked_fields, country_lookup, hardware_lookup)

    combined_json_path = out_dir / "combined_inference.json"
    # Save collapsed (one row per paper) for convenience; plotting still uses per-model candidates.
    collapse_for_json = infer_df.groupby("id_paper").tail(1).reset_index(drop=True)
    save_combined_json(collapse_for_json, combined_json_path)
    print(f"[benchmark_id] combined inference written to {combined_json_path}")

    if not args.all_papers:
        inferred_ids = set(infer_df["id_paper"].tolist())
        base_df = base_df[base_df["id_paper"].isin(inferred_ids)]

    if base_df.empty:
        print("[benchmark_id] No overlapping papers between dataset and inference; nothing to plot.")
        return

    # Merge like v4/benchmark.py for proper column alignment
    merged = base_df.merge(infer_df, on="id_paper", how="left", suffixes=("_base", "_inf"))
    # Keep only one row per paper to avoid counting duplicates
    merged = merged.drop_duplicates(subset=["id_paper"], keep="first")

    # Only include fields that exist in both base and infer
    # For ID-based fields (country, hardware), check id_* columns instead
    common_fields = []
    for field in tracked_fields:
        if field == "country":
            if "id_country_base" in merged.columns or "id_country_inf" in merged.columns:
                common_fields.append(field)
        elif field == "hardware":
            if "id_hardware_base" in merged.columns or "id_hardware_inf" in merged.columns:
                common_fields.append(field)
        elif f"{field}_inf" in merged.columns:
            common_fields.append(field)
    
    metrics = build_metrics(common_fields)
    if not metrics:
        print("[benchmark_id] no common fields between dataset and inference; aborting plot.")
        return

    distances, missing, merged_rows = compute_distances(merged, metrics, infer_df)

    # Reorder columns: id_paper, then grouped fields with their IDs
    # For country and hardware, group: field_base, field_inf, id_field_base, id_field_inf
    ordered_cols = ["id_paper"]
    
    # Define field order with special handling for country/hardware (grouped with IDs)
    field_order = [
        "model", "abstract", "country", "year", "parameters", "hardware",
        "hardware_compute", "hardware_number", "hardware_power",
        "training_compute", "training_time", "power_draw", "co2eq"
    ]
    
    for field in field_order:
        base_col = f"{field}_base"
        inf_col = f"{field}_inf"
        
        # Add base and inf columns
        if base_col in merged.columns:
            ordered_cols.append(base_col)
        if inf_col in merged.columns:
            ordered_cols.append(inf_col)
        
        # For country and hardware, also add their ID columns right after
        if field == "country":
            if "id_country_base" in merged.columns:
                ordered_cols.append("id_country_base")
            if "id_country_inf" in merged.columns:
                ordered_cols.append("id_country_inf")
        elif field == "hardware":
            if "id_hardware_base" in merged.columns:
                ordered_cols.append("id_hardware_base")
            if "id_hardware_inf" in merged.columns:
                ordered_cols.append("id_hardware_inf")
    
    # Add any remaining columns not yet included
    for col in merged.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    merged_ordered = merged[ordered_cols]
    
    merged_path = out_dir / "merged.csv"
    merged_ordered.to_csv(merged_path, index=False)
    print(f"[benchmark_id] merged comparison CSV written to {merged_path}")
    plot_path = out_dir / "benchmark.png"
    _plot_metrics_grid(plot_path, metrics, distances, missing)
    print(f"[benchmark_id] figure saved to {plot_path}")

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
    print(f"[benchmark_id] gradient stacked plots saved to {overview_dir}")

    summary_path = out_dir / "distance_summary.csv"
    global_score = save_distance_summary(distances, summary_path)
    print(f"[benchmark_id] distance summary saved to {summary_path}")
    print(f"[benchmark_id] global weighted average distance-to-zero: {global_score}")
    
    # Create percentage table PNG
    table_path = out_dir / "percentage_table.png"
    _create_percentage_table(distances, table_path)
    print(f"[benchmark_id] percentage table saved to {table_path}")
    
    # Plot 1: Papers per year with non-null values for each column (from inference)
    papers_year_path = overview_dir / "papers_per_year.png"
    _plot_papers_per_year(infer_df, papers_year_path, metrics)
    print(f"[benchmark_id] papers per year plot saved to {papers_year_path}")
    
    # Plot 2: Distance by year with std (using inferred year)
    distance_year_path = overview_dir / "distance_by_year.png"
    _plot_distance_by_year(infer_df, merged_rows, metrics, distance_year_path)
    print(f"[benchmark_id] distance by year plot saved to {distance_year_path}")
    
    # Print ID match accuracies
    country_vals = distances.get("country", [])
    if country_vals:
        accuracy = sum(1 for v in country_vals if v == 0.0) / len(country_vals) * 100
        print(f"[benchmark_id] Country ID match accuracy: {accuracy:.1f}% ({sum(1 for v in country_vals if v == 0.0)}/{len(country_vals)})")
    
    hardware_vals = distances.get("hardware", [])
    if hardware_vals:
        accuracy = sum(1 for v in hardware_vals if v == 0.0) / len(hardware_vals) * 100
        print(f"[benchmark_id] Hardware ID match accuracy: {accuracy:.1f}% ({sum(1 for v in hardware_vals if v == 0.0)}/{len(hardware_vals)})")


if __name__ == "__main__":
    main()

