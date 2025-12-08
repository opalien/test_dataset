#!/usr/bin/env python3
"""
Benchmark inferred results against the Epoch AI dataset with ID-based country and hardware matching.

- Reads ground truth from epoch.db (model_info + paper_info tables).
- Reads inference outputs from JSON files.
- Matches inferred models to ground truth models using a robust matching algorithm.
- For country field: maps text to ID using the country table via Jaro-Winkler.
- For hardware field: maps text to ID using the hardware table via token-based matching.
- Computes distances and generates plots.
"""
import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
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
    # Fields from epoch.ai documentation
    "model": ("text", "Model (JW)"),
    "country": ("id", "Country (ID match)"),  # Country (of organization)
    "year": ("numeric", "Year (rel)"),  # Publication date
    "parameters": ("numeric", "Parameters (rel)"),  # Parameters
    "hardware": ("id", "Hardware (ID match)"),  # Training hardware
    "hardware_number": ("numeric", "Hardware number (rel)"),  # Hardware quantity
    "training_compute": ("numeric", "Training compute (rel)"),  # Training compute (FLOP)
    "training_time": ("numeric", "Training time (rel)"),  # Training time (hours)
    "power_draw": ("numeric", "Power draw (rel)"),  # Training power draw (W) - calculated
    # Aliases for inference JSON fields
    "h_number": ("numeric", "Hardware number (rel)"),
}


# ---------------------------------------------------------------------------
# Country ID Lookup

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
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "prc": "China",
    "people's republic of china": "China",
    # South Korea variants -> map to "South Korea" (ID 228 in country table)
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "korea (republic of)": "South Korea",
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
        self.carbon_intensity: dict[int, float | None] = {}  # gCO2eq/kWh
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
                print(f"[benchmark_epoch] Warning: Could not load country table: {e}")
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
        
        # Check aliases
        if cache_key in COUNTRY_ALIASES:
            canonical = COUNTRY_ALIASES[cache_key]
            if canonical.lower() in self._name_to_id:
                result = self._name_to_id[canonical.lower()]
                self._cache[cache_key] = result
                return result
        
        # Exact match
        if cache_key in self._name_to_id:
            result = self._name_to_id[cache_key]
            self._cache[cache_key] = result
            return result
        
        # Jaro-Winkler match
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
    
    def get_carbon_intensity(self, cid: int | None) -> float | None:
        """Get carbon intensity in gCO2eq/kWh for a country ID."""
        if cid is None:
            return None
        return self.carbon_intensity.get(cid)


# ---------------------------------------------------------------------------
# Hardware ID Lookup

def _tokenize_hardware(name: str) -> list[str]:
    """Tokenize hardware name for matching."""
    if not name:
        return []
    s = name.lower().strip()
    
    # Normalize separators
    s = s.replace("-", " ").replace("_", " ")
    
    # Remove common noise words (but keep model identifiers!)
    for noise in ["(estimated)", "gpus", "gpu", "graphics card", "graphics", "accelerator", "accelerators"]:
        s = s.replace(noise, " ")
    
    tokens = [t.strip() for t in s.split() if t.strip()]
    
    expanded = []
    for t in tokens:
        # Split "1080ti" -> ["1080", "ti"]
        match = re.match(r'^(\d{4})(ti|super|xt)$', t)
        if match:
            expanded.append(match.group(1))
            expanded.append(match.group(2))
        else:
            expanded.append(t)
    
    return expanded


def _extract_model_numbers(tokens: list[str]) -> set[str]:
    """Extract key model identifiers (e.g., a100, h100, v100, 3090, etc.)
    
    Note: Form factors (sxm, nvl, pcie) are NOT included as they are often
    not extracted by LLMs and would cause false mismatches.
    """
    model_nums = set()
    for t in tokens:
        # Match GPU model numbers like a100, h100, v100, 3090, 4090
        if re.match(r'^[a-z]?\d{2,4}[a-z]?$', t):
            model_nums.add(t)
        # Performance suffixes (these ARE significant)
        if t in ("ti", "super", "xt"):
            model_nums.add(t)
    return model_nums


def _extract_memory_size(tokens: list[str]) -> str | None:
    """Extract memory size (e.g., '80gb', '40gb') from tokens."""
    for t in tokens:
        # Match patterns like "80gb", "40gb", "80g", "40g"
        match = re.match(r'^(\d+)g[b]?$', t)
        if match:
            return match.group(1) + "gb"
    return None


def _has_sxm_variant(name: str) -> bool:
    """Check if name contains SXM variant indicator."""
    return "sxm" in name.lower()


def _has_nvl_variant(name: str) -> bool:
    """Check if name contains NVL variant indicator."""
    return "nvl" in name.lower()


def _has_pcie_variant(name: str) -> bool:
    """Check if name contains PCIe variant indicator."""
    return "pcie" in name.lower() or "pci" in name.lower()


def _token_match_score(query_tokens: list[str], ref_tokens: list[str]) -> float:
    if not query_tokens or not ref_tokens:
        return 0.0
    
    query_models = _extract_model_numbers(query_tokens)
    ref_models = _extract_model_numbers(ref_tokens)
    
    if query_models and not query_models.intersection(ref_models):
        return 0.1
    
    total_score = 0.0
    matched_ref = set()
    
    for qt in query_tokens:
        best_score = 0.0
        best_idx = -1
        for i, rt in enumerate(ref_tokens):
            if i in matched_ref:
                continue
            if qt == rt:
                score = 1.0
            else:
                if qt in query_models or rt in ref_models:
                    if qt == rt:
                        score = 1.0
                    elif qt in rt or rt in qt:
                        score = 0.9
                    else:
                        score = JaroWinkler.normalized_similarity(qt, rt)
                        if score > 1.0:
                            score = score / 100.0
                        if score < 0.95:
                            score *= 0.5
                else:
                    score = JaroWinkler.normalized_similarity(qt, rt)
                    if score > 1.0:
                        score = score / 100.0
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_idx >= 0 and best_score > 0.7:
            matched_ref.add(best_idx)
        total_score += best_score
    
    avg_score = total_score / len(query_tokens)
    coverage = len(matched_ref) / len(ref_tokens) if ref_tokens else 0
    return 0.7 * avg_score + 0.3 * coverage


class HardwareLookup:
    """Lookup hardware ID from name using token-based Jaro-Winkler similarity."""
    
    def __init__(self, db_path: Path):
        self.hardware: dict[int, str] = {}
        self.hardware_power: dict[int, float | None] = {}
        self.hardware_compute: dict[int, float | None] = {}
        self._name_to_id: dict[str, int] = {}
        self._tokens: dict[int, list[str]] = {}
        self._cache: dict[str, int | None] = {}
        
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            try:
                # Epoch hardware table has: id, name, compute, power
                for row in cur.execute("SELECT id, name, compute, power FROM hardware").fetchall():
                    hid = int(row[0])
                    name = str(row[1]) if row[1] else ""
                    compute = float(row[2]) if row[2] is not None else None
                    power = float(row[3]) if row[3] is not None else None
                    
                    self.hardware[hid] = name
                    self.hardware_power[hid] = power
                    self.hardware_compute[hid] = compute
                    self._tokens[hid] = _tokenize_hardware(name)
                    if name:
                        self._name_to_id[name.lower().strip()] = hid
            except Exception as e:
                print(f"[benchmark_epoch] Warning: Could not load hardware table: {e}")
            conn.close()
    
    def get_id(self, name: str | None) -> int | None:
        if not name or not self.hardware:
            return None
        
        name_clean = str(name).strip()
        if not name_clean or name_clean.lower() in ("nan", "none", "null", "n/a", "na", "", "gpu", "gpus"):
            return None
        
        cache_key = name_clean.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if cache_key in self._name_to_id:
            result = self._name_to_id[cache_key]
            self._cache[cache_key] = result
            return result
        
        # Tokenize query
        query_tokens = _tokenize_hardware(name_clean)
        if not query_tokens:
            self._cache[cache_key] = None
            return None
        
        # Extract memory size from query (e.g., "80gb", "40gb")
        query_memory = _extract_memory_size(query_tokens)
        
        # Check if query mentions specific variant
        query_has_sxm = _has_sxm_variant(name_clean)
        query_has_nvl = _has_nvl_variant(name_clean)
        query_has_pcie = _has_pcie_variant(name_clean)
        query_has_variant = query_has_sxm or query_has_nvl or query_has_pcie
        
        # Collect all candidates with their scores
        candidates: list[tuple[int, float]] = []
        
        for hid, ref_tokens in self._tokens.items():
            if not ref_tokens:
                continue
            
            # Calculate token match score
            score = _token_match_score(query_tokens, ref_tokens)
            reverse_score = _token_match_score(ref_tokens, query_tokens)
            final_score = (score + reverse_score) / 2
            
            if final_score >= 0.6:
                candidates.append((hid, final_score))
        
        if not candidates:
            self._cache[cache_key] = None
            return None
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_score = candidates[0][1]
        
        # Get candidates with similar scores (within 0.1)
        close_candidates = [(hid, score) for hid, score in candidates if score >= best_score - 0.1]
        
        if len(close_candidates) > 1:
            # Apply smart filtering for ambiguous cases
            
            # 1. If query has memory size, prefer matching memory
            if query_memory:
                memory_matches = []
                for hid, score in close_candidates:
                    ref_name = self.hardware.get(hid, "").lower()
                    ref_tokens = _tokenize_hardware(ref_name)
                    ref_memory = _extract_memory_size(ref_tokens)
                    if ref_memory == query_memory:
                        memory_matches.append((hid, score))
                if memory_matches:
                    close_candidates = memory_matches
            
            # 2. If no specific variant in query, prefer SXM (most common for ML training)
            if not query_has_variant and len(close_candidates) > 1:
                sxm_matches = []
                for hid, score in close_candidates:
                    ref_name = self.hardware.get(hid, "")
                    if _has_sxm_variant(ref_name):
                        sxm_matches.append((hid, score))
                if sxm_matches:
                    close_candidates = sxm_matches
            
            # 3. If still multiple, pick highest score
            best_id = max(close_candidates, key=lambda x: x[1])[0]
        else:
            best_id = candidates[0][0]
        
        self._cache[cache_key] = best_id
        return best_id
    
    def get_name(self, hid: int | None) -> str | None:
        if hid is None:
            return None
        return self.hardware.get(hid)
    
    def get_power(self, hid: int | None) -> float | None:
        """Get hardware power in Watts (DB stores kW, we convert to W)."""
        if hid is None:
            return None
        power_kw = self.hardware_power.get(hid)
        if power_kw is not None:
            return power_kw * 1000.0
        return None


# ---------------------------------------------------------------------------
# Model Matching

# Tokens to ignore in name matching (don't affect meaning significantly)
NOISE_TOKENS = {
    'model', 'models', 'release', 'alpha', 'beta', 'rc', 'experimental',
    'the', 'a', 'an', 'of', 'for', 'with', 'and', 'or'
}

# Size/variant tokens that MUST match if present (very important for model identity)
SIZE_VARIANT_TOKENS = {
    # Size markers
    'mini', 'nano', 'micro', 'tiny', 'small', 'medium', 'large', 'xl', 'xxl', 'xxxl',
    'base', 'lite', 'light', 'plus', 'max', 'ultra', 'pro', 'main',
    # Capability markers
    'instruct', 'chat', 'thinking', 'reasoning', 'preview', 'vision', 'omni',
    # Number size markers (extracted separately)
}

# Pattern to extract size from model name (e.g., "7B", "70B", "1.5B", "1T")
SIZE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*([bBtT])\b')


def _extract_model_size(name: str) -> float | None:
    """Extract model size in billions of parameters from name."""
    if not name:
        return None
    match = SIZE_PATTERN.search(name)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == 'T':
        value *= 1000  # Convert T to B
    return value


def _normalize_model_name(name: str | None) -> str:
    """Normalize model name for comparison."""
    if not name:
        return ""
    s = str(name).lower().strip()
    # Handle "nan" as empty
    if s in ('nan', 'none', 'null', 'n/a', ''):
        return ""
    # Normalize all types of hyphens and dashes to regular hyphen then to space
    # Unicode hyphens: \u2010 (hyphen), \u2011 (non-breaking hyphen), \u2012 (figure dash),
    # \u2013 (en dash), \u2014 (em dash), \u2212 (minus)
    for h in ['\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2212', '‑', '–', '—']:
        s = s.replace(h, '-')
    # Remove common noise characters
    for noise in ['(', ')', '[', ']', '"', "'", ',', '.', ':']:
        s = s.replace(noise, ' ')
    # Normalize hyphens and underscores to spaces
    s = s.replace('-', ' ').replace('_', ' ')
    # Normalize whitespace
    return ' '.join(s.split())


def _tokenize_model_name(name: str) -> tuple[list[str], list[str], float | None]:
    """
    Tokenize model name into:
    - core_tokens: main identity tokens
    - variant_tokens: size/variant tokens that must match
    - size: extracted size in billions
    """
    normalized = _normalize_model_name(name)
    if not normalized:
        return [], [], None
    
    tokens = normalized.split()
    core = []
    variants = []
    
    for t in tokens:
        if t in NOISE_TOKENS:
            continue
        if t in SIZE_VARIANT_TOKENS:
            variants.append(t)
        else:
            core.append(t)
    
    size = _extract_model_size(name)
    return core, variants, size


def _model_name_similarity(name1: str, name2: str) -> float:
    """
    Compute similarity between two model names.
    Order-invariant: uses token-set matching.
    Considers: core tokens, variant tokens, and size compatibility.
    """
    n1 = _normalize_model_name(name1)
    n2 = _normalize_model_name(name2)
    
    # Handle empty names
    if not n1 or not n2:
        return 0.0
    
    # Exact match (after normalization)
    if n1 == n2:
        return 1.0
    
    # Parse both names
    core1, var1, size1 = _tokenize_model_name(name1)
    core2, var2, size2 = _tokenize_model_name(name2)
    
    # All tokens for order-invariant matching
    all_tokens1 = set(core1 + var1)
    all_tokens2 = set(core2 + var2)
    
    # Token overlap (Jaccard-like)
    if all_tokens1 and all_tokens2:
        intersection = all_tokens1 & all_tokens2
        union = all_tokens1 | all_tokens2
        token_overlap = len(intersection) / len(union) if union else 0.0
    else:
        token_overlap = 0.0
    
    # Use token_set_ratio for order-invariant fuzzy matching
    token_ratio = rf_fuzz.token_set_ratio(n1, n2) / 100.0
    
    # Variant mismatch penalty
    var_set1 = set(var1)
    var_set2 = set(var2)
    
    if var_set1 and var_set2:
        common_vars = var_set1 & var_set2
        all_vars = var_set1 | var_set2
        if len(all_vars) > 0 and len(common_vars) == 0:
            # Conflicting variants = penalty
            variant_penalty = 0.5
        else:
            variant_penalty = 1.0
    elif var_set1 or var_set2:
        variant_penalty = 0.85
    else:
        variant_penalty = 1.0
    
    # Size compatibility check
    if size1 is not None and size2 is not None:
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 1.0
        if size_ratio < 0.3:
            # Very different sizes
            return 0.15
        size_match = size_ratio
    else:
        size_match = 1.0
    
    # Combine: token_ratio (order-invariant) + token_overlap bonus + penalties
    base_score = 0.6 * token_ratio + 0.4 * token_overlap
    final_score = base_score * variant_penalty * (0.5 + 0.5 * size_match)
    
    return min(1.0, final_score)


def _parameter_similarity(params1: Any, params2: Any) -> float | None:
    """Compute similarity based on parameter count (relative difference)."""
    try:
        p1 = float(params1) if params1 is not None else None
        p2 = float(params2) if params2 is not None else None
    except (ValueError, TypeError):
        return None
    
    if p1 is None or p2 is None or math.isnan(p1) or math.isnan(p2):
        return None
    
    if p1 == 0 and p2 == 0:
        return 1.0
    
    denom = max(abs(p1), abs(p2))
    if denom == 0:
        return None
    
    rel_diff = abs(p1 - p2) / denom
    return max(0.0, 1.0 - rel_diff)


@dataclass
class ModelMatch:
    """Result of matching an inferred model to a ground truth model."""
    base_idx: int
    inf_idx: int
    score: float
    base_model: dict[str, Any]
    inf_model: dict[str, Any]


def _is_valid_inferred_model(model: dict[str, Any]) -> bool:
    """Check if an inferred model record is valid (not empty/nan)."""
    model_name = model.get("model", "")
    if not model_name:
        return False
    name_str = str(model_name).strip().lower()
    if name_str in ("nan", "none", "null", "n/a", ""):
        return False
    return True


def _compute_column_similarity(base: dict[str, Any], inf: dict[str, Any], field: str) -> float | None:
    """Compute similarity for a specific column between base and inferred model."""
    base_val = base.get(field)
    inf_val = inf.get(field)
    
    # Handle None/NaN
    if base_val is None or inf_val is None:
        return None
    if isinstance(base_val, float) and math.isnan(base_val):
        return None
    if isinstance(inf_val, float) and math.isnan(inf_val):
        return None
    
    # Numeric comparison
    try:
        b_num = float(base_val)
        i_num = float(inf_val)
        if math.isnan(b_num) or math.isnan(i_num):
            return None
        if b_num == 0 and i_num == 0:
            return 1.0
        denom = max(abs(b_num), abs(i_num))
        if denom == 0:
            return None
        rel_diff = abs(b_num - i_num) / denom
        return max(0.0, 1.0 - rel_diff)
    except (ValueError, TypeError):
        pass
    
    # String comparison
    try:
        b_str = str(base_val).strip().lower()
        i_str = str(inf_val).strip().lower()
        if not b_str or not i_str:
            return None
        # Token-set ratio for order-invariant matching
        return rf_fuzz.token_set_ratio(b_str, i_str) / 100.0
    except:
        return None


def match_models_for_paper(
    base_models: list[dict[str, Any]],
    inf_models: list[dict[str, Any]],
    *,
    min_threshold: float = 0.4,
) -> list[ModelMatch]:
    """
    Match inferred models to ground truth models for a single paper.
    Uses a greedy best-match algorithm with multi-column similarity.
    
    Scoring combines:
    - Model name similarity (order-invariant, 40%)
    - Parameter similarity (30%)
    - Other column similarities when available (30%)
    """
    if not base_models:
        return []
    
    # Filter valid inferred models
    valid_inf_models = [(i, m) for i, m in enumerate(inf_models) if _is_valid_inferred_model(m)]
    
    if not valid_inf_models:
        return []
    
    # Columns to use for matching (besides name and parameters)
    extra_columns = ["hardware_number", "training_time", "year"]
    
    # Compute pairwise scores
    scores: list[tuple[int, int, float]] = []
    for bi, base in enumerate(base_models):
        base_name = base.get("model", "")
        
        for orig_ii, inf in valid_inf_models:
            inf_name = inf.get("model", "")
            
            # Name similarity (order-invariant)
            name_sim = _model_name_similarity(base_name, inf_name)
            
            # Skip very poor name matches
            if name_sim < 0.15:
                continue
            
            # Parameter similarity
            param_sim = _parameter_similarity(base.get("parameters"), inf.get("parameters"))
            
            # Extra column similarities
            extra_sims = []
            for col in extra_columns:
                sim = _compute_column_similarity(base, inf, col)
                if sim is not None:
                    extra_sims.append(sim)
            
            # Compute final score
            # Weight: name (40%) + params (30%) + extras (30%)
            components = []
            components.append((name_sim, 0.4))
            
            if param_sim is not None:
                components.append((param_sim, 0.3))
            else:
                # Redistribute weight to name if no params
                components[0] = (name_sim, 0.5)
            
            if extra_sims:
                avg_extra = sum(extra_sims) / len(extra_sims)
                components.append((avg_extra, 0.3 if param_sim is not None else 0.2))
            
            # Normalize weights
            total_weight = sum(w for _, w in components)
            if total_weight > 0:
                score = sum(s * w for s, w in components) / total_weight
            else:
                score = name_sim
            
            # Bonus for exact name match
            if _normalize_model_name(base_name) == _normalize_model_name(inf_name):
                score = min(1.0, score + 0.1)
            
            scores.append((bi, orig_ii, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy assignment
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

def load_ground_truth(db_path: Path, country_lookup: CountryLookup, hardware_lookup: HardwareLookup, *, arxiv_only: bool = False, confidence_filter: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ground truth from epoch.db.
    Returns (paper_df, model_df) where:
    - paper_df: one row per paper with paper-level info
    - model_df: one row per model with model-level info
    
    Also calculates derived field power_draw if missing.
    
    Args:
        arxiv_only: If True, only include papers from arxiv.
        confidence_filter: If set, only include papers with this confidence level.
                          Valid values: "Confident", "Likely", "Speculative", "Unknown"
    """
    conn = sqlite3.connect(str(db_path))
    
    # Load confidence from epoch table (link is the join key)
    confidence_df = pd.read_sql_query("""
        SELECT Link as link, Confidence as confidence
        FROM epoch
        WHERE Link IS NOT NULL
    """, conn)
    
    # Load paper info
    if arxiv_only:
        paper_df = pd.read_sql_query("""
            SELECT id_paper, link, abstract, country, id_country, year
            FROM paper_info
            WHERE link LIKE '%arxiv%'
        """, conn)
    else:
        paper_df = pd.read_sql_query("""
            SELECT id_paper, link, abstract, country, id_country, year
            FROM paper_info
        """, conn)
    
    # Merge confidence into paper_df
    # Drop duplicates in confidence_df (keep first) to avoid duplicate rows after merge
    confidence_df = confidence_df.drop_duplicates(subset=["link"], keep="first")
    paper_df = paper_df.merge(confidence_df, on="link", how="left")
    # Fill missing confidence with "Unknown"
    paper_df["confidence"] = paper_df["confidence"].fillna("Unknown")
    # Ensure no duplicate papers
    paper_df = paper_df.drop_duplicates(subset=["id_paper"], keep="first")
    
    # Filter by confidence if specified
    if confidence_filter:
        paper_df = paper_df[paper_df["confidence"] == confidence_filter]
    
    # Get the set of valid paper IDs (for filtering models)
    valid_paper_ids = set(paper_df["id_paper"].tolist())
    
    # Load model info (without co2eq - not in epoch.ai)
    model_df = pd.read_sql_query("""
        SELECT id_model, id_paper, model, architecture, parameters,
               id_hardware, hardware, hardware_number,
               training_compute, training_time, power_draw
        FROM model_info
    """, conn)
    
    conn.close()
    
    # Filter models to only include those from valid papers
    model_df = model_df[model_df["id_paper"].isin(valid_paper_ids)]
    
    filter_desc = []
    if arxiv_only:
        filter_desc.append("arxiv")
    if confidence_filter:
        filter_desc.append(f"confidence={confidence_filter}")
    if filter_desc:
        print(f"[benchmark_epoch] Filtered ({', '.join(filter_desc)}): {len(paper_df)} papers, {len(model_df)} models")
    
    # Enrich paper_df with country ID if not set
    if "id_country" in paper_df.columns:
        def ensure_country_id(row):
            if pd.notna(row.get("id_country")):
                return int(row["id_country"])
            return country_lookup.get_id(row.get("country"))
        paper_df["id_country"] = paper_df.apply(ensure_country_id, axis=1)
    
    # Enrich model_df with hardware ID if not set
    if "id_hardware" in model_df.columns:
        def ensure_hardware_id(row):
            if pd.notna(row.get("id_hardware")):
                return int(row["id_hardware"])
            return hardware_lookup.get_id(row.get("hardware"))
        model_df["id_hardware"] = model_df.apply(ensure_hardware_id, axis=1)
    
    # Calculate power_draw if missing using epoch.ai formula:
    # Training power draw (W) = PUE × Server overhead × Power per GPU × Hardware quantity
    def calc_power_draw(row):
        if pd.notna(row.get("power_draw")) and row.get("power_draw") > 0:
            return row["power_draw"]
        hn = row.get("hardware_number")
        hid = row.get("id_hardware")
        if pd.isna(hn) or pd.isna(hid):
            return None
        # Get hardware power from lookup
        hp = hardware_lookup.get_power(int(hid))
        if hp is None:
            return None
        # Get year for PUE calculation
        year = paper_df[paper_df["id_paper"] == row["id_paper"]]["year"].iloc[0] if row["id_paper"] in paper_df["id_paper"].values else 2024
        if pd.isna(year):
            year = 2024
        # PUE calculation (from epoch.ai docs)
        if year >= 2009:
            pue = 1.23 * math.exp((year - 2008) * math.log(1.08/1.23)/16)
        else:
            pue = 1.23
        # Server overhead (1.82 for multi-GPU, 1.0 for single GPU)
        server_overhead = 1.82 if float(hn) > 1 else 1.0
        # Calculate instantaneous power in Watts
        return pue * server_overhead * hp * float(hn)
    
    model_df["power_draw"] = model_df.apply(calc_power_draw, axis=1)
    
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
            print(f"[benchmark_epoch] skip {path}: {exc}")
            continue
        
        if not isinstance(data, list):
            continue
        
        for entry in data:
            pid = entry.get("id_paper")
            if pid is None:
                continue
            
            models = entry.get("models", [])
            if not models:
                # No models - create empty record
                records.append({"id_paper": int(pid)})
                continue
            
            for model in models:
                if not isinstance(model, dict):
                    continue
                
                cleaned = {k: _extract_scalar(v) for k, v in model.items()}
                record = {"id_paper": int(pid)}
                
                for field in tracked_fields:
                    # Handle aliases (h_* -> hardware_*)
                    if field.startswith("hardware_") and field.replace("hardware_", "h_") in cleaned:
                        record[field] = cleaned.get(field.replace("hardware_", "h_"))
                    else:
                        record[field] = cleaned.get(field)
                
                # Also store model name
                record["model"] = cleaned.get("model")
                
                # Convert country to ID
                if "country" in record and record["country"]:
                    record["id_country"] = country_lookup.get_id(record["country"])
                
                # Convert hardware to ID
                if "hardware" in record and record["hardware"]:
                    record["id_hardware"] = hardware_lookup.get_id(record["hardware"])
                    # Infer hardware_power from ID if missing
                    if _is_missing(record.get("hardware_power")) and record.get("id_hardware"):
                        hw_power = hardware_lookup.get_power(record["id_hardware"])
                        if hw_power is not None:
                            record["hardware_power"] = hw_power
                
                # Calculate power_draw using Epoch formula (instantaneous power in Watts)
                # Formula: power_draw (W) = PUE × Server_overhead × Power_per_GPU (W) × Hardware_quantity
                # PUE for 2024-2025 ≈ 1.07-1.08
                # Server_overhead = 1.82 if hardware_number > 1, else 1.0
                if _is_missing(record.get("power_draw")):
                    hn = _parse_numeric(record.get("hardware_number"))
                    hp = _parse_numeric(record.get("hardware_power"))  # in Watts
                    
                    if hn is not None and hp is not None and hn > 0:
                        # Get year for PUE calculation
                        year = _parse_numeric(record.get("year"))
                        if year is None:
                            year = 2024  # Default to recent
                        
                        # Calculate PUE based on year (Epoch formula)
                        if year >= 2009:
                            pue = 1.23 * math.exp((year - 2008) * math.log(1.08/1.23)/16)
                        else:
                            pue = 1.23
                        
                        # Server overhead (1.82 for multi-GPU, 1.0 for single GPU)
                        server_overhead = 1.82 if hn > 1 else 1.0
                        
                        # Calculate instantaneous power in Watts
                        record["power_draw"] = pue * server_overhead * hp * hn
                
                records.append(record)
    
    if not records:
        return pd.DataFrame(columns=["id_paper", "model"] + tracked_fields + ["id_country", "id_hardware"])
    
    df = pd.DataFrame(records).sort_values(["id_paper"]).reset_index(drop=True)
    # Remove co2eq_method column if it exists
    if "co2eq_method" in df.columns:
        df = df.drop(columns=["co2eq_method"])
    return df


# ---------------------------------------------------------------------------
# Metrics

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
    """
    Compute distances for matched model pairs.
    """
    distances: dict[str, list[float]] = {m.field: [] for m in metrics}
    missing: dict[str, list[int]] = {m.field: [0, 0, 0] for m in metrics}
    result_rows: list[dict[str, Any]] = []
    
    # Build paper-level info lookup
    paper_info = paper_df.set_index("id_paper").to_dict("index")
    
    for match in matches:
        base = match.base_model
        inf = match.inf_model
        pid = base.get("id_paper") or inf.get("id_paper")
        
        # Get paper-level info
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
            
            # Determine base and inf values
            # Paper-level fields
            if field in ("country", "year") and field not in base:
                base_val = pinfo.get(field)
            else:
                base_val = base.get(field)
            
            inf_val = inf.get(field)
            
            # ID-based fields
            if metric.kind == "id":
                if field == "country":
                    base_id = pinfo.get("id_country")
                    inf_id = inf.get("id_country")
                elif field == "hardware":
                    base_id = base.get("id_hardware")
                    inf_id = inf.get("id_hardware")
                else:
                    continue
                
                # Clean IDs
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
                    if inf_id is None:
                        miss[0] += 1
                    else:
                        miss[2] += 1
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
                    if inf_num is None:
                        miss[0] += 1
                    else:
                        miss[2] += 1
                    continue
                
                if inf_num is None:
                    miss[1] += 1
                    continue
                
                rel = _signed_rel(inf_num, base_num)
                if rel is None:
                    miss[1] += 1
                    continue
                
                dist = abs(rel)
                distances[field].append(dist)
                out_row[f"{field}_dist"] = dist
                
            else:  # text
                base_txt = None if _is_missing(base_val) else str(base_val).strip()
                inf_txt = None if _is_missing(inf_val) else str(inf_val).strip()
                
                out_row[f"{field}_base"] = base_txt
                out_row[f"{field}_inf"] = inf_txt
                
                if base_txt is None:
                    if inf_txt is None:
                        miss[0] += 1
                    else:
                        miss[2] += 1
                    continue
                
                if inf_txt is None:
                    miss[1] += 1
                    continue
                
                jw_dist = 1.0 - _jw(base_txt, inf_txt)
                distances[field].append(jw_dist)
                out_row[f"{field}_dist"] = jw_dist
        
        result_rows.append(out_row)
    
    missing = {k: tuple(v) for k, v in missing.items()}  # type: ignore
    return distances, missing, result_rows  # type: ignore


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


def _plot_gradient_stacked(out_path: Path, metric: Metric, distances: list[float], missing: tuple[int, int, int], n_bins: int = 10) -> None:
    """Create a gradient stacked bar plot for a single metric."""
    total_rows = len(distances) + sum(missing)
    if total_rows == 0:
        return
    
    fig, ax = plt.subplots(figsize=(4, 6))
    
    frac_both_null = missing[0] / total_rows if total_rows else 0
    frac_infer_null = missing[1] / total_rows if total_rows else 0
    frac_base_null = missing[2] / total_rows if total_rows else 0
    
    cmap = plt.get_cmap('viridis')
    
    bottom = 0.0
    if distances:
        abs_distances = [abs(d) for d in distances]
        counts, _ = np.histogram(abs_distances, bins=n_bins, range=(0.0, 1.0))
        
        for i, count in enumerate(counts):
            if count == 0:
                continue
            height = count / total_rows
            color = cmap(i / max(1, n_bins - 1))
            ax.bar([0], [height], bottom=bottom, color=color, edgecolor='none', width=0.6)
            bottom += height
    
    if frac_both_null > 0:
        ax.bar([0], [frac_both_null], bottom=bottom, color=BAR_COLORS[0], edgecolor='black', width=0.6)
        bottom += frac_both_null
    if frac_infer_null > 0:
        ax.bar([0], [frac_infer_null], bottom=bottom, color=BAR_COLORS[1], edgecolor='black', width=0.6)
        bottom += frac_infer_null
    if frac_base_null > 0:
        ax.bar([0], [frac_base_null], bottom=bottom, color=BAR_COLORS[2], edgecolor='black', width=0.6)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('fraction')
    ax.set_xticks([0])
    ax.set_xticklabels([metric.display_name], rotation=45, ha='right')
    ax.set_title(f'{metric.display_name}')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('distance', fontsize=8)
    
    handles = [
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_all_gradient_stacked(out_path: Path, metrics: list[Metric], distances: dict[str, list[float]], missing: dict[str, tuple[int, int, int]], n_bins: int = 10) -> None:
    """Create a combined gradient stacked bar plot for all metrics."""
    if not metrics:
        return
    
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
    ax.set_ylabel('fraction')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m.field for m in valid_metrics], rotation=45, ha='right')
    ax.set_title('All metrics: gradient stack')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('distance', fontsize=8)
    
    handles = [
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[0], edgecolor='black', label='both null'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[1], edgecolor='black', label='infer null'),
        plt.matplotlib.patches.Patch(facecolor=BAR_COLORS[2], edgecolor='black', label='base null'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_match_score_distribution(matches: list[ModelMatch], out_path: Path) -> None:
    """Plot distribution of match scores."""
    if not matches:
        return
    
    scores = [m.score for m in matches]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=20, range=(0, 1), color=HIST_COLOR, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax.axvline(x=np.mean(scores), color='red', linestyle='-', linewidth=2, label=f'Mean ({np.mean(scores):.2f})')
    
    ax.set_xlabel('Match Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Model Match Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI

def run_single_benchmark(
    db_path: Path,
    outputs_path: Path,
    out_dir: Path,
    tracked_fields: list[str],
    country_lookup: CountryLookup,
    hardware_lookup: HardwareLookup,
    *,
    arxiv_only: bool = False,
    confidence_filter: str | None = None,
    label: str = "all",
) -> dict[str, Any] | None:
    """
    Run benchmark for a specific configuration.
    Returns summary dict or None if no matches.
    """
    # Load ground truth with filters
    paper_df, model_df = load_ground_truth(
        db_path, country_lookup, hardware_lookup,
        arxiv_only=arxiv_only,
        confidence_filter=confidence_filter
    )
    
    if len(paper_df) == 0:
        print(f"[benchmark_epoch/{label}] No papers found with filter; skipping.")
        return None
    
    print(f"[benchmark_epoch/{label}] Loaded {len(paper_df)} papers, {len(model_df)} models")
    
    # Load inference
    infer_df = load_inference(outputs_path, tracked_fields, country_lookup, hardware_lookup)
    
    # Filter inference to only include papers that exist in ground truth
    valid_gt_paper_ids = set(paper_df["id_paper"].tolist())
    infer_df = infer_df[infer_df["id_paper"].isin(valid_gt_paper_ids)]
    
    if len(infer_df) == 0:
        print(f"[benchmark_epoch/{label}] No inference data after filter; skipping.")
        return None
    
    # Match models for each paper
    all_matches: list[ModelMatch] = []
    matched_papers = 0
    unmatched_papers = 0
    
    inferred_paper_ids = set(infer_df["id_paper"].unique())
    
    for pid in inferred_paper_ids:
        base_models = model_df[model_df["id_paper"] == pid].to_dict("records")
        if not base_models:
            unmatched_papers += 1
            continue
        
        inf_models = infer_df[infer_df["id_paper"] == pid].to_dict("records")
        if not inf_models:
            unmatched_papers += 1
            continue
        
        matches = match_models_for_paper(base_models, inf_models)
        if matches:
            all_matches.extend(matches)
            matched_papers += 1
        else:
            unmatched_papers += 1
    
    print(f"[benchmark_epoch/{label}] Matched {len(all_matches)} model pairs from {matched_papers} papers")
    
    if not all_matches:
        print(f"[benchmark_epoch/{label}] No matches found; skipping.")
        return None
    
    # Build metrics
    metrics = build_metrics(tracked_fields)
    if not metrics:
        return None
    
    # Compute distances
    distances, missing, result_rows = compute_distances(all_matches, paper_df, metrics)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    result_df = pd.DataFrame(result_rows)
    merged_path = out_dir / "matched_models.csv"
    result_df.to_csv(merged_path, index=False)
    
    # Plot main benchmark
    plot_path = out_dir / "benchmark.png"
    _plot_metrics_grid(plot_path, metrics, distances, missing)
    
    # Summary
    summary_path = out_dir / "distance_summary.csv"
    global_score = save_distance_summary(distances, summary_path)
    print(f"[benchmark_epoch/{label}] Global distance: {global_score:.4f}")
    
    # Percentage table
    table_path = out_dir / "percentage_table.png"
    _create_percentage_table(distances, table_path)
    
    # Overview plots
    overview_dir = out_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    
    # Build metrics for plotting
    plot_metrics = build_metrics(tracked_fields)
    _plot_all_gradient_stacked(overview_dir / "grad_stacked_all.png", plot_metrics, distances, missing)
    _plot_match_score_distribution(all_matches, overview_dir / "match_scores.png")
    
    # Individual gradient stacked plots per metric
    for metric in plot_metrics:
        vals = [float(v) for v in distances.get(metric.field, []) if v is not None]
        miss = missing.get(metric.field, (0, 0, 0))
        grad_path = overview_dir / f"grad_stacked_{metric.field}.png"
        _plot_gradient_stacked(grad_path, metric, vals, miss)
    
    return {
        "label": label,
        "papers": len(paper_df),
        "models": len(model_df),
        "matched_pairs": len(all_matches),
        "matched_papers": matched_papers,
        "global_distance": global_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference vs Epoch AI dataset with ID-based matching.")
    parser.add_argument("--database", required=True, help="Path to epoch.db")
    parser.add_argument("--outputs", required=True, help="JSON file or directory with inference results")
    parser.add_argument("--out-dir", default="results_epoch/benchmark", help="Output directory")
    parser.add_argument(
        "--columns",
        nargs="*",
        choices=sorted(COLUMN_META.keys()),
        help="Columns to benchmark (default: all)",
    )
    parser.add_argument(
        "--arxiv-only",
        action="store_true",
        help="Only include papers from arxiv (filter by link containing 'arxiv')",
    )
    parser.add_argument(
        "--no-confidence-split",
        action="store_true",
        help="Skip confidence-based sub-benchmarks (only run 'all')",
    )
    args = parser.parse_args()

    db_path = Path(args.database)
    outputs_path = Path(args.outputs)
    out_dir = Path(args.out_dir)

    # Load lookup tables
    country_lookup = CountryLookup(db_path)
    hardware_lookup = HardwareLookup(db_path)
    print(f"[benchmark_epoch] Loaded {len(country_lookup.countries)} countries, {len(hardware_lookup.hardware)} hardware")

    # Determine tracked fields (aligned with epoch.ai documentation)
    all_fields = ["model", "parameters", "hardware", "hardware_number", 
                  "training_time", "training_compute", "power_draw", 
                  "country", "year"]
    if args.columns:
        tracked_fields = [f for f in args.columns if f in all_fields]
    else:
        tracked_fields = all_fields

    # Define benchmark configurations
    configs = [
        {"confidence_filter": None, "label": "all", "subdir": "all"},
    ]
    
    if not args.no_confidence_split:
        configs.extend([
            {"confidence_filter": "Confident", "label": "confident", "subdir": "confident"},
            {"confidence_filter": "Likely", "label": "likely", "subdir": "likely"},
            {"confidence_filter": "Speculative", "label": "speculative", "subdir": "speculative"},
        ])
    
    # Run benchmarks
    summaries = []
    for config in configs:
        print(f"\n{'='*60}")
        print(f"[benchmark_epoch] Running benchmark: {config['label']}")
        print(f"{'='*60}")
        
        result = run_single_benchmark(
            db_path=db_path,
            outputs_path=outputs_path,
            out_dir=out_dir / config["subdir"],
            tracked_fields=tracked_fields,
            country_lookup=country_lookup,
            hardware_lookup=hardware_lookup,
            arxiv_only=args.arxiv_only,
            confidence_filter=config["confidence_filter"],
            label=config["label"],
        )
        
        if result:
            summaries.append(result)
    
    # Print final summary
    if summaries:
        print(f"\n{'='*60}")
        print("[benchmark_epoch] SUMMARY")
        print(f"{'='*60}")
        print(f"{'Config':<15} {'Papers':>8} {'Models':>8} {'Matched':>8} {'Distance':>10}")
        print("-" * 55)
        for s in summaries:
            print(f"{s['label']:<15} {s['papers']:>8} {s['models']:>8} {s['matched_pairs']:>8} {s['global_distance']:>10.4f}")
    
    # Also save overall summary CSV
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out_dir / "confidence_summary.csv", index=False)
        print(f"\n[benchmark_epoch] Saved confidence summary to {out_dir / 'confidence_summary.csv'}")


if __name__ == "__main__":
    main()

