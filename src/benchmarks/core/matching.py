from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz as rf_fuzz


@dataclass
class ModelMatch:
    base_idx: int
    inf_idx: int
    score: float
    base_model: dict[str, Any]
    inf_model: dict[str, Any]


def normalize_model_name(name: str | None) -> str:
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


def model_name_similarity(name1: str, name2: str) -> float:
    n1 = normalize_model_name(name1)
    n2 = normalize_model_name(name2)

    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    return rf_fuzz.token_set_ratio(n1, n2) / 100.0


def is_valid_inferred_model(model: dict[str, Any]) -> bool:
    model_name = model.get("model", "")
    if not model_name:
        return False
    name_str = str(model_name).strip().lower()
    return name_str not in ("nan", "none", "null", "n/a", "")


def match_models_for_paper(
    base_models: list[dict[str, Any]],
    inf_models: list[dict[str, Any]],
    min_threshold: float = 0.4,
) -> list[ModelMatch]:
    if not base_models:
        return []

    valid_inf_models = [(i, m) for i, m in enumerate(inf_models) if is_valid_inferred_model(m)]
    if not valid_inf_models:
        return []

    scores: list[tuple[int, int, float]] = []
    for bi, base in enumerate(base_models):
        base_name = base.get("model", "")
        for orig_ii, inf in valid_inf_models:
            inf_name = inf.get("model", "")
            score = model_name_similarity(base_name, inf_name)
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
