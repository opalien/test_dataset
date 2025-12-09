import re
import sqlite3
from pathlib import Path

from rapidfuzz.distance import JaroWinkler


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
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "korea (republic of)": "South Korea",
    "rok": "South Korea",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "deutschland": "Germany",
    "holland": "Netherlands",
    "russian federation": "Russia",
}


class CountryLookup:
    def __init__(self, db_path: Path) -> None:
        self.countries: dict[int, str] = {}
        self.carbon_intensity: dict[int, float | None] = {}
        self._name_to_id: dict[str, int] = {}
        self._cache: dict[str, int | None] = {}

        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            for row in cur.execute("SELECT id_country, name, carbon_intensity FROM country").fetchall():
                cid, name = int(row[0]), str(row[1]) if row[1] else ""
                self.countries[cid] = name
                self.carbon_intensity[cid] = float(row[2]) if row[2] is not None else None
                if name:
                    self._name_to_id[name.lower().strip()] = cid
        except Exception:
            pass
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

    def get_carbon_intensity(self, cid: int | None) -> float | None:
        if cid is None:
            return None
        return self.carbon_intensity.get(cid)


def _tokenize_hardware(name: str) -> list[str]:
    if not name:
        return []
    s = name.lower().strip().replace("-", " ").replace("_", " ")
    for noise in ["(estimated)", "gpus", "gpu", "graphics card", "graphics", "accelerator", "accelerators"]:
        s = s.replace(noise, " ")
    tokens = [t.strip() for t in s.split() if t.strip()]

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
    model_nums = set()
    for t in tokens:
        if re.match(r'^[a-z]?\d{2,4}[a-z]?$', t):
            model_nums.add(t)
        if t in ("ti", "super", "xt"):
            model_nums.add(t)
    return model_nums


def _token_match_score(query_tokens: list[str], ref_tokens: list[str]) -> float:
    if not query_tokens or not ref_tokens:
        return 0.0

    query_models = _extract_model_numbers(query_tokens)
    ref_models = _extract_model_numbers(ref_tokens)

    if query_models and not query_models.intersection(ref_models):
        return 0.1

    total_score = 0.0
    matched_ref: set[int] = set()

    for qt in query_tokens:
        best_score = 0.0
        best_idx = -1
        for i, rt in enumerate(ref_tokens):
            if i in matched_ref:
                continue
            if qt == rt:
                score = 1.0
            elif qt in query_models or rt in ref_models:
                if qt in rt or rt in qt:
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
    def __init__(self, db_path: Path) -> None:
        self.hardware: dict[int, str] = {}
        self.hardware_power: dict[int, float | None] = {}
        self.hardware_compute: dict[int, float | None] = {}
        self._name_to_id: dict[str, int] = {}
        self._tokens: dict[int, list[str]] = {}
        self._cache: dict[str, int | None] = {}

        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        try:
            for row in cur.execute("SELECT id, name, compute, power FROM hardware").fetchall():
                hid = int(row[0])
                name = str(row[1]) if row[1] else ""
                self.hardware[hid] = name
                self.hardware_compute[hid] = float(row[2]) if row[2] is not None else None
                self.hardware_power[hid] = float(row[3]) if row[3] is not None else None
                self._tokens[hid] = _tokenize_hardware(name)
                if name:
                    self._name_to_id[name.lower().strip()] = hid
        except Exception:
            pass
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

        query_tokens = _tokenize_hardware(name_clean)
        if not query_tokens:
            self._cache[cache_key] = None
            return None

        best_id: int | None = None
        best_score = 0.0

        for hid, ref_tokens in self._tokens.items():
            if not ref_tokens:
                continue
            score = _token_match_score(query_tokens, ref_tokens)
            reverse_score = _token_match_score(ref_tokens, query_tokens)
            final_score = (score + reverse_score) / 2

            if final_score > best_score:
                best_score = final_score
                best_id = hid

        if best_score < 0.6:
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
        return power_kw * 1000.0 if power_kw is not None else None

    def get_compute(self, hid: int | None) -> float | None:
        if hid is None:
            return None
        return self.hardware_compute.get(hid)
