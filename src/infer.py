#!/usr/bin/env python3
import argparse
import ast
import json
import os
import time
from pathlib import Path
from typing import Any

import dataset
from google import genai
from google.genai import types as genai_types


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")
DEFAULT_DB = os.environ.get("INFER_DATABASE", "data/greenmir.db")
DEFAULT_TABLE = os.environ.get("INFER_TABLE", "paper_text")
DEFAULT_FIELDS = os.environ.get(
    "INFER_FIELDS",
    ["parameters", "hardware", "h_power", "h_number", "h_compute", "country", "year", "co2eq", "training_time", "training_compute"],
)

MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "100"))
BACKOFF_SECONDS = float(os.environ.get("GEMINI_BACKOFF_SECONDS", "10.0"))
RATE_LIMIT_DELAY = float(os.environ.get("GEMINI_RATE_DELAY", "0.0"))


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _normalize_model_name(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def _extract_text_response(resp: Any) -> str:
    txt = getattr(resp, "text", None)
    if txt:
        return str(txt).strip()
    candidates = getattr(resp, "candidates", None) or []
    if candidates and isinstance(candidates, list):
        parts = getattr(candidates[0], "content", None)
        if parts and hasattr(parts, "parts"):
            texts = []
            for part in parts.parts:
                if hasattr(part, "text"):
                    texts.append(part.text)
                elif isinstance(part, dict):
                    texts.append(part.get("text", ""))
            return "\n".join(t for t in texts if t).strip()
    return ""


def send_prompt(prompt: str, *, client: genai.Client, model: str, timeout: float, thinking_budget: int | None, label: str) -> str:
    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.0,
        top_k=1,
    )
    if thinking_budget is not None and "2.5" in model:
        config.thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if RATE_LIMIT_DELAY > 0:
                time.sleep(RATE_LIMIT_DELAY)
            resp = client.models.generate_content(
                model=_normalize_model_name(model),
                contents=prompt,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            thoughts_tokens = getattr(usage, "thoughts_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            if thoughts_tokens is not None:
                print(f"Thoughts tokens: {thoughts_tokens}")
            if output_tokens is not None:
                print(f"Output tokens: {output_tokens}")

            text = _extract_text_response(resp)
            if text:
                return text
            # If empty text, retry until max attempts
            log(f"[{label}] empty response; retry {attempt}/{MAX_RETRIES}")
        except Exception as err:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gemini request failed after retries: {err}") from err
            sleep_time = BACKOFF_SECONDS * attempt
            log(f"[{label}] error; retry {attempt}/{MAX_RETRIES} in {sleep_time:.1f}s ({err})")
            time.sleep(sleep_time)
    return ""


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini extraction over papers in a dataset DB.")
    parser.add_argument("--database", default=DEFAULT_DB, help="SQLite DB path (dataset URI also allowed).")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Table containing paper text (default: paper_text).")
    parser.add_argument("--questions-dir", default="old_version/v4/questions_estimation", help="Directory with prompt files.")
    parser.add_argument("--fields", nargs="*", default=DEFAULT_FIELDS, help="Fields to extract for each model (prompts must exist).")
    parser.add_argument("--output", default="papers.json", help="Output JSON path.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N papers.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N papers.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"), required=True, help="Google API key.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout for Gemini calls.")
    parser.add_argument("--thinking-budget", type=int, default=None, help="Optional thinkingBudget (only 2.5 models).")
    args = parser.parse_args()

    # Resolve prompts
    qdir = Path(args.questions_dir)
    model_prompt_path = qdir / "model_enumeration.txt"
    if not model_prompt_path.exists():
        parser.error(f"Missing model enumeration prompt: {model_prompt_path}")
    questions = {"model": load_prompt(model_prompt_path)}
    for field in args.fields:
        p = qdir / f"{field}.txt"
        if not p.exists():
            parser.error(f"Missing prompt for field '{field}': {p}")
        questions[field] = load_prompt(p)

    db = dataset.connect(f"sqlite:///{args.database}" if "://" not in args.database else args.database)
    table = db[args.table]

    # Build set of arxiv paper IDs for filtering
    arxiv_ids: set[int] = set()
    if "paper_info" in db.tables:
        for row in db["paper_info"]:
            link = row.get("link") or ""
            if "arxiv" in link.lower():
                arxiv_ids.add(row.get("id_paper"))
        log(f"Found {len(arxiv_ids)} arxiv papers to process")

    client = genai.Client(api_key=args.api_key)

    papers: list[dict[str, Any]] = []
    processed = 0
    for idx, row in enumerate(table, start=1):
        if idx <= args.offset:
            continue
        id_paper = row.get("id_paper")
        # Skip non-arxiv papers if we have the filter
        if arxiv_ids and id_paper not in arxiv_ids:
            continue
        if args.limit is not None and processed >= args.limit:
            break
        article_text = row.get("text") or ""
        log(f"[paper {id_paper}] enumerate models")
        model_prompt = questions["model"].replace("{article_text}", article_text)
        model_resp = send_prompt(
            model_prompt,
            client=client,
            model=args.model,
            timeout=args.timeout,
            thinking_budget=args.thinking_budget,
            label=f"model_enumeration id_paper={id_paper}",
        )
        try:
            model_list = ast.literal_eval(model_resp)
            if not isinstance(model_list, list):
                raise ValueError("model enumeration did not return a list")
        except Exception as err:
            log(f"[paper {id_paper}] failed to parse model list: {err}")
            continue

        paper_entry = {"id_paper": id_paper, "models": []}
        for model_name in model_list:
            model_entry = {"model": model_name}
            for field in args.fields:
                prompt = questions[field].replace("{model_name}", str(model_name)).replace("{article_text}", article_text)
                log(f"[paper {id_paper}] model='{model_name}' field='{field}'")
                resp = send_prompt(
                    prompt,
                    client=client,
                    model=args.model,
                    timeout=args.timeout,
                    thinking_budget=args.thinking_budget,
                    label=f"{field} id_paper={id_paper} model={model_name}",
                )
                try:
                    model_entry[field] = ast.literal_eval(resp)
                except Exception:
                    model_entry[field] = None
            paper_entry["models"].append(model_entry)

        papers.append(paper_entry)
        processed += 1

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Saved results to {out_path}")
    
    log(f"Processed {processed} papers")


if __name__ == "__main__":
    main()
