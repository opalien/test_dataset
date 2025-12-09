#!/usr/bin/env python3
import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any

import dataset
from google import genai
from google.genai import types as genai_types


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)





def extract_text_response(resp: Any) -> str:
    if txt := getattr(resp, "text", None):
        return str(txt).strip()
    candidates = getattr(resp, "candidates", None) or []
    if candidates and isinstance(candidates, list):
        parts = getattr(candidates[0], "content", None)
        if parts and hasattr(parts, "parts"):
            texts = [
                part.text if hasattr(part, "text") else part.get("text", "")
                for part in parts.parts
            ]
            return "\n".join(t for t in texts if t).strip()
    return ""


def send_prompt(
    prompt: str,
    *,
    client: genai.Client,
    model: str,
    thinking_budget: int | None,
    label: str,
    max_retries: int,
    backoff_seconds: float,
    rate_limit_delay: float,
) -> str:
    config = genai_types.GenerateContentConfig(temperature=0.0, top_p=0.0, top_k=1)
    if thinking_budget is not None and "2.5" in model:
        config.thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

    for attempt in range(1, max_retries + 1):
        try:
            if rate_limit_delay > 0:
                time.sleep(rate_limit_delay)
            resp = client.models.generate_content(
                model=f"models/{model}",
                contents=prompt,
                config=config,
            )
            usage = getattr(resp, "usage_metadata", None)
            if thoughts := getattr(usage, "thoughts_token_count", None):
                print(f"Thoughts tokens: {thoughts}")
            if output := getattr(usage, "candidates_token_count", None):
                print(f"Output tokens: {output}")

            if text := extract_text_response(resp):
                return text
            log(f"[{label}] empty response; retry {attempt}/{max_retries}")
        except Exception as err:
            if attempt == max_retries:
                raise RuntimeError(f"Gemini request failed after retries: {err}") from err
            sleep_time = backoff_seconds * attempt
            log(f"[{label}] error; retry {attempt}/{max_retries} in {sleep_time:.1f}s ({err})")
            time.sleep(sleep_time)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini extraction over papers in a dataset DB.")
    parser.add_argument("--database", default="data/greenmir.db")
    parser.add_argument("--table", default="paper_text")
    parser.add_argument("--questions-dir", default="src/questions/estimated")
    parser.add_argument(
        "--fields",
        nargs="*",
        default=["parameters", "hardware", "h_power", "h_number", "h_compute", "country", "year", "co2eq", "training_time", "training_compute"],
    )
    parser.add_argument("--output", default="papers.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default="models/gemini-2.5-flash")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--backoff-seconds", type=float, default=10.0)
    parser.add_argument("--rate-limit-delay", type=float, default=0.0)
    args = parser.parse_args()

    qdir = Path(args.questions_dir)
    model_prompt_path = qdir / "model_enumeration.txt"
    if not model_prompt_path.exists():
        parser.error(f"Missing model enumeration prompt: {model_prompt_path}")

    questions: dict[str, str] = {"model": model_prompt_path.read_text(encoding="utf-8")}
    for field in args.fields:
        p = qdir / f"{field}.txt"
        if not p.exists():
            parser.error(f"Missing prompt for field '{field}': {p}")
        questions[field] = p.read_text(encoding="utf-8")

    db = dataset.connect(f"sqlite:///{args.database}" if "://" not in args.database else args.database)
    table = db[args.table]

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
        if arxiv_ids and id_paper not in arxiv_ids:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        article_text = row.get("text") or ""
        log(f"[paper {id_paper}] enumerate models")

        model_resp = send_prompt(
            questions["model"].replace("{article_text}", article_text),
            client=client,
            model=args.model,
            thinking_budget=args.thinking_budget,
            label=f"model_enumeration id_paper={id_paper}",
            max_retries=args.max_retries,
            backoff_seconds=args.backoff_seconds,
            rate_limit_delay=args.rate_limit_delay,
        )

        try:
            model_list = ast.literal_eval(model_resp)
            if not isinstance(model_list, list):
                raise ValueError("model enumeration did not return a list")
        except Exception as err:
            log(f"[paper {id_paper}] failed to parse model list: {err}")
            continue

        paper_entry: dict[str, Any] = {"id_paper": id_paper, "models": []}
        for model_name in model_list:
            model_entry: dict[str, Any] = {"model": model_name}
            for field in args.fields:
                prompt = questions[field].replace("{model_name}", str(model_name)).replace("{article_text}", article_text)
                log(f"[paper {id_paper}] model='{model_name}' field='{field}'")
                resp = send_prompt(
                    prompt,
                    client=client,
                    model=args.model,
                    thinking_budget=args.thinking_budget,
                    label=f"{field} id_paper={id_paper} model={model_name}",
                    max_retries=args.max_retries,
                    backoff_seconds=args.backoff_seconds,
                    rate_limit_delay=args.rate_limit_delay,
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
