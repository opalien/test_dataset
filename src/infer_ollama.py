#!/usr/bin/env python3
import argparse
import ast
import json
import time
from pathlib import Path
from typing import Any

import dataset
from ollama import Client


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def send_prompt(
    prompt: str,
    *,
    client: Client,
    model: str,
    label: str,
    max_retries: int,
    backoff_seconds: float,
    timeout: float,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "top_p": 0.0, "num_predict": 4096},
            )
            if text := response.message.content:
                return text.strip()
            log(f"[{label}] empty response; retry {attempt}/{max_retries}")
        except Exception as err:
            if attempt == max_retries:
                raise RuntimeError(f"Ollama request failed after retries: {err}") from err
            sleep_time = backoff_seconds * attempt
            log(f"[{label}] error; retry {attempt}/{max_retries} in {sleep_time:.1f}s ({err})")
            time.sleep(sleep_time)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ollama extraction over papers in a dataset DB.")
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
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--backoff-seconds", type=float, default=5.0)
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

    client = Client(host=f"http://{args.host}:{args.port}")
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
            label=f"model_enumeration id_paper={id_paper}",
            max_retries=args.max_retries,
            backoff_seconds=args.backoff_seconds,
            timeout=args.timeout,
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
                    label=f"{field} id_paper={id_paper} model={model_name}",
                    max_retries=args.max_retries,
                    backoff_seconds=args.backoff_seconds,
                    timeout=args.timeout,
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
