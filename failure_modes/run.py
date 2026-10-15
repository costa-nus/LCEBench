"""Extract one failure-mechanism sentence per evaluated error trajectory."""

import argparse
import json
import math
import os
from pathlib import Path

from .prompts import build_label_prompt

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record["dataset"] == "appworld":
                    history = record.get("history_messages")
                    if not isinstance(history, list) or not history:
                        raise ValueError("AppWorld requires nonempty history_messages")
                    for field in ("task_context", "ground_truth_code", "test_report", "feedback"):
                        if field not in record or record[field] is None:
                            raise ValueError(f"AppWorld requires {field}")
                    build_label_prompt(record)
                else:
                    for field in ("question", "raw_output"):
                        if not isinstance(record.get(field), str) or not record[field].strip():
                            raise ValueError(f"{field} must be a nonempty string")
                    if "extracted_prediction" not in record:
                        raise ValueError("missing extracted_prediction")
                score = record.get("score")
                if (
                    not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or score < 0
                    or score >= 1
                ):
                    raise ValueError("score must be a finite error score in [0, 1)")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no error trajectories")
    return records


def parse_response(text):
    """Parse the JSON envelope without enforcing semantic label constraints."""
    if not isinstance(text, str):
        raise ValueError("the extractor returned no text")
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {"failure_mechanism"}:
        raise ValueError("expected a JSON object with only failure_mechanism")
    mechanism = result["failure_mechanism"]
    if not isinstance(mechanism, str) or not mechanism.strip():
        raise ValueError("failure_mechanism must be a nonempty string")
    return mechanism.strip()


def extract(client, model, messages, max_tokens, retries):
    last_error = None
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                seed=0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return parse_response(response.choices[0].message.content)
        except Exception as error:
            last_error = error
    raise RuntimeError(f"failure extraction failed after {retries} attempts") from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("example.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--retries", type=int, default=10)
    args = parser.parse_args()
    if args.retries < 1 or args.max_tokens < 1:
        parser.error("--retries and --max-tokens must be positive")
    output = args.output or Path("failure_mechanisms.jsonl")
    if args.input.resolve() == output.resolve():
        parser.error("--output must not overwrite --input")
    records = load_records(args.input)

    from openai import OpenAI

    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
    )
    try:
        with output.open("x", encoding="utf-8") as destination:
            for record in records:
                messages = [{"role": "user", "content": build_label_prompt(record)}]
                result = {field: record[field] for field in ("method", "model", "dataset")}
                result["failure_mechanism"] = extract(
                    client, args.model, messages, args.max_tokens, args.retries
                )
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
    finally:
        client.close()
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
