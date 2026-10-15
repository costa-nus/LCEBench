"""
Shared helpers for ACE: API clients, answer extraction, Qwen-tokenizer prompt budgeting
(the playbook is truncated first) and parallel evaluation.
"""
import os
import re
import json
from typing import Callable, Dict, Optional, Protocol, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from transformers import AutoTokenizer


def initialize_clients(student_api_base: str, teacher_api_base: str):
    """Generator (student) client and reflector/curator (teacher) clients for vLLM servers."""
    generator_client = openai.OpenAI(api_key="EMPTY", base_url=student_api_base)
    reflector_client = openai.OpenAI(api_key="EMPTY", base_url=teacher_api_base)
    curator_client = openai.OpenAI(api_key="EMPTY", base_url=teacher_api_base)
    print(f"generator(student)={student_api_base}  reflector/curator(teacher)={teacher_api_base}")
    return generator_client, reflector_client, curator_client


def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta"
    }

    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")

    if clean_name in slug_map:
        return slug_map[clean_name]

    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])


def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None

    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start

    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None


def extract_answer(response):
    """Extract final answer from model response"""
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer

    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer

        # Try to get final answer from JSON style response with regex matching
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer

        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer

        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer

        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content

        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            if answer:
                return answer

        return "No final answer found"


# ---------------------------------------------------------------- prompt budgeting

_TOK = AutoTokenizer.from_pretrained(os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-8B"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "40960"))
_CTX_MARGIN = 256


def count_tokens(prompt: str) -> int:
    return len(_TOK.encode(prompt, add_special_tokens=False))


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False): ...

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str: ...


class PromptTooLongError(ValueError):
    """Raised when a valid input-token budget cannot be created."""


def _truncate_text_tail_to_budget(text: str, *, tokenizer: TokenizerLike, budget: int) -> str:
    """Remove tokens from the end of ``text`` until it fits ``budget``."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    low, high = 0, min(len(token_ids), budget)
    best_text = ""

    # Decode and re-count candidates because encode/decode is not guaranteed to
    # preserve the exact token count for every tokenizer.
    while low <= high:
        mid = (low + high) // 2
        candidate = tokenizer.decode(token_ids[:mid], skip_special_tokens=True)
        candidate_tokens = len(tokenizer.encode(candidate, add_special_tokens=False))

        if candidate_tokens <= budget:
            best_text = candidate
            low = mid + 1
        else:
            high = mid - 1

    return best_text


def truncate_playbook_prompt(
    prompt: str,
    *,
    tokenizer: TokenizerLike,
    max_model_len: int,
    max_output_tokens: int,
    context_margin: int,
    playbook: Optional[str] = None,
    prompt_builder: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Fit ``prompt`` by truncating the playbook passed to ``prompt_builder`` first.

    When playbook metadata is available, the playbook is truncated first. If
    the prompt is still too long after the playbook is completely removed, the
    remaining prompt is truncated from its end. When playbook metadata is not
    available, the complete prompt is truncated from its end as a fallback.
    """

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    budget = max_model_len - max_output_tokens - context_margin
    if budget <= 0:
        raise PromptTooLongError(
            "No input-token budget remains: "
            f"max_model_len={max_model_len}, "
            f"max_output_tokens={max_output_tokens}, "
            f"context_margin={context_margin}."
        )

    prompt_tokens = count(prompt)
    if prompt_tokens <= budget:
        return prompt

    if playbook is None or prompt_builder is None:
        fallback_prompt = _truncate_text_tail_to_budget(prompt, tokenizer=tokenizer, budget=budget)
        print(
            "Prompt is too long and no playbook metadata was provided "
            f"({prompt_tokens} > {budget}). Truncated the full prompt from "
            f"the end: {prompt_tokens} -> {count(fallback_prompt)} tokens."
        )
        return fallback_prompt

    if prompt_builder(playbook) != prompt:
        raise ValueError("prompt_builder(playbook) must exactly reproduce the original prompt.")

    prompt_without_playbook = prompt_builder("")
    protected_tokens = count(prompt_without_playbook)
    if protected_tokens > budget:
        fallback_prompt = _truncate_text_tail_to_budget(prompt_without_playbook, tokenizer=tokenizer, budget=budget)
        print(
            "Prompt remains too long after removing the entire playbook "
            f"({protected_tokens} > {budget}). Truncated the remaining prompt "
            f"from the end: {protected_tokens} -> {count(fallback_prompt)} tokens."
        )
        return fallback_prompt

    playbook_ids = tokenizer.encode(playbook, add_special_tokens=False)
    low, high = 0, len(playbook_ids)
    best_prompt = prompt_without_playbook
    best_playbook_token_count = 0

    # Find the longest playbook-token prefix whose fully rebuilt prompt fits.
    # Rebuilding on every check accounts for tokenizer merges at slot borders.
    while low <= high:
        mid = (low + high) // 2
        candidate_prompt = prompt_builder(tokenizer.decode(playbook_ids[:mid], skip_special_tokens=True))

        if count(candidate_prompt) <= budget:
            best_prompt = candidate_prompt
            best_playbook_token_count = mid
            low = mid + 1
        else:
            high = mid - 1

    print(
        "Prompt is too long "
        f"({prompt_tokens} > {budget}). Truncated playbook only: "
        f"{len(playbook_ids)} -> {best_playbook_token_count} playbook tokens; "
        f"final prompt={count(best_prompt)} tokens."
    )
    return best_prompt


def truncate_prompt(
    prompt: str,
    max_output_tokens: int = 16384,
    *,
    playbook: Optional[str] = None,
    prompt_builder: Optional[Callable[[str], str]] = None,
) -> str:
    """Fit a prompt into the model context, truncating its playbook first (see truncate_playbook_prompt)."""
    return truncate_playbook_prompt(
        prompt,
        tokenizer=_TOK,
        max_model_len=MAX_MODEL_LEN,
        max_output_tokens=max_output_tokens,
        context_margin=_CTX_MARGIN,
        playbook=playbook,
        prompt_builder=prompt_builder,
    )


# ---------------------------------------------------------------- evaluation

def evaluate_single_test_sample(args_tuple, task) -> Tuple[Dict, str]:
    """Generate an answer for one sample with the given playbook and score it."""
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        question = task_dict["question"]
        reference = task_dict[task.reference_key]

        gen_response, bullet_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir,
            **task.generator_kwargs(task_dict),
        )

        final_answer = extract_answer(gen_response)

        if task.fractional:
            is_correct, feedback = task.check(final_answer, task_dict)
            return {
                "index": i,
                "final_answer": final_answer,
                "feedback": feedback,
                "is_correct": is_correct,
                "success": True,
                "item": reference,
            }, None

        is_correct = task.score(final_answer, reference)
        return {
            "index": i,
            "final_answer": final_answer,
            "target": reference,
            "is_correct": is_correct,
            "success": True
        }, None

    except Exception as e:
        return None, f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}"


def evaluate_test_set(task, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20,
                      use_json_mode=False) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of a sample list with the given playbook.

    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    references_key = "items" if task.fractional else "targets"
    results = {
        "correct": 0, "total": 0, "no_answer": 0,
        "answers": [], references_key: [], "errors": []
    }

    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, task)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            result, error = future.result()

            if error:
                print(error)
                continue

            if result and result["success"]:
                if task.fractional:
                    # IFBench: score = fraction of instructions followed
                    results["correct"] += result["is_correct"]
                    results["total"] += 1
                    results["items"].append(result["item"])
                    results["answers"].append(result["final_answer"])
                    if result["is_correct"] != 1:
                        results["errors"].append({
                            "index": result["index"],
                            "prediction": result["final_answer"],
                            "feedback": result["feedback"],
                            "score": result["is_correct"]
                        })
                else:
                    results["correct"] += (1 if result["is_correct"] else 0)
                    results["total"] += 1
                    results["answers"].append(result["final_answer"])
                    results["targets"].append(result["target"])
                    if not result["is_correct"]:
                        results["errors"].append({
                            "index": result["index"],
                            "prediction": result["final_answer"],
                            "ground_truth": result["target"]
                        })

                if result["final_answer"] == "No final answer found":
                    results["no_answer"] += 1

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")

    if results["answers"] and results[references_key]:
        accuracy = task.evaluate_accuracy(results["answers"], results[references_key])

        final_results = {
            "accuracy": accuracy,
            "correct": results["correct"],
            "total": results["total"],
            "no_answer": results["no_answer"]
        }

        error_logs = {
            "accuracy": accuracy,
            "errors": results["errors"]
        }

        print(f"\n📊 Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")
    else:
        final_results = {"accuracy": 0.0, "correct": 0, "total": 0}
        error_logs = {}
        print(f"\n📊 No valid results!")

    return final_results, error_logs
