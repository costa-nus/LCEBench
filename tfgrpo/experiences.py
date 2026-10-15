"""Teacher side of Training-Free GRPO: update the experience library once per epoch.

1. summarise every rollout of the groups whose mean reward is strictly between 0 and 1,
2. critique each group and propose add / modify operations,
3. merge all proposals into the library with one batch-update call.
"""

import asyncio
import copy
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone

import openai
from transformers import AutoTokenizer

from .prompts import BATCH_EXPERIENCE_UPDATE_TEMPLATE

# Prompts are budgeted with the Qwen tokenizer against the vLLM context window (--max-model-len).
TOKENIZER = AutoTokenizer.from_pretrained(os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-8B"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "40960"))
_CTX_MARGIN = 256
QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def served_model_name(model: str) -> str:
    """vLLM must serve the model as ``Qwen/<model>`` (e.g. ``--served-model-name Qwen/Qwen3-8B``)."""
    return model if model.startswith("Qwen/") else f"Qwen/{model}"


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def truncate_prompt(prompt: str, max_output_tokens: int = 16384) -> str:
    # The output budget takes at most half of the window so a valid prompt budget always remains.
    out = min(max_output_tokens, MAX_MODEL_LEN // 2)
    budget = max(MAX_MODEL_LEN - out - _CTX_MARGIN, 256)
    ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    if len(ids) > budget:
        print(f"Prompt is too long ({len(ids)} > {budget}). Truncating.")
        return TOKENIZER.decode(ids[:budget], skip_special_tokens=True)
    return prompt


# ---------------------------------------------------------------- string-experience guard

class ExperienceTypeError(TypeError):
    """A generated experience has a non-string value."""


def only_string_experiences(experiences):
    """Return only entries whose experience value is a string."""
    if not isinstance(experiences, dict):
        return {}
    return {key: value for key, value in experiences.items() if isinstance(value, str)}


def format_string_experiences(experiences, empty: str = "") -> str:
    """Format string-valued experiences as "[ID]. text" lines for insertion into a prompt."""
    filtered = only_string_experiences(experiences)
    if not filtered:
        return empty
    return "\n".join(f"[{key}]. {value}" for key, value in filtered.items())


def validate_generated_experience_operations(operations, *, allowed_options, stage):
    """Reject generated operations whose ``experience`` field is not a string."""
    if not isinstance(operations, list):
        raise ExperienceTypeError(f"{stage}: operations must be a list, got {type(operations).__name__}")

    allowed = set(allowed_options)
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            raise ExperienceTypeError(
                f"{stage}: operation[{index}] must be a dict, got {type(operation).__name__}"
            )
        option = operation.get("option")
        if option in allowed or "experience" in operation:
            experience = operation.get("experience")
        else:
            continue
        if not isinstance(experience, str):
            raise ExperienceTypeError(
                f"{stage}: operation[{index}] option={option!r} has "
                f"non-string experience type={type(experience).__name__}"
            )
    return operations


def append_experience_type_error(save_dir, *, stage, error, response=None):
    """Persist one invalid generated-experience event."""
    os.makedirs(save_dir, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "stage": stage,
        "error": str(error),
        "response": response,
    }
    with open(os.path.join(save_dir, "experience_type_errors.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=repr) + "\n")


# ---------------------------------------------------------------- updater

class OpenAIExperienceUpdater:
    """Async experience updater: summaries and critiques run in parallel, the batch update is serial.

    ``teacher_fields(rollout)`` returns the task-specific template fields (ground truth, gold
    supporting facts, or checker feedback) for the summary/critique templates.
    """

    def __init__(self, client, model_name, summary_template, critique_template, teacher_fields, max_concurrency=6):
        self.client = client
        self.model_name = served_model_name(model_name)
        self.summary_template = summary_template
        self.critique_template = critique_template
        self.teacher_fields = teacher_fields
        self.sem = asyncio.Semaphore(max_concurrency)

    async def _generate_one(self, messages, max_tokens=16384, temperature=0.7, max_retries=5) -> str:
        last_exc = None

        # Output tokens must fit next to the prompt in the model context.
        prompt_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        allowed_out = max(MAX_MODEL_LEN - prompt_tokens - _CTX_MARGIN, 256)
        max_tokens = min(max_tokens, allowed_out)

        for i in range(max_retries):
            try:
                async with self.sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_body=QWEN_EXTRA_BODY,
                        seed=getattr(self.client, "sampling_seed", None),
                    )
                return resp.choices[0].message.content

            except openai.BadRequestError as e:
                msg = str(e)
                safety_margin = 32
                if "maximum context length" in msg:
                    match = re.search(r"request has (\d+) input tokens", msg)
                    if match:
                        allowed = MAX_MODEL_LEN - int(match.group(1)) - safety_margin
                        if allowed <= 0:
                            raise RuntimeError("Prompt too long")
                        print(f"[AutoFix] Adjust max_tokens: {max_tokens} → {allowed}")
                        max_tokens = allowed
                        continue
                last_exc = e

            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.4 * (2 ** i) + random.random() * 0.2)
        raise RuntimeError("Qwen API failed") from last_exc

    async def _generate_many(self, prompts, max_tokens=16384):
        tasks = [
            self._generate_one(
                messages=[{"role": "user", "content": truncate_prompt(p, max_output_tokens=max_tokens)}],
                max_tokens=max_tokens,
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    async def run(self, rollouts, experiences, save_dir, only_partial_correct=True):
        print("Start Async OpenAI Experience Updater...")
        experiences = only_string_experiences(experiences)

        problem_to_summarized_rollouts = await self._single_rollout_summary(rollouts, save_dir, only_partial_correct)
        critiques = await self._single_query_critique(
            problem_to_summarized_rollouts, experiences, save_dir, only_partial_correct
        )
        new_experiences = await self._batch_update(experiences, critiques, save_dir)

        return {f"G{i}": exp for i, exp in enumerate(only_string_experiences(new_experiences).values())}

    # Step 1: rollout summaries (parallel)
    async def _single_rollout_summary(self, rollouts, save_dir, only_partial_correct=True):
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename):
            with open(filename) as f:
                return json.load(f)

        problem_to_rollouts = defaultdict(list)
        for r in rollouts:
            problem_to_rollouts[r["problem"]].append(r)

        filtered = []
        for rs in problem_to_rollouts.values():
            avg = sum(r["reward"] for r in rs) / len(rs)
            if 0 < avg < 1 or not only_partial_correct:
                filtered.extend(rs)

        if not filtered:
            print("[INFO] No partial-correct rollouts.")
            return {}

        prompts = [
            self.summary_template.format(
                trajectory=r["trajectories"][0]["trajectory"],
                grade="This trajectory delivers **" + ("correct" if r["reward"] else "wrong") + "** answer",
                **self.teacher_fields(r),
            )
            for r in filtered
        ]

        print(f"[Summary] Generating {len(prompts)} summaries (parallel)")
        responses = await self._generate_many(prompts)

        results = defaultdict(list)
        for resp, r in zip(responses, filtered):
            results[r["problem"]].append({"trajectory_summary": resp, **r})

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Step 2: group critiques (parallel over problems)
    async def _single_query_critique(self, problem_to_summarized_rollouts, experiences, save_dir, only_partial_correct=True):
        max_operations = True

        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename):
            with open(filename) as f:
                return json.load(f)

        prompts = []
        rollout_groups = []
        for rollouts in problem_to_summarized_rollouts.values():
            avg = sum(r["reward"] for r in rollouts) / len(rollouts)
            if only_partial_correct and not (0 < avg < 1):
                continue

            formatted_traj = "\n\n".join(
                f"Trajectory {i+1} ({'correct' if r['reward'] else 'wrong'}):\n{r['trajectory_summary']}"
                for i, r in enumerate(rollouts)
            )
            prompts.append(
                self.critique_template.format(
                    max_operations=max_operations,
                    problem=rollouts[0]["problem"],
                    trajectories=formatted_traj,
                    experiences=format_string_experiences(experiences, empty="None"),
                    **self.teacher_fields(rollouts[0]),
                )
            )
            rollout_groups.append(rollouts)

        print(f"[Critique] Generating {len(prompts)} critiques (parallel)")
        responses = await self._generate_many(prompts)

        results = []
        for resp, group in zip(responses, rollout_groups):
            try:
                json_str = resp.split("```json")[-1].split("```")[0].strip()
                ops = json.loads(json_str)
                validate_generated_experience_operations(ops, allowed_options=("modify", "add"), stage="single")
                results.append({"rollouts": group, "critique": resp, "operations": ops[:max_operations]})
            except ExperienceTypeError as e:
                append_experience_type_error(save_dir, stage="single", error=e, response=resp)
                print(f"[ERROR] Single experience generation failed: {e}")
            except Exception as e:
                print(f"[WARN] Critique JSON parse failed: {e}")

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Step 3: batch update (serial)
    async def _batch_update(self, experiences, critiques, save_dir, max_retries=3):
        filename = os.path.join(save_dir, "batch_update.json")
        if os.path.exists(filename):
            with open(filename) as f:
                return only_string_experiences(json.load(f)["new_experiences"])

        candidate = copy.deepcopy(only_string_experiences(experiences))
        to_modify = []
        max_ID = 0
        for c in critiques:
            for op in c.get("operations", []):
                try:
                    if op["option"] == "modify" and op["modified_from"] in candidate:
                        to_modify.append(op)
                    elif op["option"] == "add":
                        candidate[f"C{max_ID}"] = op["experience"]
                        max_ID += 1
                except Exception:
                    pass

        update_prompt = BATCH_EXPERIENCE_UPDATE_TEMPLATE.format(experiences=candidate, updates=to_modify)

        for _ in range(max_retries):
            try:
                resp = await self._generate_one(messages=[{"role": "user", "content": update_prompt}], max_tokens=16384)
                json_str = resp.split("```json")[-1].split("```")[0].strip()
                revision_plan = json.loads(json_str)
                validate_generated_experience_operations(revision_plan, allowed_options=("modify", "merge"), stage="merge")
                break
            except ExperienceTypeError as e:
                append_experience_type_error(save_dir, stage="merge", error=e, response=locals().get("resp"))
                print(f"[ERROR] Merge experience generation failed: {e}")
                revision_plan = None
            except Exception as e:
                print(f"[WARN] Merge JSON parse failed: {e}")
                revision_plan = None

        new_exps = copy.deepcopy(candidate)
        if revision_plan:
            for op in revision_plan:
                try:
                    if op["option"] == "modify":
                        new_exps[op["modified_from"]] = op["experience"]
                    elif op["option"] == "merge":
                        for mid in op["merged_from"]:
                            new_exps.pop(mid, None)
                        new_exps[f"C{max_ID}"] = op["experience"]
                        max_ID += 1
                except Exception:
                    pass

        new_exps = only_string_experiences(new_exps)
        with open(filename, "w") as f:
            json.dump({"new_experiences": new_exps}, f, indent=2)
        return new_exps
