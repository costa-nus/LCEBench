"""Training-Free GRPO (Qwen via vLLM): learn an experience library for one task and one seed.

    python -m tfgrpo.run --task math500 --student_lm Qwen3-8B --teacher_lm Qwen3-8B --seed 0

Per epoch: the student samples ``train_k`` rollouts per training question, the teacher updates the
experience library from groups with mixed rewards, and the student is evaluated on the validation
split; the best-on-validation library is finally evaluated on the test split.
"""

import argparse
import asyncio
import json
import os
import random
import warnings
from typing import Dict, Optional

import numpy as np
import torch
from openai import AsyncOpenAI
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .experiences import (
    MAX_MODEL_LEN,
    QWEN_EXTRA_BODY,
    TOKENIZER,
    OpenAIExperienceUpdater,
    format_string_experiences,
    served_model_name,
)
from .tasks import TASKS

warnings.filterwarnings("ignore", message="Logical operators")

_CTX_MARGIN = 256


def set_seed(seed):
    # The train DataLoader shuffles with the global torch RNG, so the seed fixes the question order.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return {key: [sample[key] for sample in batch] for key in batch[0]}


def build_async_client(api_base: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=api_base, api_key="EMPTY")


async def generate_one_async(client, sem, model_name, user_prompt, max_tokens=16384, temperature=0.7,
                             max_retries=5, base_sleep=0.4) -> str:
    last_exc: Optional[Exception] = None

    # Reserve ``max_tokens`` for the answer and truncate the prompt with the Qwen tokenizer.
    budget = MAX_MODEL_LEN - max_tokens - _CTX_MARGIN
    ids = TOKENIZER.encode(user_prompt, add_special_tokens=False)
    if len(ids) > budget:
        print(f"Prompt is too long ({len(ids)} > {budget}). Truncating.")
        user_prompt = TOKENIZER.decode(ids[:budget], skip_special_tokens=True)

    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=served_model_name(model_name),
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=QWEN_EXTRA_BODY,
                    seed=getattr(client, "sampling_seed", None),
                )
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            await asyncio.sleep(base_sleep * (2 ** attempt) + random.uniform(0, 0.25))

    raise RuntimeError(f"Qwen API failed after {max_retries} retries") from last_exc


def student_prompt(task, sample, experiences):
    return task.student_template.format(experiences=format_string_experiences(experiences), **task.student_fields(sample))


async def evaluate_one_sample_async(client, sem, model_name, task, sample, K, temperature, experiences, max_tokens):
    prompt = student_prompt(task, sample, experiences)
    responses = await asyncio.gather(*[
        generate_one_async(client, sem, model_name, prompt, max_tokens=max_tokens, temperature=temperature)
        for _ in range(K)
    ])
    if any(not isinstance(r, str) for r in responses):
        print(f"[bad response] {[type(r).__name__ for r in responses]}")
        return 0.0
    scores = [task.score(r, sample) for r in responses]
    return sum(scores) / len(scores) if scores else 0.0


async def evaluate_mean_at_k_async(client, model_name, task, dataset, K, temperature, experiences=None,
                                   max_tokens=16384, limit=None, max_concurrency=8) -> float:
    sem = asyncio.Semaphore(max_concurrency)
    total = min(len(dataset), limit) if limit is not None else len(dataset)
    tasks = [
        evaluate_one_sample_async(client, sem, model_name, task, dataset[i], K, temperature, experiences, max_tokens)
        for i in range(total)
    ]

    scores = []
    pbar = tqdm(total=len(tasks), desc=f"Evaluating Mean@{K}")
    for coro in asyncio.as_completed(tasks):
        scores.append(await coro)
        pbar.set_postfix(mean_acc=f"{sum(scores)/len(scores):.3f}")
        pbar.update(1)
    pbar.close()
    return sum(scores) / len(scores) if scores else 0.0


async def collect_rollouts_for_one_sample_async(client, sem, model_name, task, sample, experiences, train_k,
                                                temperature, max_tokens):
    user_prompt = student_prompt(task, sample, experiences)
    responses = await asyncio.gather(*[
        generate_one_async(client, sem, model_name, user_prompt, max_tokens=max_tokens, temperature=temperature)
        for _ in range(train_k)
    ])
    rollouts = []
    for resp in responses:
        if not isinstance(resp, str):
            print(f"[BAD RESP] type={type(resp)}, value={repr(resp)[:200]}")
            resp = ""
        rollouts.append(task.rollout(sample, resp, user_prompt))
    return rollouts


async def collect_rollouts_async(client, model_name, task, train_loader, experiences, train_k, temperature,
                                 max_tokens, train_limit_per_epoch, max_concurrency):
    sem = asyncio.Semaphore(max_concurrency)

    sample_tasks = []
    for count, batch in enumerate(train_loader):
        if count >= train_limit_per_epoch:
            break
        sample = {key: value[0] for key, value in batch.items()}
        sample_tasks.append(
            collect_rollouts_for_one_sample_async(
                client, sem, model_name, task, sample, experiences, train_k, temperature, max_tokens
            )
        )

    all_rollouts = []
    pbar = tqdm(total=len(sample_tasks), desc="Training (collect rollouts)")
    for coro in asyncio.as_completed(sample_tasks):
        all_rollouts.extend(await coro)
        pbar.update(1)
    pbar.close()
    return all_rollouts


def _model_tag(name: str) -> str:
    return os.path.basename(name.rstrip("/"))


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


async def run_one_seed(args):
    task = TASKS[args.task]
    seed = args.seed
    set_seed(seed)

    student_client = build_async_client(args.student_api_base)
    teacher_client = build_async_client(args.teacher_api_base)
    student_client.sampling_seed = teacher_client.sampling_seed = seed

    save_root = os.path.join(
        args.output_dir, args.task, f"T-{_model_tag(args.teacher_lm)}_S-{_model_tag(args.student_lm)}", f"seed-{seed}"
    )
    os.makedirs(save_root, exist_ok=True)
    print(f"[seed={seed}] TF-GRPO on {task.tag}: student={args.student_lm} teacher={args.teacher_lm} -> {save_root}")

    train_dataset = ListDataset(task.load("train"))
    val_dataset = ListDataset(task.load("val"))
    test_dataset = ListDataset(task.load("test"))
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True, collate_fn=collate_fn)

    agent_experiences: Dict[str, str] = {}
    best_experiences: Dict[str, str] = {}
    best_val_acc = -1.0
    best_epoch = -1

    updater = OpenAIExperienceUpdater(
        client=teacher_client,
        model_name=args.teacher_lm,
        summary_template=task.summary_template,
        critique_template=task.critique_template,
        teacher_fields=task.teacher_fields,
    )

    eval_kwargs = dict(K=args.eval_k, temperature=args.test_temperature, max_tokens=args.max_tokens,
                       max_concurrency=args.max_concurrency)

    baseline_acc = None
    if args.eval_baseline:
        print(f"\n[seed={seed}] ==================== BASELINE TEST (no experiences) ====================")
        baseline_acc = await evaluate_mean_at_k_async(
            student_client, args.student_lm, task, test_dataset, experiences={}, limit=args.eval_limit, **eval_kwargs
        )
        print(f"[seed={seed}] [BASELINE] Mean@{args.eval_k}: {baseline_acc:.2%}")
        _dump({"baseline_acc": baseline_acc}, os.path.join(save_root, "baseline_test_metrics.json"))

    for epoch in range(args.num_epochs):
        save_dir = os.path.join(save_root, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[seed={seed}] ==================== EPOCH {epoch+1}/{args.num_epochs} ====================")

        # A. student rollouts
        all_rollouts = await collect_rollouts_async(
            student_client, args.student_lm, task, train_loader, agent_experiences, args.train_k,
            args.train_temperature, args.max_tokens, args.train_limit_per_epoch, args.max_concurrency,
        )
        _dump(all_rollouts, os.path.join(save_dir, f"rollouts_epoch_{epoch+1}.json"))

        # B. teacher updates the experience library
        agent_experiences = await updater.run(
            rollouts=all_rollouts,
            experiences=agent_experiences,
            save_dir=save_dir,
            only_partial_correct=args.only_partial_correct,
        )

        # C. validation
        val_acc = await evaluate_mean_at_k_async(
            student_client, args.student_lm, task, val_dataset, experiences=agent_experiences, limit=args.val_limit,
            **eval_kwargs,
        )
        print(f"[seed={seed}] [VAL][EPOCH {epoch+1}] Mean@{args.eval_k}: {val_acc:.2%}")
        _dump({"epoch": epoch + 1, "val_acc": val_acc}, os.path.join(save_dir, "val_metrics.json"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_experiences = dict(agent_experiences)
            _dump(best_experiences, os.path.join(save_root, "best_experiences.json"))
            _dump({"best_epoch": best_epoch, "best_val_acc": best_val_acc}, os.path.join(save_root, "best_val_info.json"))
            print(f"[seed={seed}] [BEST UPDATE] epoch={best_epoch}, val={best_val_acc:.2%}")

    # Test with the best-on-validation experiences
    final_experiences = best_experiences if best_experiences else agent_experiences
    post_acc = await evaluate_mean_at_k_async(
        student_client, args.student_lm, task, test_dataset, experiences=final_experiences, limit=args.eval_limit,
        **eval_kwargs,
    )
    print(f"[seed={seed}] [TEST] Mean@{args.eval_k}: {post_acc:.2%}")

    result = {"seed": seed}
    if baseline_acc is not None:
        result["baseline_acc"] = baseline_acc
    result.update({"best_val_acc": best_val_acc, "best_epoch": best_epoch, "post_acc": post_acc})
    _dump(result, os.path.join(save_root, "summary.json"))
    print(f"[seed={seed}] best val {best_val_acc:.2%} (epoch {best_epoch}) | test {post_acc:.2%}")
    return result


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def parse_args():
    default_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--student_lm", default="Qwen3-8B", help="model that solves the task (vLLM name Qwen/<model>)")
    p.add_argument("--teacher_lm", default="Qwen3-8B", help="model that updates the experiences")
    p.add_argument("--student_api_base", default=default_base)
    p.add_argument("--teacher_api_base", default=default_base)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="outputs/tfgrpo")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--train_limit_per_epoch", type=int, default=1000)
    p.add_argument("--train_k", type=int, default=5, help="group size: rollouts per training question")
    p.add_argument("--eval_k", type=int, default=1, help="samples per question for Mean@K evaluation")
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--train_temperature", type=float, default=0.7)
    p.add_argument("--test_temperature", type=float, default=0.7)
    p.add_argument("--max_concurrency", type=int, default=32, help="concurrent student requests")
    p.add_argument("--val_limit", type=int, default=None)
    p.add_argument("--eval_limit", type=int, default=None)
    p.add_argument("--only_partial_correct", type=str2bool, default=True,
                   help="learn only from groups whose mean reward is strictly between 0 and 1")
    p.add_argument("--eval_baseline", action="store_true", help="also evaluate the empty library on test first")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_one_seed(parse_args()))
