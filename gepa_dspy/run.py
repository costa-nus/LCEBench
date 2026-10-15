"""GEPA (DSPy, auto="heavy") with Qwen models served by vLLM, one task and one seed.

    python -m gepa_dspy.run --task math500 --student_lm Qwen3-8B --teacher_lm Qwen3-8B --seed 0

The student model runs the program; the teacher model is GEPA's reflection LM. The program is
evaluated on train/val/test before and after optimization (val is GEPA's validation set).
"""

import argparse
import json
import os
import random
import time

import dspy
import numpy as np
import torch
from dspy import GEPA, Evaluate

from .tasks import TASKS

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_lm(model_name: str, api_base: str, seed: int = 0):
    """Qwen model behind a vLLM OpenAI-compatible server (served as ``Qwen/<model>``)."""
    served = model_name if model_name.startswith("Qwen/") else f"Qwen/{model_name}"
    return dspy.LM(
        f"openai/{served}",
        api_base=api_base,
        api_key="EMPTY",
        model_type="chat",
        max_tokens=16384,
        temperature=0.7,
        cache=False,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        seed=seed,
    )


def instructions_of(program):
    return program.named_predictors()[0][1].signature.instructions


def run(args):
    examples, program_cls, metric, training_metric = TASKS[args.task]
    set_seed(args.seed)

    lm = build_lm(args.student_lm, args.student_api_base, seed=args.seed)
    reflection_lm = build_lm(args.teacher_lm, args.teacher_api_base, seed=args.seed)
    dspy.settings.configure(lm=lm)

    trainset, devset, testset = examples("train"), examples("val"), examples("test")
    model = program_cls()
    print(instructions_of(model))

    def evaluate(program):
        return [
            float(Evaluate(devset=data, num_threads=args.num_threads, provide_traceback=True)(
                program, metric=metric, display_progress=True))
            for data in (trainset, devset, testset)
        ]

    initial_train, initial_dev, initial_test = evaluate(model)

    optimizer = GEPA(
        metric=training_metric,
        auto="heavy",
        num_threads=args.num_threads,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=reflection_lm,
    )
    optimized_model = optimizer.compile(student=model, trainset=trainset, valset=devset)

    final_train, final_dev, final_test = evaluate(optimized_model)

    try:
        instructions = instructions_of(optimized_model)
    except Exception:
        instructions = "N/A"

    return {
        "task": args.task,
        "seed": args.seed,
        "teacher_lm": args.teacher_lm,
        "student_lm": args.student_lm,
        "initial_train_score": initial_train,
        "initial_dev_score": initial_dev,
        "initial_test_score": initial_test,
        "final_train_score": final_train,
        "final_dev_score": final_dev,
        "final_test_score": final_test,
        "instructions": instructions,
    }


def _model_tag(name: str) -> str:
    return os.path.basename(name.rstrip("/"))


def parse_args():
    default_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--student_lm", default="Qwen3-8B", help="model that runs the program (vLLM name Qwen/<model>)")
    p.add_argument("--teacher_lm", default="Qwen3-8B", help="GEPA reflection model")
    p.add_argument("--student_api_base", default=default_base)
    p.add_argument("--teacher_api_base", default=default_base)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_threads", type=int, default=32)
    p.add_argument("--output_dir", default="outputs/gepa")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(args.output_dir, args.task, f"T-{_model_tag(args.teacher_lm)}_S-{_model_tag(args.student_lm)}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"seed-{args.seed}.json")
    print(f"GEPA on {args.task} | seed={args.seed} | teacher={args.teacher_lm} student={args.student_lm} -> {out_file}")

    start = time.time()
    result = run(args)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"initial_test_score: {result['initial_test_score']:.4f}")
    print(f"final_test_score:   {result['final_test_score']:.4f}")
    print(f"Elapsed (sec):      {time.time() - start:.2f}")


if __name__ == "__main__":
    main()
