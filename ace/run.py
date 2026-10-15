"""ACE offline adaptation with Qwen models served by vLLM, one task and one seed.

    python -m ace.run --task math500 --student_lm Qwen3-8B --teacher_lm Qwen3-8B --seed 0

The student model is the generator; the teacher model is the reflector and the curator (and
merges duplicate bullets). The playbook with the best validation accuracy is tested at the end.
"""

import argparse
import os
import random

import numpy as np
import torch

from .ace import ACE
from .tasks import TASKS
from .utils import initialize_clients


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _served_model_name(name: str) -> str:
    """vLLM must serve the model as ``Qwen/<model>`` (--served-model-name)."""
    return name if name.startswith("Qwen/") else f"Qwen/{name}"


def _model_tag(name: str) -> str:
    return os.path.basename(name.rstrip("/"))


def parse_args():
    default_base = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--student_lm", default="Qwen3-8B", help="generator model (vLLM name Qwen/<model>)")
    p.add_argument("--teacher_lm", default="Qwen3-8B", help="reflector / curator model")
    p.add_argument("--student_api_base", default=default_base)
    p.add_argument("--teacher_api_base", default=default_base)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="outputs/ace")
    p.add_argument("--test_workers", type=int, default=10, help="parallel requests for validation / test")
    p.add_argument("--initial_playbook_path", default=None)
    p.add_argument("--disable_bulletpoint_analyzer", action="store_true",
                   help="disable playbook bullet de-duplication / merging (enabled in the paper runs)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    task = TASKS[args.task]

    teacher_lm = _served_model_name(args.teacher_lm)
    student_lm = _served_model_name(args.student_lm)
    save_dir = os.path.join(args.output_dir, args.task, f"T-{_model_tag(teacher_lm)}_S-{_model_tag(student_lm)}",
                            f"seed-{args.seed}")
    print(f"[seed={args.seed}] save_dir={save_dir}")

    train_samples, val_samples, test_samples = task.load("train"), task.load("val"), task.load("test")
    print(f"Offline mode: Training on {len(train_samples)} examples, "
          f"validating on {len(val_samples)}, testing on {len(test_samples)}")

    initial_playbook = None
    if args.initial_playbook_path and os.path.exists(args.initial_playbook_path):
        with open(args.initial_playbook_path, "r") as f:
            initial_playbook = f.read()
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    use_bulletpoint_analyzer = not args.disable_bulletpoint_analyzer
    generator_client, reflector_client, curator_client = initialize_clients(args.student_api_base, args.teacher_api_base)
    ace_system = ACE(
        task,
        generator_client,
        reflector_client,
        curator_client,
        generator_model=student_lm,
        reflector_model=teacher_lm,
        curator_model=teacher_lm,
        max_tokens=16384,
        temperature=0.7,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=0.90,
    )
    for client in (ace_system.generator_client, ace_system.reflector_client, ace_system.curator_client):
        client.sampling_seed = args.seed

    config = {
        'num_epochs': 3,
        'max_num_rounds': 3,
        'curator_frequency': 5,
        'eval_steps': task.eval_steps,
        'save_steps': task.save_steps,
        'playbook_token_budget': 10000,
        'task_name': task.name,
        'mode': 'offline',
        'json_mode': False,
        'no_ground_truth': task.no_ground_truth,
        'save_dir': save_dir,
        'test_workers': args.test_workers,
        'initial_playbook_path': args.initial_playbook_path,
        'use_bulletpoint_analyzer': use_bulletpoint_analyzer,  # playbook bullet de-duplication and merging
        'bulletpoint_analyzer_threshold': 0.90,
    }

    ace_system.run(train_samples=train_samples, val_samples=val_samples, test_samples=test_samples, config=config)


if __name__ == "__main__":
    main()
