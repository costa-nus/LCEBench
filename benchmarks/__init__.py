"""Datasets and scoring code shared by GEPA, ACE and TF-GRPO.

Splits (train / val / test):
    hotpotqa     150 / 300 / 300   data/hotpotqa/{train,val,test}.jsonl
    ifbench      150 / 300 / 294   data/ifbench/{train,val,test}.jsonl
    math500      100 / 200 / 200   data/math500/{train,val,test}.jsonl
    middlematch  100 / 100 / 300   data/middlematch/middle_match_dataset.json ([:100], [100:200], [200:])
"""

import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TASKS = ("hotpotqa", "ifbench", "math500", "middlematch")
SPLITS = ("train", "val", "test")


def extract_answer_tag(text, strip_fallback=True):
    """Return the content of the last <answer>...</answer>; without tags, the whole text
    (stripped if ``strip_fallback``). Non-string input gives ""."""
    if not isinstance(text, str):
        return ""
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if len(matches) < 1:
        return text.strip() if strip_fallback else text
    return matches[-1].group(1).strip()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_split(task, split):
    """Return the raw records of one split as a list of dicts."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    if task == "middlematch":
        with open(os.path.join(DATA_DIR, "middlematch", "middle_match_dataset.json"), "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        return {"train": data[:100], "val": data[100:200], "test": data[200:]}[split]
    return load_jsonl(os.path.join(DATA_DIR, task, f"{split}.jsonl"))
