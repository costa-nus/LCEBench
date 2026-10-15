# Revisiting LLMs as Context Engineers

This repository provides the implementation of Revisiting LLMs as Context Engineers.

## Contributions

- **LLMs can** engineer contexts via task-specific training, but no single method can achieve the best performance in all datasets at test time.

- **Scaling** helps but cannot reverse existing failures of LCEs. Even scaling to newer generations and more parameters remains insufficient to make LCEs more effective across different settings.

- **When & How:** Through aggregate analysis, we find that LCEs work better when facing less diverse failure modes of reasoning. Case studies show that the ability to abstract principled contexts from heterogeneous failures is the major bottleneck and that the success of some LCE methods is attributed to the improvement in adherence to (format) instruction rather than reasoning.


## Setup

```
benchmarks/    datasets + scoring shared by the three methods (IFBench checkers, MATH verifier, ...)
data/          train / val / test splits
gepa_dspy/     run.py (optimization loop), tasks.py (DSPy programs and feedback metrics); not named
               "gepa" so that it does not shadow the gepa package used by DSPy
ace/           run.py, ace.py (training loop), generator / reflector / curator, bullet de-duplication
tfgrpo/        run.py (rollouts, validation, test), experiences.py (teacher updates), prompts.py, tasks.py
failure_modes/ run.py (extracts the failure mode of each failed trajectory), prompts.py, example.jsonl
scripts/       serve_qwen.sh, run.sh (4 tasks x 5 seeds)
```

| Method | Context | Entry point |
|---|---|---|
| [GEPA](https://arxiv.org/abs/2507.19457) (DSPy, `auto="heavy"`) | the instruction of the task program | `python -m gepa_dspy.run` |
| [ACE](https://arxiv.org/abs/2510.04618) | a playbook of bullet points (generator / reflector / curator) | `python -m ace.run` |
| [Training-Free GRPO](https://arxiv.org/abs/2510.08191) | an experience library learned from groups of rollouts | `python -m tfgrpo.run` |

Benchmarks: **HotpotQA**, **IFBench**, **MATH500**, **MiddleMatch** (`--task hotpotqa|ifbench|math500|middlematch`).


```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm                     # IFBench checkers
python -m nltk.downloader punkt punkt_tab stopwords averaged_perceptron_tagger_eng
```


## Training

All 4 tasks x 5 seeds (seeds run as parallel processes, logs in `logs/<method>/`):

```bash
MODEL=Qwen/Qwen3-8B PORT=8000 bash scripts/serve_qwen.sh
MODEL=/path/to/Qwen3-8B SERVED_NAME=Qwen/Qwen3-8B PORT=8000 bash scripts/serve_qwen.sh   # local weights
```


```bash
bash scripts/run.sh gepa
bash scripts/run.sh ace
bash scripts/run.sh tfgrpo
TASKS="math500" SEEDS="0 1" bash scripts/run.sh tfgrpo     # subset
```

A single run:

```bash
python -m gepa_dspy.run --task hotpotqa --student_lm Qwen3-8B --teacher_lm Qwen3-8B --seed 0
python -m ace.run       --task ifbench  --student_lm Qwen3-8B --teacher_lm Qwen3-8B --seed 0
python -m tfgrpo.run    --task math500  --student_lm Qwen3-8B --teacher_lm Qwen3-8B --seed 0
```

The *student* model solves the task; the *teacher* model performs the optimization (GEPA reflection LM, ACE
reflector + curator, TF-GRPO experience updates). They can be different Qwen models on different servers:
`--student_api_base http://host:8000/v1 --teacher_api_base http://host:8001/v1`
(`STUDENT=Qwen3-8B TEACHER=Qwen3-32B STUDENT_API_BASE=... TEACHER_API_BASE=... bash scripts/run.sh ...`).


## Outputs

`outputs/<method>/<task>/T-<teacher>_S-<student>/`:

- GEPA: `seed-<N>.json` (initial / final train, val, test scores and the optimized instruction)
- ACE: `seed-<N>/ace_run_<time>_<task>_offline/` (`final_results.json`, `best_playbook.txt`, `intermediate_playbooks/`, LLM call logs)
- TF-GRPO: `seed-<N>/` (`summary.json`, `best_experiences.json`, per-epoch rollouts, summaries, critiques, updates, validation scores)


## Failure Mode Extraction

`failure_modes/` uses an LLM extractor (default `Qwen/Qwen3-8B` behind vLLM, temperature 0, thinking disabled)
to extract one sentence describing the reusable failure mechanism of each evaluated error trajectory:

```bash
VLLM_BASE_URL=http://localhost:8000/v1 python -m failure_modes.run \
    --input failure_modes/example.jsonl --output failure_mechanisms.jsonl
```

## License

This project is released under the [MIT License](LICENSE). Third-party code retains its original licenses and copyright notices.
