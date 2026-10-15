"""Print the test score (%) of every method / task / teacher-student setting as mean ± std over seeds, the score
before optimization and the gain (after minus before).

    python scripts/summarize.py [outputs] [--csv gains.csv]

``--csv`` writes method,model,dataset,teacher,before,after,gain,seeds per setting, the gains used by
``python -m failure_modes.vendi --gains``. ``model`` is the student, whose rollouts the failure modes come from.
"""

import argparse
import csv
import glob
import json
import os
import statistics

# method -> (result files below <outputs>/<method>/<task>/T-<teacher>_S-<student>/,
#            test score after optimization in %, test score before optimization in % or None)
READERS = {
    "tfgrpo": (
        "seed-*/summary.json",
        lambda d: 100 * d["post_acc"],
        lambda d: 100 * d["baseline_acc"] if "baseline_acc" in d else None,  # absent with --eval_baseline false
    ),
    "gepa": (
        "seed-*.json",
        lambda d: d["final_test_score"],
        lambda d: d["initial_test_score"],
    ),
    "ace": (
        "seed-*/ace_run_*/final_results.json",
        lambda d: 100 * d["final_test_results"]["accuracy"],
        lambda d: 100 * d["initial_test_results"]["accuracy"],
    ),
}
# Method names used by failure_modes (the `method` field of its records).
FAILURE_MODE_METHOD = {"tfgrpo": "TFGRPO", "gepa": "GEPA", "ace": "ACE"}


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outputs", nargs="?", default="outputs")
    parser.add_argument("--csv", help="write the per-setting gains to this CSV file")
    args = parser.parse_args()

    rows = []
    print(f"{'method':8s} {'task':12s} {'setting':28s} {'test (%)':>15s} {'gain':>7s}  seeds")
    for method, (pattern, after_score, before_score) in READERS.items():
        for setting_dir in sorted(glob.glob(os.path.join(args.outputs, method, "*", "*"))):
            after, before = [], []
            for path in sorted(glob.glob(os.path.join(setting_dir, pattern))):
                with open(path, encoding="utf-8") as f:
                    result = json.load(f)
                after.append(after_score(result))
                before.append(before_score(result))
            if not after:
                continue
            task, setting = setting_dir.split(os.sep)[-2:]
            teacher, student = setting[len("T-"):].rsplit("_S-", 1)
            after_mean, after_std = mean_std(after)
            # The gain is only defined when every seed also has the score before optimization.
            before_mean = statistics.mean(before) if None not in before else None
            gain = after_mean - before_mean if before_mean is not None else None
            fmt = lambda value: f"{value:7.2f}" if value is not None else f"{'-':>7s}"
            print(f"{method:8s} {task:12s} {setting:28s} {after_mean:7.2f} ± {after_std:5.2f} "
                  f"{fmt(gain)}  {len(after)}")
            if gain is not None:
                rows.append({
                    "method": FAILURE_MODE_METHOD[method], "model": student, "dataset": task, "teacher": teacher,
                    "before": round(before_mean, 4), "after": round(after_mean, 4), "gain": round(gain, 4),
                    "seeds": len(after),
                })

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["method", "model", "dataset", "teacher", "before", "after", "gain", "seeds"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} settings with a gain to {args.csv}")


if __name__ == "__main__":
    main()
