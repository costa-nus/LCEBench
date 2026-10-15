"""Vendi Score of failure modes per setting, and its correlation with LCE gains."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELL = ("method", "model", "dataset")


def vendi_score(embeddings):
    """exp of the eigenvalue entropy of K/N, K the cosine-similarity matrix."""
    x = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(x @ x.T / len(x))
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def subsampled_vendi(embeddings, n, draws, rng):
    """Vendi Scores of `draws` random subsets of size n (the score depends on n)."""
    return np.array([
        vendi_score(embeddings[rng.choice(len(embeddings), n, replace=False)])
        for _ in range(draws)
    ])


def correlate(x, y):
    """Pearson r and p-value between the Vendi Score and the gain."""
    from scipy.stats import linregress

    fit = linregress(x, y, alternative="less")
    return float(fit.rvalue), float(fit.pvalue)


def load_mechanisms(path):
    cells = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                cells[tuple(record[field] for field in CELL)].append(record["failure_mechanism"])
    if not cells:
        raise ValueError(f"{path}: no failure mechanisms")
    return cells


def load_gains(path):
    """Gains from `scripts/summarize.py --csv`; only self-reflection settings (teacher = model) are kept."""
    with path.open(newline="", encoding="utf-8") as source:
        return {
            tuple(row[field] for field in CELL): float(row["gain"])
            for row in csv.DictReader(source)
            if row.get("teacher", row["model"]) == row["model"]
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("failure_mechanisms.jsonl"),
                        help="output of failure_modes.run")
    parser.add_argument("--gains", type=Path,
                        help="CSV from `scripts/summarize.py --csv` (method,model,dataset,gain, ...)")
    parser.add_argument("--output", type=Path, default=Path("failure_vendi.csv"))
    parser.add_argument("--n", type=int,
                        help="failure modes per subsample (default: the smallest number of failure modes of a setting)")
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--lce-methods", nargs="+", default=["ACE", "GEPA", "TFGRPO"],
                        help="methods pooled into the LCEs correlation")
    args = parser.parse_args()
    cells = load_mechanisms(args.input)
    # The Vendi Score depends on the sample size, so every setting is subsampled to the same n.
    if args.n is None:
        args.n = min(len(texts) for texts in cells.values())
        print(f"Using n={args.n} failure modes per subsample")
    if args.n < 1 or args.draws < 1:
        parser.error("--n and --draws must be positive")
    small = [cell for cell, texts in cells.items() if len(texts) < args.n]
    if small:
        parser.error(f"{len(small)} settings have fewer than --n {args.n} failure modes, e.g. {small[0]}")
    gains = load_gains(args.gains) if args.gains else {}
    missing = [cell for cell in cells if args.gains and cell not in gains]
    if missing:
        parser.error(f"{args.gains}: no gain for {len(missing)} settings, e.g. {missing[0]}")

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(args.embedding_model)
    rng = np.random.default_rng(args.seed)
    rows = []
    for cell in sorted(cells):
        scores = subsampled_vendi(encoder.encode(cells[cell]), args.n, args.draws, rng)
        row = dict(zip(CELL, cell))
        row.update(n_failures=len(cells[cell]), vendi_mean=scores.mean(), vendi_std=scores.std(ddof=1))
        if gains:
            row["gain"] = gains[cell]
        rows.append(row)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} settings to {args.output}")

    if gains:
        scopes = {method: [method] for method in sorted({row["method"] for row in rows})}
        scopes["LCEs"] = args.lce_methods
        for scope, methods in scopes.items():
            selected = [row for row in rows if row["method"] in methods]
            if len(selected) < 3:
                print(f"{scope}: {len(selected)} settings, too few for a correlation")
                continue
            r, p = correlate([row["vendi_mean"] for row in selected], [row["gain"] for row in selected])
            print(f"{scope}: n={len(selected)} r={r:.3f} p={p:.4f}")


if __name__ == "__main__":
    main()
