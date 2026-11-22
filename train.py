"""Train a small self-learning digit classifier. Run: python train.py"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def split_indices(y, seed):
    train_val, test = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=seed
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.80, stratify=y[train_val], random_state=seed
    )
    rng = np.random.default_rng(seed)
    labeled = np.concatenate([
        rng.choice(train[y[train] == digit], 5, replace=False) for digit in range(10)
    ])
    pool = np.setdiff1d(train, labeled)
    return labeled, pool, val, test


def split_data(seed):
    x, y = load_digits(return_X_y=True)
    labeled, pool, val, test = split_indices(y, seed)
    return (x[labeled], y[labeled]), x[pool], (x[val], y[val]), (x[test], y[test])


def select_examples(probabilities, classes, threshold):
    labels = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    selected = []
    for digit in classes:
        eligible = np.flatnonzero((labels == digit) & (confidence >= threshold))
        order = np.argsort(-confidence[eligible], kind="stable")
        selected.extend(eligible[order[:5]])
    rows = np.asarray(selected, dtype=int)
    return rows, labels[rows]


def fit(x, y):
    return LogisticRegression(C=0.1, max_iter=2000).fit(x, y)


def self_train(labeled, pool, validation, threshold, target, max_rounds, *,
               frozen_teacher=False, selection_log=None, verbose=True):
    x, y = labeled
    baseline = best = current = fit(x, y)
    best_round = 0
    added = 0
    history = []
    if selection_log is not None:
        pool_ids = np.arange(len(pool))
    for step in range(max_rounds + 1):
        score = float(current.score(*validation))
        history.append(dict(round=step, added=added, training_size=len(y),
                            pool_size=len(pool), val_accuracy=score))
        if verbose:
            print(f"Round {step:2}: added={added:2}, train={len(y):3}, val={score:.2%}")
        if score > history[best_round]["val_accuracy"]:
            best, best_round = current, step
        if score >= target:
            reason = "target_reached"
            break
        if step == max_rounds:
            reason = "max_rounds"
            break
        if not len(pool):
            reason = "empty_pool"
            break
        teacher = baseline if frozen_teacher else current
        probabilities = teacher.predict_proba(pool)
        rows, predictions = select_examples(probabilities, teacher.classes_, threshold)
        if not len(rows):
            reason = "no_candidates"
            break
        if selection_log is not None:
            for row, prediction in zip(rows, predictions):
                selection_log.append(dict(round=step + 1, pool_id=int(pool_ids[row]),
                                          prediction=int(prediction),
                                          confidence=float(probabilities[row].max())))
            pool_ids = np.delete(pool_ids, rows)
        x = np.concatenate([x, pool[rows]])
        y = np.concatenate([y, predictions])
        pool = np.delete(pool, rows, axis=0)
        added = len(rows)
        current = fit(x, y)
    return baseline, best, best_round, history, reason


def plot_history(path, history, best_round, target):
    rounds = [r["round"] for r in history]
    scores = [r["val_accuracy"] for r in history]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(rounds, scores, "o-", color="#176a71", label="Validation accuracy")
    ax.axhline(target, color="#707070", ls="--", label=f"Target: {target:.0%}")
    ax.scatter(best_round, scores[best_round], s=110, facecolors="none",
               edgecolors="#bb500d", zorder=3, label=f"Selected: round {best_round}")
    ax.set(xlabel="Self-training round (0 = baseline)", ylabel="Accuracy",
           title="Handwritten digits: learning from pseudo-labels", xticks=rounds)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--validation-only", action="store_true",
                        help="Skip test scores when exploring settings.")
    args = parser.parse_args()
    if not (0 <= args.threshold <= 1 and 0 <= args.target <= 1):
        parser.error("--threshold and --target must be between 0 and 1")
    if args.max_rounds < 0 or not 0 <= args.seed < 2**32:
        parser.error("--max-rounds must be nonnegative and --seed must fit uint32")

    labeled, pool, validation, test = split_data(args.seed)
    with threadpool_limits(limits=1):
        baseline, best, best_round, history, reason = self_train(
            labeled, pool, validation, args.threshold, args.target, args.max_rounds
        )
        summary = {
            "seed": args.seed, "threshold": args.threshold, "target": args.target,
            "max_rounds": args.max_rounds, "split_sizes": {
                "labeled": len(labeled[1]), "unlabeled": len(pool),
                "validation": len(validation[1]), "test": len(test[1]),
            },
            "stop_reason": reason, "best_round": best_round,
            "baseline_val_accuracy": history[0]["val_accuracy"],
            "best_val_accuracy": history[best_round]["val_accuracy"],
        }
        if not args.validation_only:
            summary["baseline_test_accuracy"] = float(baseline.score(*test))
            summary["best_test_accuracy"] = float(best.score(*test))

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_history(args.output / "accuracy.png", history, best_round, args.target)
    joblib.dump(best, args.output / "best_model.joblib")
    print(f"Stop: {reason}; selected round: {best_round}; output: {args.output}")
    if not args.validation_only:
        print(f"Test: baseline={summary['baseline_test_accuracy']:.2%}, "
              f"selected={summary['best_test_accuracy']:.2%}")


if __name__ == "__main__":
    main()
