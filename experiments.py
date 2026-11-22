"""Compare three learning policies and inspect pseudo-labels after training."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_limits

from train import plt, self_train, split_data, split_indices


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_seeds():
    rows = []
    for seed in range(10):
        labeled, pool, val, test = split_data(seed)
        policies = {}
        for name in ("frozen", "recursive"):
            base, best, step, history, reason = self_train(
                labeled, pool, val, 0.90, 0.90, 10,
                frozen_teacher=(name == "frozen"), verbose=False,
            )
            policies[name] = (best, step, history[step], reason)
        # Both policies have finished validation-based model selection.
        row = {"seed": seed, "baseline_test": base.score(*test)}
        for name, (model, step, selected, reason) in policies.items():
            row.update({f"{name}_test": model.score(*test),
                        f"{name}_best_round": step,
                        f"{name}_pseudo_count": selected["training_size"] - len(labeled[1]),
                        f"{name}_stop": reason})
        rows.append(row)
    return rows


def plot_gains(path, rows):
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    seeds = np.array([r["seed"] for r in rows])
    for name, offset, color in [("frozen", -0.18, "#61748a"), ("recursive", 0.18, "#147770")]:
        gains = [100 * (r[f"{name}_test"] - r["baseline_test"]) for r in rows]
        ax.bar(seeds + offset, gains, 0.36, label=name.capitalize(), color=color)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(xticks=seeds, xlabel="Split seed", ylabel="Gain (percentage points)",
           title="Change in test accuracy versus 50 labeled examples")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def audit_default(output):
    seed = 42
    labeled, pool, val, _ = split_data(seed)
    log = []
    self_train(labeled, pool, val, 0.90, 0.90, 10, selection_log=log, verbose=False)

    # Only now join the predictions to ground truth, by original dataset index.
    digits = load_digits()
    _, pool_indices, _, _ = split_indices(digits.target, seed)
    for entry in log:
        index = int(pool_indices[entry.pop("pool_id")])
        entry.update(dataset_index=index, truth=int(digits.target[index]))
    write_csv(output / "pseudo_labels_seed42.csv", log)
    errors = sorted((e for e in log if e["prediction"] != e["truth"]),
                    key=lambda e: -e["confidence"])
    fig, axes = plt.subplots(1, 4, figsize=(8, 2.7), layout="constrained")
    fig.suptitle("Most confident wrong pseudo-labels (seed 42)")
    for ax in axes:
        ax.axis("off")
    for ax, e in zip(axes, errors):
        ax.imshow(digits.images[e["dataset_index"]], cmap="gray_r", vmin=0, vmax=16,
                  interpolation="nearest")
        ax.set_title(f"True {e['truth']} | Pred. {e['prediction']}\n"
                     f"p={e['confidence']:.5f}, round {e['round']}", fontsize=10)
    fig.savefig(output / "wrong_labels.png", dpi=150)
    plt.close(fig)
    print(f"Seed 42: {len(errors)} incorrect pseudo-labels among {len(log)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/experiments"))
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        rows = compare_seeds()
        write_csv(output / "seed_comparison.csv", rows)
        plot_gains(output / "seed_gains.png", rows)
        for policy in ("baseline", "frozen", "recursive"):
            print(f"{policy:10} mean test accuracy: {mean(r[policy+'_test'] for r in rows):.2%}")
        audit_default(output)


if __name__ == "__main__":
    main()
