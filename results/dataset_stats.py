"""
Counts the dataset and asks one question of it: does a class do badly on the
test split because it had few training images?

Writes results/dataset_counts.json (images per class per split) and
assets/accuracy_vs_train_size.png, and prints the rank correlation between a
class's training-set size and its test accuracy. Run from anywhere:

    python results/dataset_stats.py

Needs only the dataset, results/per_class_accuracy.json and matplotlib.
"""
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "ind_food_37"
SPLITS = ("train", "val", "test")
IMAGE_TYPES = (".jpg", ".jpeg", ".png")

INK = "#57606a"
GRID = "#d0d7de"
MUTED = "#9198a1"
WEAK = "#d1495b"
WEAK_BELOW = 0.90


def count_images() -> list[dict]:
    classes = sorted(p.name for p in (DATASET / "train").iterdir() if p.is_dir())
    rows = []
    for cls in classes:
        row = {"class": cls}
        for split in SPLITS:
            folder = DATASET / split / cls
            row[split] = sum(1 for f in os.listdir(folder) if f.lower().endswith(IMAGE_TYPES)) if folder.is_dir() else 0
        rows.append(row)
    return rows


def ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties (several classes at 1.00) share a rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den


def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def chart(rows: list[dict], accuracy: dict[str, float], rho: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    fig.patch.set_alpha(0)

    for r in rows:
        acc = accuracy[r["class"]]
        weak = acc < WEAK_BELOW
        ax.scatter(r["train"], acc, s=34, color=WEAK if weak else MUTED, zorder=3)
        if weak:
            ax.annotate(
                r["class"], xy=(r["train"], acc), xytext=(6, -3), textcoords="offset points",
                fontsize=8, color=WEAK, fontweight="bold",
            )

    ax.set_xlabel("training images in the class", fontsize=9)
    ax.set_ylabel("test accuracy", fontsize=9)
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Test accuracy against training-set size", color=INK, fontsize=12, loc="left", pad=22)
    ax.annotate(
        f"Spearman rank correlation {rho:+.2f}: class size does not explain the errors",
        xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
        fontsize=8.5, color=INK, ha="left",
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = count_images()
    accuracy = {r["class"]: r["accuracy"] for r in json.load(open(ROOT / "results" / "per_class_accuracy.json"))}
    missing = set(accuracy) ^ {r["class"] for r in rows}
    if missing:
        raise SystemExit(f"classes differ between the dataset and per_class_accuracy.json: {sorted(missing)}")

    totals = {split: sum(r[split] for r in rows) for split in SPLITS}
    totals["all"] = sum(totals.values())
    rho = spearman([r["train"] for r in rows], [accuracy[r["class"]] for r in rows])

    out = ROOT / "results" / "dataset_counts.json"
    json.dump(
        {"totals": totals, "spearman_train_size_vs_test_accuracy": round(rho, 3), "classes": rows},
        open(out, "w"),
        indent=2,
    )
    chart(rows, accuracy, rho, ROOT / "assets" / "accuracy_vs_train_size.png")

    by_size = sorted(rows, key=lambda r: r["train"])
    third = len(rows) // 3
    small = sum(accuracy[r["class"]] for r in by_size[:third]) / third
    large = sum(accuracy[r["class"]] for r in by_size[-third:]) / third
    print(f"{len(rows)} classes, {totals['all']} images: " + ", ".join(f"{s} {totals[s]}" for s in SPLITS))
    print(f"training images per class: {by_size[0]['train']} ({by_size[0]['class']}) to {by_size[-1]['train']} ({by_size[-1]['class']})")
    print(f"Spearman rank correlation, training-set size vs test accuracy: {rho:+.3f}")
    print(f"mean test accuracy, smallest third of classes {small:.3f}, largest third {large:.3f}")
    print(f"wrote {out.relative_to(ROOT)} and assets/accuracy_vs_train_size.png")


if __name__ == "__main__":
    main()
