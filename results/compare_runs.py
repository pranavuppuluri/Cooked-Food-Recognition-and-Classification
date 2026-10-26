"""
Compares training runs made with `python -m indfood.train`, side by side.

    python results/compare_runs.py runs/baseline runs/augment

Reads each run's history.json and config.json, writes
results/augmentation_experiment.json with the numbers behind the chart, and
draws assets/augmentation_experiment.png in the style of the other figures.
The first run named is the reference the others are compared against.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK = "#57606a"
GRID = "#d0d7de"
ACCENT = "#1f6feb"
MUTED = "#9198a1"
WEAK = "#d1495b"


def style(ax):
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def load(run: Path) -> dict:
    history = json.load(open(run / "history.json"))
    config = json.load(open(run / "config.json"))
    best = max(history, key=lambda h: h["val_acc"])
    return {
        "run": run.name,
        "augment": config.get("augment", False),
        "epochs": len(history),
        "best_val_acc": best["val_acc"],
        "best_epoch": best["epoch"],
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "final_val_loss": history[-1]["val_loss"],
        "minutes": round(sum(h.get("seconds", 0) for h in history) / 60, 1),
        "history": history,
    }


def main(paths: list[str]) -> None:
    runs = [load(Path(p)) for p in paths]
    ref, others = runs[0], runs[1:]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=200)
    fig.patch.set_alpha(0)
    colours = [MUTED, ACCENT, WEAK, INK]
    for r, colour in zip(runs, colours):
        ep = [h["epoch"] for h in r["history"]]
        label = f"{r['run']}: " + ("crop, flip, colour jitter" if r["augment"] else "no augmentation")
        a1.plot(ep, [h["val_acc"] for h in r["history"]], color=colour, lw=2.0, label=label)
        a1.plot(ep, [h["train_acc"] for h in r["history"]], color=colour, lw=1.2, ls=":", alpha=0.8)
        a2.plot(ep, [h["val_loss"] for h in r["history"]], color=colour, lw=2.0, label=label)
    a1.set_title("Accuracy (solid: validation, dotted: train)", color=INK, fontsize=11, loc="left", pad=10)
    a1.set_ylim(0, 1)
    a2.set_title("Validation loss", color=INK, fontsize=11, loc="left", pad=10)
    for ax in (a1, a2):
        ax.set_xlabel("epoch", fontsize=9)
        style(ax)
    leg = a1.legend(frameon=False, fontsize=8.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "augmentation_experiment.png", transparent=True, bbox_inches="tight")

    summary = [{k: v for k, v in r.items() if k != "history"} for r in runs]
    json.dump({"runs": summary, "histories": {r["run"]: r["history"] for r in runs}}, open(ROOT / "results" / "augmentation_experiment.json", "w"), indent=2)

    print(f"{'run':<12} {'aug':<5} {'epochs':>6} {'best val':>9} {'at':>4} {'final train':>12} {'final val':>10} {'minutes':>8}")
    for r in summary:
        print(f"{r['run']:<12} {str(r['augment']):<5} {r['epochs']:>6} {r['best_val_acc']:>9.4f} {r['best_epoch']:>4} {r['final_train_acc']:>12.4f} {r['final_val_acc']:>10.4f} {r['minutes']:>8}")
    for r in summary[1:]:
        print(f"{r['run']} vs {ref['run']}: best validation accuracy {r['best_val_acc'] - ref['best_val_acc']:+.4f}, "
              f"train/val gap {r['final_train_acc'] - r['final_val_acc']:+.4f} vs {ref['final_train_acc'] - ref['final_val_acc']:+.4f}")
    print("wrote assets/augmentation_experiment.png and results/augmentation_experiment.json")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/baseline", "runs/augment"])
