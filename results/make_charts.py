"""
Builds the two figures the README references, from the real numbers recovered
out of the training notebook's outputs.

Design notes:
  - Loss and accuracy are different scales, so they get two panels rather than a
    dual axis.
  - Train vs val is an emphasis pair, not a categorical one: validation is the
    series that matters, so it carries the hue and train recedes to grey.
  - Per-class accuracy is a magnitude question, so it is one hue, light to dark,
    sorted. The weakest classes are the point of the chart, so they are labelled.
  - Transparent background with mid-tone ink, so the PNGs read on both the light
    and dark GitHub themes.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#57606a"      # readable on white and on GitHub dark
GRID = "#d0d7de"
ACCENT = "#1f6feb"   # validation / the series that matters
MUTED = "#9198a1"    # training / context

hist = json.load(open("./_tmp/results/training_history.json"))
per_class = json.load(open("./_tmp/results/per_class_accuracy.json"))

ep = [h["epoch"] for h in hist]


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


# --- figure 1: training curves -------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), dpi=200)
fig.patch.set_alpha(0)

a1.plot(ep, [h["train_loss"] for h in hist], color=MUTED, lw=1.6, label="train")
a1.plot(ep, [h["val_loss"] for h in hist], color=ACCENT, lw=2.0, label="validation")
a1.set_title("Loss", color=INK, fontsize=11, loc="left", pad=10)
a1.set_xlabel("epoch", fontsize=9)
style(a1)

a2.plot(ep, [h["train_acc"] for h in hist], color=MUTED, lw=1.6, label="train")
a2.plot(ep, [h["val_acc"] for h in hist], color=ACCENT, lw=2.0, label="validation")
a2.set_title("Accuracy", color=INK, fontsize=11, loc="left", pad=10)
a2.set_xlabel("epoch", fontsize=9)
a2.set_ylim(0, 1)
style(a2)

# Annotate the single best epoch rather than labelling every point.
best = max(hist, key=lambda h: h["val_acc"])
a2.scatter([best["epoch"]], [best["val_acc"]], s=28, color=ACCENT, zorder=5)
a2.annotate(
    f"best {best['val_acc']:.3f} @ epoch {best['epoch']}",
    xy=(best["epoch"], best["val_acc"]),
    xytext=(-10, -22), textcoords="offset points",
    fontsize=8.5, color=INK, ha="right",
)

handles, labels = a1.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
                 ncol=2, bbox_to_anchor=(0.99, 1.06))
for t in leg.get_texts():
    t.set_color(INK)

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("./_tmp/assets/training_curves.png", transparent=True, bbox_inches="tight")
plt.close(fig)

# --- figure 2: per-class accuracy ----------------------------------------
rows = sorted(per_class, key=lambda r: r["accuracy"])
names = [r["class"] for r in rows]
vals = [r["accuracy"] for r in rows]

fig, ax = plt.subplots(figsize=(8.5, 9), dpi=200)
fig.patch.set_alpha(0)

# Emphasis, not a ramp: the weak classes are the point of this chart, so they
# carry the hue and everything comfortably above the mean recedes to grey.
mean = sum(vals) / len(vals)
WEAK = "#d1495b"
colors = [WEAK if v < 0.90 else MUTED for v in vals]
bars = ax.barh(names, vals, color=colors, height=0.72)

for b, v, n in zip(bars, vals, names):
    weak = v < 0.90
    ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.2f}",
            va="center", fontsize=8.5,
            color=WEAK if weak else INK,
            fontweight="bold" if weak else "normal")

ax.set_xlim(0, 1.09)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("Per-class test accuracy", color=INK, fontsize=12, loc="left", pad=22)
ax.tick_params(axis="y", labelsize=8.5)
for lbl, v in zip(ax.get_yticklabels(), vals):
    if v < 0.90:
        lbl.set_color(WEAK)
        lbl.set_fontweight("bold")
style(ax)
ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
ax.grid(axis="y", visible=False)

# Sits above the plot area so it cannot collide with a bar's value label.
ax.axvline(mean, color=INK, lw=1, ls="--", alpha=0.55)
ax.annotate(f"mean {mean:.2f}", xy=(mean, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 6), textcoords="offset points",
            fontsize=8.5, color=INK, ha="center")
ax.annotate(f"{sum(1 for v in vals if v < 0.90)} classes below 0.90",
            xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 6),
            textcoords="offset points", fontsize=8.5, color=WEAK, ha="left")

fig.tight_layout()
fig.savefig("./_tmp/assets/per_class_accuracy.png", transparent=True, bbox_inches="tight")
plt.close(fig)

print("wrote training_curves.png and per_class_accuracy.png")
