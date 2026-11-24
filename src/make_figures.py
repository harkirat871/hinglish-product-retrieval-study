"""Generate manuscript figures from results/results.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
PAPER_FIGS = ROOT / "paper" / "figures"
RESULT_FIGS = ROOT / "results" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)
RESULT_FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHODS = ["BM25 (word)", "TF-IDF (char n-gram)", "Hybrid (RRF)"]
LABELS = ["BM25", "Character TF-IDF", "Hybrid RRF"]
LEVELS = ["clean", "light", "heavy"]
LEVEL_LABELS = ["Clean", "Light variation", "Heavy variation"]


def save_figure(fig, stem: str) -> None:
    fig.savefig(PAPER_FIGS / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGS / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(RESULT_FIGS / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def overall_accuracy() -> None:
    values, lower, upper = [], [], []
    for method in METHODS:
        metric = RESULTS[method]["ALL"]
        value = metric["top1"] * 100
        values.append(value)
        lower.append(value - metric["top1_ci"][0] * 100)
        upper.append(metric["top1_ci"][1] * 100 - value)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(METHODS))
    ax.barh(y, values, height=0.56)
    ax.errorbar(values, y, xerr=np.array([lower, upper]), fmt="none",
                capsize=4, linewidth=1.2)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title("Overall retrieval accuracy on answerable queries",
                 loc="left", pad=12, weight="bold")
    ax.grid(axis="x", alpha=0.22)
    for yi, value in zip(y, values):
        ax.text(value + 1.4, yi, f"{value:.1f}%", va="center", weight="bold")
    ax.text(0, -0.19,
            "Error bars show 95% percentile-bootstrap intervals (2,000 resamples).",
            transform=ax.transAxes, fontsize=8.5, color="dimgray")
    fig.tight_layout()
    save_figure(fig, "fig1_overall_top1")


def stratified_accuracy(metric_name: str, stem: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.7))
    x = np.arange(len(LEVELS))
    width = 0.23
    for i, (method, label) in enumerate(zip(METHODS, LABELS)):
        values = [RESULTS[method][level][metric_name] * 100 for level in LEVELS]
        bars = ax.bar(x + (i - 1) * width, values, width, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                    f"{value:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, LEVEL_LABELS)
    ax.set_ylim(0, 110)
    ax.set_ylabel(f"{metric_name.upper()} accuracy (%)")
    ax.set_title(title, loc="left", pad=12, weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save_figure(fig, stem)


def workflow_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    boxes = [
        (0.18, 1.12, 2.05, 1.15, "Synthetic inputs",
         "100-item catalog\n180 generated queries"),
        (2.75, 1.12, 2.05, 1.15, "Text retrieval",
         "BM25\nCharacter TF-IDF"),
        (5.32, 1.12, 1.85, 1.15, "Rank fusion",
         "Reciprocal Rank\nFusion (RRF)"),
        (7.68, 1.12, 2.45, 1.15, "Evaluation",
         "Top-1 / Top-3\nby noise level"),
    ]
    for x, y, width, height, heading, detail in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2, edgecolor="black", facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.77, heading, ha="center",
                va="center", weight="bold", fontsize=10)
        ax.text(x + width / 2, y + 0.36, detail, ha="center",
                va="center", fontsize=8.5)
    for x1, x2 in [(2.23, 2.75), (4.80, 5.32), (7.17, 7.68)]:
        ax.add_patch(FancyArrowPatch(
            (x1 + 0.03, 1.69), (x2 - 0.03, 1.69),
            arrowstyle="-|>", mutation_scale=13,
            linewidth=1.2, color="black",
        ))
    ax.text(5.15, 2.95, "Experimental workflow", ha="center",
            va="center", fontsize=13, weight="bold")
    ax.text(5.15, 0.52,
            "The same catalog is indexed by each method; the hybrid system combines both rankings.",
            ha="center", va="center", fontsize=8.5, color="dimgray")
    fig.tight_layout()
    save_figure(fig, "fig4_experimental_workflow")


def main() -> None:
    overall_accuracy()
    stratified_accuracy("top1", "fig2_top1_by_noise",
                        "Top-1 retrieval accuracy by query-variation level")
    stratified_accuracy("top3", "fig3_top3_by_noise",
                        "Top-3 retrieval accuracy by query-variation level")
    workflow_diagram()
    print("Figures written to paper/figures/ and results/figures/")


if __name__ == "__main__":
    main()
