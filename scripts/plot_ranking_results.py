from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Macro-AP", "MRR", "Recall@10", "NDCG@10"]
COLORS = {
    "HIT": "#2F6B9A",
    "TCMIO": "#D97941",
    "MHGNN-BPR": "#4C78A8",
    "HistGB": "#F2A541",
    "Equal-weight fusion": "#3A9D7D",
}


def save_figure(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_multiseed():
    means = {
        "HIT": [0.4869, 0.5585, 0.6921, 0.5419],
        "TCMIO": [0.5007, 0.5709, 0.7266, 0.5606],
    }
    stds = {
        "HIT": [0.0045, 0.0052, 0.0092, 0.0065],
        "TCMIO": [0.0010, 0.0037, 0.0096, 0.0013],
    }

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.7), sharey=True)
    axes = axes.ravel()
    x = np.arange(2)
    for idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        values = [means[d][idx] for d in ("HIT", "TCMIO")]
        errors = [stds[d][idx] for d in ("HIT", "TCMIO")]
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            width=0.58,
            color=[COLORS["HIT"], COLORS["TCMIO"]],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x, ["HIT", "TCMIO"])
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_axisbelow(True)
        for bar, value, error in zip(bars, values, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + 0.018,
                f"{value:.4f}\n±{error:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )
    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    fig.suptitle(
        "Full-candidate ranking performance across three random seeds",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Bars show mean ± standard deviation over three complete five-fold runs (seeds 0, 17, and 42).",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.95))
    save_figure(fig, "full_candidate_ranking_multiseed")


def plot_complementarity():
    means = {
        "HIT": {
            "MHGNN-BPR": [0.2689, 0.3021, 0.6387, 0.3564],
            "HistGB": [0.4351, 0.5184, 0.5924, 0.4811],
            "Equal-weight fusion": [0.4869, 0.5585, 0.6921, 0.5419],
        },
        "TCMIO": {
            "MHGNN-BPR": [0.2877, 0.3247, 0.6896, 0.3857],
            "HistGB": [0.4560, 0.5383, 0.6070, 0.5007],
            "Equal-weight fusion": [0.5007, 0.5709, 0.7266, 0.5606],
        },
    }
    stds = {
        "HIT": {
            "MHGNN-BPR": [0.0371, 0.0439, 0.0146, 0.0343],
            "HistGB": [0.0078, 0.0085, 0.0107, 0.0090],
            "Equal-weight fusion": [0.0045, 0.0052, 0.0092, 0.0065],
        },
        "TCMIO": {
            "MHGNN-BPR": [0.0222, 0.0281, 0.0144, 0.0201],
            "HistGB": [0.0030, 0.0040, 0.0055, 0.0036],
            "Equal-weight fusion": [0.0010, 0.0037, 0.0096, 0.0013],
        },
    }
    methods = ["MHGNN-BPR", "HistGB", "Equal-weight fusion"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    axes = axes.ravel()
    x = np.arange(2)
    width = 0.23
    offsets = [-width, 0.0, width]
    for metric_idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        for method, offset in zip(methods, offsets):
            values = [means[d][method][metric_idx] for d in ("HIT", "TCMIO")]
            errors = [stds[d][method][metric_idx] for d in ("HIT", "TCMIO")]
            bars = ax.bar(
                x + offset,
                values,
                yerr=errors,
                capsize=3,
                width=width,
                label=method,
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.45,
            )
            for bar, value, error in zip(bars, values, errors):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + 0.011,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    rotation=90,
                )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x, ["HIT", "TCMIO"])
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
        fontsize=9,
    )
    fig.suptitle(
        "Multi-seed complementarity of deep and target-statistics branches",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "Bars show mean ± standard deviation over three complete five-fold runs (seeds 0, 17, and 42).",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.89))
    save_figure(fig, "branch_complementarity_multiseed")


def plot_bootstrap_gain():
    datasets = ["HIT", "TCMIO"]
    gains = np.array([0.0516, 0.0448])
    lower = np.array([0.0471, 0.0408])
    upper = np.array([0.0561, 0.0491])
    errors = np.vstack([gains - lower, upper - gains])
    y = np.arange(len(datasets))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
    ax.errorbar(
        gains,
        y,
        xerr=errors,
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=2,
        color="#3A9D7D",
        ecolor="#3A9D7D",
    )
    for yi, gain, lo, hi in zip(y, gains, lower, upper):
        ax.text(
            hi + 0.0012,
            yi,
            f"{gain:.4f} [{lo:.4f}, {hi:.4f}]",
            va="center",
            fontsize=9,
        )
    ax.set_yticks(y, datasets)
    ax.set_xlim(-0.002, 0.067)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Macro-AP gain over HistGB")
    ax.set_title(
        "Paired bootstrap gain of equal-weight fusion over HistGB",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "bootstrap_macro_ap_gain")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_multiseed()
    plot_complementarity()
    plot_bootstrap_gain()
