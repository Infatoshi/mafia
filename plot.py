"""Generate training curve plots from results JSON.

Usage:
    uv run python plot.py                    # uses results-v3.json
    uv run python plot.py path/to/results.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(values, window=5):
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_training(results_path, out_dir=None):
    with open(results_path) as f:
        data = json.load(f)

    metrics = data["metrics"]
    out_dir = Path(out_dir or "assets")
    out_dir.mkdir(exist_ok=True)

    iters = [m["iter"] for m in metrics]
    win_rates = [m["win_rate"] for m in metrics]
    avg_rewards = [m["avg_reward"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    grad_norms = [m.get("grad_norm", 0) for m in metrics]
    baseline_wr = data.get("baseline_win_rate", 0.5)

    # -- Style ----------------------------------------------------------------
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.family": "monospace",
        "font.size": 11,
    })

    # -- 1. Win Rate ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(iters, win_rates, color="#58a6ff", alpha=0.35, width=0.8, label="per-batch (8 games)")
    ax.plot(iters, smooth(win_rates, 10), color="#58a6ff", linewidth=2.5, label="10-iter moving avg")
    ax.axhline(y=baseline_wr, color="#f85149", linestyle="--", linewidth=1.5, alpha=0.8, label=f"baseline ({baseline_wr:.0%})")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mafia Win Rate")
    ax.set_title("Mafia Win Rate During Training", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "win_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'win_rate.png'}")

    # -- 2. Reward ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, avg_rewards, color="#8b949e", alpha=0.4, linewidth=0.8)
    ax.plot(iters, smooth(avg_rewards, 10), color="#3fb950", linewidth=2.5, label="avg reward (smoothed)")
    ax.axhline(y=0, color="#f85149", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(iters, avg_rewards, 0, alpha=0.1, color="#3fb950")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Average Reward")
    ax.set_title("Reward Signal Over Training", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'reward.png'}")

    # -- 3. Dashboard (2x2) --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Win rate
    ax = axes[0, 0]
    ax.bar(iters, win_rates, color="#58a6ff", alpha=0.35, width=0.8)
    ax.plot(iters, smooth(win_rates, 10), color="#58a6ff", linewidth=2.5)
    ax.axhline(y=baseline_wr, color="#f85149", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title("Win Rate", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[0, 1]
    ax.plot(iters, avg_rewards, color="#8b949e", alpha=0.4, linewidth=0.8)
    ax.plot(iters, smooth(avg_rewards, 10), color="#3fb950", linewidth=2.5)
    ax.axhline(y=0, color="#f85149", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Average Reward", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1, 0]
    ax.plot(iters, losses, color="#d2a8ff", alpha=0.5, linewidth=0.8)
    ax.plot(iters, smooth(losses, 10), color="#d2a8ff", linewidth=2.5)
    ax.set_title("Policy Loss", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    # Gradient norm
    ax = axes[1, 1]
    ax.plot(iters, grad_norms, color="#f0883e", alpha=0.5, linewidth=0.8)
    ax.plot(iters, smooth(grad_norms, 10), color="#f0883e", linewidth=2.5)
    ax.set_title("Gradient Norm", fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Mafia RL Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_dir / 'dashboard.png'}")

    # -- Summary stats --------------------------------------------------------
    print(f"\n  {len(metrics)} iterations, {data.get('wall_time_minutes', 0):.0f} min")
    print(f"  baseline: {baseline_wr:.0%} -> trained: {data.get('trained_win_rate', 0):.0%}")
    last_10 = metrics[-10:]
    avg_last = sum(m["win_rate"] for m in last_10) / len(last_10)
    print(f"  last 10 iters avg win rate: {avg_last:.0%}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results-v3.json"
    plot_training(path)
