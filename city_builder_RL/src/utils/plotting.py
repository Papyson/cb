# city_builder_rl/utils/plotting.py
from typing import List, Optional, Tuple, Dict
import os

# Headless-safe backend
if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

def _ema(x: List[float], alpha: float = 0.1) -> np.ndarray:
    if not x:
        return np.array([])
    y = np.zeros(len(x), dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def plot_training_curves(
    returns_history: List[float],
    paretoD_history: Optional[List[float]] = None,
    entropy_history: Optional[List[float]] = None,
    tau_history: Optional[List[float]] = None,
    eval_points: Optional[List[Tuple[int, float]]] = None,
    out_path: Optional[str] = None,
    title: str = "A2C Training"
):
    """One-figure dashboard with 3 panels: Return, ParetoD, Entropy/τ."""
    T = len(returns_history)
    x = np.arange(1, T + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    # 1) Return
    axes[0].plot(x, returns_history, label="Return (raw)", linewidth=1, alpha=0.35)
    axes[0].plot(x, _ema(returns_history, alpha=0.1), label="Return (EMA)", linewidth=2)
    if eval_points:
        xs, ys = zip(*eval_points)
        axes[0].plot(xs, ys, "o-", label="Eval Avg Return", linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Return")
    axes[0].legend()

    # 2) Pareto distance
    if paretoD_history:
        axes[1].plot(x, paretoD_history, label="ParetoD", linewidth=1.5)
        axes[1].plot(x, _ema(paretoD_history, alpha=0.1), label="ParetoD (EMA)", linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Pareto Distance")
    axes[1].set_title("Pareto Proximity (↓ better)")
    axes[1].legend()

    # 3) Entropy and τ (two y-axes)
    ax3 = axes[2]
    if entropy_history:
        ax3.plot(x, entropy_history, label="Entropy", linewidth=1.5)
        ax3.plot(x, _ema(entropy_history, alpha=0.1), label="Entropy (EMA)", linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Entropy")
    ax3.set_title("Policy Entropy & Temperature")

    if tau_history:
        ax3b = ax3.twinx()
        ax3b.plot(x, tau_history, "r--", label="τ", linewidth=1.5, alpha=0.7)
        ax3b.set_ylabel("Temperature τ")
        # Combine legends
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax3b.get_legend_handles_labels()
        ax3b.legend(h1 + h2, l1 + l2, loc="upper right")
    else:
        ax3.legend(loc="upper right")

    fig.suptitle(title)

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[PLOT] Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()
