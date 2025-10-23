from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

def plot_training_curves(returns_history: List[float],
                         eval_points: Optional[List[Tuple[int, float]]] = None,
                         out_path: Optional[str] = None):
    plt.figure(figsize=(8, 4.5))
    plt.plot(returns_history, label="Train Return")
    if eval_points:
        xs, ys = zip(*eval_points)
        plt.plot(xs, ys, "o-", label="Eval Avg Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"[PLOT] Saved {out_path}")
    else:
        plt.show()
