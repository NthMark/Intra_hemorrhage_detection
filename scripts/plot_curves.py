"""
Plot training/validation loss & AUC curves from training_history.json.

Usage:
    python scripts/plot_curves.py                          # default paths
    python scripts/plot_curves.py --history models/checkpoints/training_history.json
    python scripts/plot_curves.py --out reports/figures/loss_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def diagnose(train_losses: list[float], val_losses: list[float]) -> str:
    """Return a plain-text overfitting / underfitting diagnosis."""
    if len(train_losses) < 2:
        return "Not enough epochs to diagnose."

    final_train = train_losses[-1]
    final_val   = val_losses[-1]
    gap         = final_val - final_train          # positive → val worse → overfit
    trend_train = train_losses[-1] - train_losses[0]  # negative = still descending

    lines = [
        f"  Final train loss : {final_train:.4f}",
        f"  Final val loss   : {final_val:.4f}",
        f"  Gap (val-train)  : {gap:+.4f}",
    ]

    if final_train > 0.4 and final_val > 0.4:
        verdict = "UNDERFITTING — both losses are high. Try longer training, larger model, or lower regularisation."
    elif gap > 0.15:
        verdict = "OVERFITTING — val loss much higher than train loss. Try more augmentation, dropout, or early stopping."
    elif gap > 0.05:
        verdict = "MILD OVERFITTING — small gap, keep monitoring."
    elif trend_train < -0.01 and final_train < 0.2:
        verdict = "GOOD FIT — train loss still improving with low val gap."
    else:
        verdict = "GOOD FIT — losses converged and are close together."

    lines.append(f"\n  Verdict: {verdict}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves and diagnose fit")
    parser.add_argument(
        "--history",
        default="models/checkpoints/training_history.json",
        help="Path to training_history.json (default: models/checkpoints/training_history.json)",
    )
    parser.add_argument(
        "--out",
        default="reports/figures/loss_curves.png",
        help="Output image path (default: reports/figures/loss_curves.png)",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        print(f"[ERROR] History file not found: {history_path}")
        print("  → Run 'make train' first to generate training_history.json")
        raise SystemExit(1)

    with open(history_path) as f:
        history = json.load(f)

    epochs      = [h["epoch"]      for h in history]
    train_loss  = [h["train_loss"] for h in history]
    val_loss    = [h["val_loss"]   for h in history]
    val_auc     = [h.get("val_auc", None) for h in history]
    has_auc     = any(v is not None for v in val_auc)

    # ── Print diagnosis ───────────────────────────────────────────────────────
    print("\n=== Training Curve Diagnosis ===")
    print(diagnose(train_loss, val_loss))
    if has_auc:
        print(f"\n  Best val AUC : {max(v for v in val_auc if v is not None):.4f}  "
              f"(epoch {val_auc.index(max(v for v in val_auc if v is not None)) + 1})")
    print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        raise SystemExit(1)

    nrows = 2 if has_auc else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4 * nrows), tight_layout=True)
    if nrows == 1:
        axes = [axes]

    # — Loss subplot —
    ax = axes[0]
    ax.plot(epochs, train_loss, "b-o", markersize=4, label="Train loss")
    ax.plot(epochs, val_loss,   "r-o", markersize=4, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Focal)")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade gap between curves to make overfit/underfit visual
    ax.fill_between(epochs, train_loss, val_loss,
                    where=[v > t for t, v in zip(train_loss, val_loss)],
                    alpha=0.12, color="red",  label="Overfit region")
    ax.fill_between(epochs, train_loss, val_loss,
                    where=[v < t for t, v in zip(train_loss, val_loss)],
                    alpha=0.12, color="blue", label="Underfit region")

    # — AUC subplot —
    if has_auc:
        ax2 = axes[1]
        ax2.plot(epochs, val_auc, "g-o", markersize=4, label="Val AUC (macro)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC")
        ax2.set_title("Validation AUC over Epochs")
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        best_epoch = val_auc.index(max(v for v in val_auc if v is not None))
        ax2.axvline(epochs[best_epoch], color="green", linestyle="--", alpha=0.5,
                    label=f"Best epoch {epochs[best_epoch]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
