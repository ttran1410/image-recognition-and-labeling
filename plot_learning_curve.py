from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_learning_curve(epochs, train_losses, val_losses, output_path):
    if not (len(epochs) == len(train_losses) == len(val_losses)):
        raise ValueError("Epoch and loss histories must have the same length.")
    if not epochs:
        raise ValueError("At least one epoch is required to plot a learning curve.")

    best_index = min(range(len(val_losses)), key=val_losses.__getitem__)
    best_epoch = epochs[best_index]
    best_val_loss = val_losses[best_index]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", linewidth=2, label="Training Loss")
    ax.plot(epochs, val_losses, marker="o", linewidth=2, label="Validation Loss")
    ax.scatter(
        [best_epoch],
        [best_val_loss],
        color="crimson",
        s=80,
        zorder=5,
        label=f"Best Val Loss (epoch {best_epoch})",
    )
    ax.axvline(best_epoch, color="crimson", linestyle="--", alpha=0.35)
    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
