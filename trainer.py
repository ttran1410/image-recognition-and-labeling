import os
import torch
import torch.optim as optim
from plot_learning_curve import save_learning_curve
from utils import show_batch


def train_model(model, train_loader, val_loader, device,lr, wd, epochs, out_dir):
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            #print(len(train_loader.dataset))
            images = [img.to(device=device, dtype=torch.float32) for img in images]
            targets = [
                {
                    "boxes": t["boxes"].to(device=device, dtype=torch.float32),
                    "labels": t["labels"].to(device=device, dtype=torch.int64),
                }
                for t in targets
            ]

            #show_batch(images, targets)
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            if not isinstance(loss_dict, dict):
                raise TypeError(
                    f"Expected training loss dict, got {type(loss_dict).__name__} instead."
                )
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        dataset_len = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
        if dataset_len > 0:
            train_epoch_loss = running_loss / dataset_len
        else:
            train_epoch_loss = float("inf")

        val_loss = validate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_epoch_loss,
                "val_loss": val_loss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

        # Save training info to file
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "training_log.txt"), "a") as f:
            f.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}\n"
            )
        # Also print training info to console    
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    if history:
        save_learning_curve(
            epochs=[entry["epoch"] for entry in history],
            train_losses=[entry["train_loss"] for entry in history],
            val_losses=[entry["val_loss"] for entry in history],
            output_path=os.path.join(out_dir, "learning_curve.png"),
        )

def validate_model(model, val_loader, device):
    was_training = model.training
    model.train()

    val_loss_sum = 0.0
    val_count = 0

    try:
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device=device, dtype=torch.float32) for img in images]
                targets = [
                    {
                        "boxes": t["boxes"].to(device=device, dtype=torch.float32),
                        "labels": t["labels"].to(device=device, dtype=torch.int64),
                    }
                    for t in targets
                ]

                loss_dict = model(images, targets)
                if not isinstance(loss_dict, dict):
                    raise TypeError(
                        f"Expected validation loss dict, got {type(loss_dict).__name__} instead."
                    )
                loss = sum(loss_value for loss_value in loss_dict.values())

                val_loss_sum += loss.item() * len(images)
                val_count += len(images)
    finally:
        model.train(was_training)

    if val_count == 0:
        return float("inf")

    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss
