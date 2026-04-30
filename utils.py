import random
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def resize_box_xyxy(box, old_w, old_h, new_w, new_h):
    x1, y1, x2, y2 = box

    scale_x = new_w / old_w
    scale_y = new_h / old_h

    x1 *= scale_x
    y1 *= scale_y
    x2 *= scale_x
    y2 *= scale_y

    return x1, y1, x2, y2

def show_batch(images, targets):
    for i in range(len(images)):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        boxes = targets[i]["boxes"].detach().cpu().numpy()
        labels = targets[i]["labels"].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"class {label}",
                fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        ax.set_title(f"Sample {i + 1} in batch")
        ax.axis("off")
        plt.show()

def select_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # macOS metal device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def make_session_dir(base_dir: str = "./sessions", name: str = None) -> str:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder = name or f"session_{stamp}"
    out = base / folder
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass