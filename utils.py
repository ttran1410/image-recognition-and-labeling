import random
import time
import numpy as np
from pathlib import Path
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