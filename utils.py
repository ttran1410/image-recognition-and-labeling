import os
import time
import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 42):
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
