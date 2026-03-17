import os
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


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


def load_checkpoint(path: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    if model is not None and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def compute_iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def save_prediction(image_tensor, boxes, labels=None, scores=None, out_path="prediction.jpg", score_thresh: float = 0.5, class_names=None):
    # image_tensor: torch tensor in [C,H,W] (0..1)
    image = TF.to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        score = None if scores is None else float(scores[i])
        if score is not None and score < score_thresh:
            continue
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = ""
        if labels is not None and i < len(labels):
            name = class_names[int(labels[i])] if class_names is not None and int(labels[i]) < len(class_names) else str(labels[i])
            label = name
        if score is not None:
            label = f"{label} {score:.2f}" if label else f"{score:.2f}"
        if label:
            draw.text((x1 + 3, y1 + 3), label, fill="yellow", font=font)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(out_path)
    return out_path
