from args import get_args
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from utils import resize_box_xyxy


class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size, transforms=None, base_dir=None):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.transforms = transforms
        # Base directory to resolve relative image/label paths. Defaults to cwd.
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    def __len__(self):
        return len(self.df)

    def _resolve(self, p: str) -> str:
        if not p:
            return p
        p = str(p)
        path = Path(p)
        if path.is_absolute():
            return str(path)
        # try relative to base_dir
        candidate = (self.base_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
        # fallback to given path
        return p

    def __getitem__(self, idx):
        args = get_args()

        row = self.df.iloc[idx]

        img_path = self._resolve(row["images"])
        lbl_path = self._resolve(row["labels"]) if "labels" in row else ""

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        img = img.resize((self.image_size, self.image_size))  # no resizing, but ensures consistent PIL image format
        image = to_tensor(img)

        boxes = []
        labels = []

        # Read label file if present; tolerate non-annotation lines
        if lbl_path and os.path.exists(lbl_path):
            try:
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cls, xc, yc, bw, bh = map(float, parts[:5])
                        except ValueError:
                            continue
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        # clamp coordinates
                        x1 = max(0.0, min(x1, w))
                        y1 = max(0.0, min(y1, h))
                        x2 = max(0.0, min(x2, w))
                        y2 = max(0.0, min(y2, h))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        x1, y1, x2, y2 = resize_box_xyxy((x1, y1, x2, y2), w, h, self.image_size, self.image_size)
                        boxes.append([x1, y1, x2, y2])
                        # map label to project classes: background=0, object=1
                        labels.append(int(cls) + 1)
            except Exception:
                boxes = []
                labels = []

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)

        areas = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0]) if boxes_t.shape[0] > 0 else torch.zeros((0,))
        iscrowd = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target