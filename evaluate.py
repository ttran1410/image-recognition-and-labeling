from __future__ import annotations

from args import parse_evaluation_args
import math
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image

import augmentations as aug
from augmentations import build_val_transforms
from model import build_model
from utils import select_device





def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_state_dict(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def annotate_predictions(image: Image.Image, prediction: dict, score_threshold: float) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    boxes = prediction.get("boxes", torch.zeros((0, 4)))
    labels = prediction.get("labels", torch.zeros((0,), dtype=torch.int64))
    scores = prediction.get("scores", torch.zeros((0,)))

    kept = scores >= score_threshold
    boxes = boxes[kept]
    labels = labels[kept]
    scores = scores[kept]

    if boxes.numel() == 0:
        message = f"No detections above {score_threshold:.2f}"
        draw.rectangle([(8, 8), (280, 36)], fill="black")
        draw.text((12, 12), message, fill="white", font=font)
        return image

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [float(coord) for coord in box.tolist()]
        text = f"class {int(label)}: {float(score):.2f}"
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        text_x = max(0, x1)
        text_y = max(0, y1 - 14)
        text_bbox = draw.textbbox((text_x, text_y), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((text_x, text_y), text, fill="white", font=font)

    return image


def save_contact_sheet(image_paths, output_path: Path, columns: int = 3, thumb_size=(320, 320)):
    if not image_paths:
        return

    thumbs = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail(thumb_size)
        canvas = Image.new("RGB", thumb_size, "white")
        offset_x = (thumb_size[0] - img.width) // 2
        offset_y = (thumb_size[1] - img.height) // 2
        canvas.paste(img, (offset_x, offset_y))
        thumbs.append(canvas)

    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "white")

    for index, thumb in enumerate(thumbs):
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * thumb_size[1]
        sheet.paste(thumb, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main():
    args = parse_evaluation_args()
    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    project_root = Path(__file__).resolve().parent
    test_csv_path = resolve_path(args.test_csv, project_root)
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else checkpoint_path.parent / "evaluation_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_csv_path)
    if args.max_images > 0:
        test_df = test_df.head(args.max_images)

    model = build_model(args.backbone, num_classes=args.num_classes + 1, pretrained=False)
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model.to(device)
    model.eval()

    preprocess = aug.Compose(build_val_transforms(args.image_size))
    saved_paths = []

    with torch.no_grad():
        for index, row in test_df.iterrows():
            image_path = resolve_path(row["images"], project_root)
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image)

            image_tensor, _ = preprocess(image, None)
            prediction = model([image_tensor.to(device=device, dtype=torch.float32)])[0]

            annotated = to_pil_image(image_tensor.cpu())
            annotated = annotate_predictions(annotated, prediction, args.score_threshold)

            output_path = output_dir / f"{image_path.stem}_pred.png"
            annotated.save(output_path)
            saved_paths.append(output_path)

            print(f"[{index + 1}/{len(test_df)}] saved {output_path.name}")

    contact_sheet_path = output_dir / "evaluation_contact_sheet.png"
    save_contact_sheet(saved_paths[:9], contact_sheet_path)

    print(f"Saved {len(saved_paths)} annotated test images to {output_dir}")
    print(f"Saved contact sheet to {contact_sheet_path}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Test CSV: {test_csv_path}")


if __name__ == "__main__":
    main()
