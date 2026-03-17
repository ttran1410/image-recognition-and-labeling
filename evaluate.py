import os
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from args import get_args
from dataset import ObjDetectionDataset
from model import build_model
from utils import select_device, load_checkpoint, make_session_dir, compute_iou
from utils import collate_fn
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


def evaluate(args):
    device = select_device(args.device)

    # locate checkpoint
    ckpt = args.resume if getattr(args, "resume", "") else os.path.join(args.out_dir, "checkpoint_best.pth")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        return

    # build model and load weights
    model = build_model(args.backbone, num_classes=2)
    model.to(device)
    load_checkpoint(ckpt, model=model, map_location=device)
    model.eval()

    # prepare test dataset
    test_csv = args.test_csv if getattr(args, "test_csv", "") else os.path.join(args.csv_dir, "test.csv")
    if not os.path.exists(test_csv):
        print(f"Test CSV not found: {test_csv}")
        return

    df = pd.read_csv(test_csv)
    dataset = ObjDetectionDataset(df)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=getattr(args, "num_workers", 0))

    out_dir = make_session_dir(args.out_dir, name=f"eval_{int(time.time())}")

    iou_thresh = 0.5
    score_thresh = getattr(args, "score_thresh", 0.5)

    TP = 0
    FP = 0
    FN = 0
    ious = []

    vis_saved = 0
    max_vis = 100

    for idx, (images, targets) in enumerate(loader):
        img = images[0]
        # move to device
        img_t = img.to(device)

        with torch.no_grad():
            preds = model([img_t])[0]

        pred_boxes = preds.get("boxes", torch.zeros((0, 4))).cpu().numpy()
        pred_scores = preds.get("scores", torch.zeros((0,))).cpu().numpy()

        # filter by score
        keep = [i for i, s in enumerate(pred_scores) if s >= score_thresh]
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]

        gt = targets[0]
        gt_boxes = gt.get("boxes", torch.zeros((0, 4))).cpu().numpy() if "boxes" in gt else np.zeros((0, 4))

        matched_gt = set()

        # match predictions to GT
        for p in pred_boxes:
            best_iou = 0.0
            best_k = -1
            for k, g in enumerate(gt_boxes):
                if k in matched_gt:
                    continue
                iou = compute_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_k = k

            if best_iou >= iou_thresh:
                TP += 1
                matched_gt.add(best_k)
                ious.append(best_iou)
            else:
                FP += 1

        FN += max(0, len(gt_boxes) - len(matched_gt))

        # save visualization
        if getattr(args, "visualize", False) and vis_saved < max_vis:
            pil = TF.to_pil_image(img.cpu())
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            # draw GT in green
            for g in gt_boxes:
                draw.rectangle([float(g[0]), float(g[1]), float(g[2]), float(g[3])], outline="green", width=2)

            # draw predictions in red with scores
            for i_p, p in enumerate(pred_boxes):
                draw.rectangle([float(p[0]), float(p[1]), float(p[2]), float(p[3])], outline="red", width=2)
                score = pred_scores[i_p] if i_p < len(pred_scores) else None
                if score is not None:
                    draw.text((float(p[0]) + 3, float(p[1]) + 3), f"{score:.2f}", fill="yellow", font=font)

            vis_path = os.path.join(out_dir, f"pred_{idx:04d}.jpg")
            pil.save(vis_path)
            vis_saved += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    results = {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou),
        "num_images": len(dataset),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
