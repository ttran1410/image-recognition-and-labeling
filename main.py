from args import get_args
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import ObjDetectionDataset
from model import build_model
import os
from pathlib import Path
from utils import set_seed, select_device, make_session_dir
from trainer import Trainer

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    # seed and device
    set_seed(args.seed)
    # 1. Read the dataset (prefer explicit CSV paths when provided)
    train_csv = args.train_csv if hasattr(args, "train_csv") and args.train_csv else os.path.join(args.csv_dir, "train.csv")
    val_csv = args.val_csv if hasattr(args, "val_csv") and args.val_csv else os.path.join(args.csv_dir, "val.csv")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    # 2. Create the dataset and dataloader
    train_ds = ObjDetectionDataset(train_df)
    val_ds = ObjDetectionDataset(val_df)
    # 3. create dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=getattr(args, "num_workers", 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=getattr(args, "num_workers", 0),
    )
    
    #4. Init the model
    # device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = build_model(args.backbone, num_classes=2)
    device_t = select_device(args.device)
    model.to(device_t)

    # create session dir
    session_dir = make_session_dir(args.out_dir)

    # Trainer
    trainer = Trainer(model, args, device_t, out_dir=session_dir)
    trainer.train(train_dl, val_dl)

if __name__ == "__main__":
    main()    