from args import get_args
from augmentations import build_train_transforms, build_val_transforms
from dataset import ObjDetectionDataset
import pandas as pd
from model import build_model
import os
import torch
from torch.utils.data import DataLoader
from trainer import train_model
from utils import set_seed, select_device, make_session_dir

def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    args = get_args()
    # seed
    #set_seed(args.seed)

    # 1. Read the dataset (prefer explicit CSV paths when provided)
    train_csv = args.train_csv if hasattr(args, "train_csv") and args.train_csv else os.path.join(args.csv_dir, "train_data.csv")
    val_csv = args.val_csv if hasattr(args, "val_csv") and args.val_csv else os.path.join(args.csv_dir, "val_data.csv")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    # 2. Create the dataset and dataloader
    train_ds = ObjDetectionDataset(
        train_df,
        image_size=args.image_size,
        transforms=build_train_transforms(args.image_size),
    )
    val_ds = ObjDetectionDataset(
        val_df,
        image_size=args.image_size,
        transforms=build_val_transforms(args.image_size),
    )
    # 3. create dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    #images, targets = next(iter(train_dl))

    #4. Init the model and device
    model = build_model(args.backbone, num_classes=args.num_classes + 1)  # +1 for background class
    device_t = select_device(args.device)
    if device_t.type == "cuda":
        device_msg = f"{device_t} ({torch.cuda.get_device_name(device_t)})"
    else:
        device_msg = str(device_t)
    print(f"Using device: {device_msg}")

    #5. Train the model
    # create session dir
    session_dir = make_session_dir(args.out_dir)
    args.out_dir = session_dir  
    train_model(
        model,
        train_dl,
        val_dl,
        device_t,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()    
