import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Object Detection Training Configuration")

    parser.add_argument(
        "--backbone",
        type=str,
        default="fasterrcnn_resnet50_fpn",
        choices=["fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3"],
        help="Model backbone architecture.",
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=1, 
        help="Number of object classes (including background)."
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=512, 
        help="Input image size (assumes square)."
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="./data/csvs",
        help="Directory containing dataset split CSV files.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="./data/csvs/train_data.csv",
        help="Path to training split CSV file.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="./data/csvs/val_data.csv",
        help="Path to validation split CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default="./sessions",
        help="Directory for checkpoints, logs, and visual outputs.",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        choices=[2, 4, 8, 10, 16, 32, 64], 
        help="Batch size for dataloaders."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.0001, 
        help="Initial learning rate."
    )
    parser.add_argument(
        "--wd", 
        type=float, 
        default=1e-4, 
        help="Weight decay."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=2, 
        help="Dataloader worker processes."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device. Use 'auto' to select the best available.",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed."
    )

    return parser.parse_args()
