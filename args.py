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
        "--test_csv",
        type=str,
        default="./data/csvs/test.csv",
        help="Path to test split CSV file.",
    )

    parser.add_argument(
        "--output_dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default="./sessions",
        help="Directory for checkpoints, logs, and visual outputs.",
    )

    parser.add_argument("--batch_size", type=int, default=8, choices=[8, 16, 32, 64], help="Batch size for dataloaders.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader worker processes.")

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Computation device. Use 'auto' to select the best available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Score threshold for prediction filtering.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable saving prediction visualizations.",
    )

    return parser.parse_args()