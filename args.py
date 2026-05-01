import argparse
import os


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
        help="Number of foreground object classes.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Input image size (assumes square).",
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
        help="Random seed.",
    )
    return parser


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object Detection Training Configuration")
    add_shared_args(parser)

    parser.add_argument(
        "--csv_dir",
        type=str,
        default="./data/csvs",
        help="Directory containing dataset split CSV files.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="./data/csvs/train.csv",
        help="Path to training split CSV file.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="./data/csvs/val.csv",
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
        help="Batch size for dataloaders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Dataloader worker processes.",
    )

    return parser


def build_evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained object detector on the test split.")
    add_shared_args(parser)

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved best_model.pth checkpoint.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="./data/csvs/test.csv",
        help="Path to the untouched test CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation outputs. Defaults to <checkpoint_folder>/evaluation_outputs.",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score required to draw a prediction.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        help="Maximum number of test images to evaluate.",
    )

    return parser


def parse_training_args() -> argparse.Namespace:
    parser = build_training_parser()
    return parser.parse_args()


def parse_evaluation_args() -> argparse.Namespace:
    parser = build_evaluation_parser()
    return parser.parse_args()
