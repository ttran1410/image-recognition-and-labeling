from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "images"
LABELS_DIR = ROOT / "data" / "labels"
CSV_DIR = ROOT / "data" / "csvs"
DATASET_CSV = CSV_DIR / "dataset.csv"
TRAIN_CSV = CSV_DIR / "train.csv"
VAL_CSV = CSV_DIR / "val.csv"
TEST_CSV = CSV_DIR / "test.csv"

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_pairs():
    label_stems = {
        path.stem
        for path in LABELS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    }

    rows = []
    missing_labels = []

    for img in sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_SUFFIXES
    ):
        if img.stem not in label_stems:
            missing_labels.append(img.stem)
            continue

        rows.append(
            {
                "images": f"data/images/{img.name}",
                "labels": f"data/labels/{img.stem}.txt",
            }
        )
        label_stems.discard(img.stem)

    return pd.DataFrame(rows, columns=["images", "labels"]), missing_labels, sorted(label_stems)


def split_dataset(dataset: pd.DataFrame, seed: int = SEED):
    shuffled = dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    n_test = int(round(total * TEST_RATIO))
    n_val = int(round(total * VAL_RATIO))
    n_train = total - n_test - n_val

    if n_train < 0:
        raise ValueError("Split ratios produce an invalid negative train size.")

    train_df = shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = shuffled.iloc[n_train + n_val :].reset_index(drop=True)

    return train_df, val_df, test_df


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    dataset, missing_labels, orphan_labels = collect_pairs()

    if dataset.empty:
        raise RuntimeError("No valid image-label pairs were found.")

    write_csv(dataset, DATASET_CSV)
    train_df, val_df, test_df = split_dataset(dataset)
    write_csv(train_df, TRAIN_CSV)
    write_csv(val_df, VAL_CSV)
    write_csv(test_df, TEST_CSV)

    print(f"Wrote {len(dataset)} rows to {DATASET_CSV}")
    print(
        "Split counts:",
        f"train={len(train_df)}",
        f"val={len(val_df)}",
        f"test={len(test_df)}",
    )

    if missing_labels:
        print("Missing labels for images:", sorted(missing_labels))

    if orphan_labels:
        print("Labels without matching images:", orphan_labels)


if __name__ == "__main__":
    main()
