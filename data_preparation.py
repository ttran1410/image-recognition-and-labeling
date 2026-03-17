from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Resolve project paths
root = Path(__file__).resolve().parent;
images_dir = root / "data" / "images";
labels_dir = root / "data" / "labels";
output_csv = root / "data" / "csvs" / "dataset.csv";

# Preload label stems to check matches and detect orphan labels
label_stems = {p.stem for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt"};

# Track missing labels and count matched pairs
missing_labels = [];
matched_count = 0;

# Build rows in memory and write with pandas
rows = [];
for img in sorted(p for p in images_dir.iterdir() if p.is_file() and not p.name.startswith(".")):
    # If no label exists for this image, record it and skip
    if img.stem not in label_stems:
        missing_labels.append(img.stem);
        continue
    # Add a valid image/label pair
    rows.append({
        "images": f"data/images/{img.name}",
        "labels": f"data/labels/{img.stem}.txt",
    });
    matched_count += 1;
    # Remove matched label so remaining ones are orphan labels
    label_stems.discard(img.stem);

output_csv.parent.mkdir(parents=True, exist_ok=True);
pd.DataFrame(rows, columns=["images", "labels"]).to_csv(output_csv, index=False);

# Report results
print(f"Wrote {matched_count} rows to {output_csv}");
if missing_labels:
    print("Missing labels for images:", sorted(missing_labels));

if label_stems:
    print("Labels without matching images:", sorted(label_stems));

dataset = pd.read_csv("data/csvs/dataset.csv")
chunk_size = 1000

train_data, valdata = train_test_split(dataset, test_size=0.3)
train_data.to_csv("data/csvs/train_data.csv", index=False)
valdata.to_csv("data/csvs/val_data.csv", index=False)
