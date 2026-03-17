# Image Recognition & Object Labeling — Quick Start

This repository trains a small PyTorch object detector on the provided dataset. The README contains quick setup and run commands to start training and perform a smoke test.

**Prerequisites**
- **Python:** 3.8+ recommended
- **Virtual env:** create one and activate it
- **Dependencies:** see `requirements.txt`

**Install**
- Create and activate a virtual environment, then install dependencies. Platform-specific commands:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, using `conda`:

```bash
conda create -n ai_proj python=3.9 -y
conda activate ai_proj
pip install --upgrade pip
pip install -r requirements.txt
```

**Prepare dataset**
- If you need to regenerate CSV splits, run `data_preparation.py`. The code expects CSVs under `data/csvs` by default.

**Smoke test (recommended first run)**
- Runs a single epoch on CPU with a very small batch to verify end-to-end behavior and checkpoint saving.

```bash
python main.py --epochs 1 --batch_size 2 --train_csv data/csvs/train_data.csv \
  --val_csv data/csvs/val_data.csv --device cpu --out_dir sessions/smoke_test --num_workers 0
```

**Full training (example)**

```bash
python main.py --epochs 20 --batch_size 4 --train_csv data/csvs/train.csv \
  --val_csv data/csvs/val.csv --device auto --out_dir sessions/run1 --num_workers 2
```

**Resume training**

```bash
python main.py --resume sessions/run1/checkpoint_last.pth --epochs 30
```

**Evaluation**

Run the evaluation script on a test CSV using a saved checkpoint (best or last). This will compute basic detection metrics and optionally save visualizations.

```bash
python evaluate.py --test_csv data/csvs/test.csv --resume sessions/run1/checkpoint_best.pth \
  --device cpu --out_dir sessions/eval_run --score_thresh 0.5 --visualize --num_workers 0
```

The script writes a `metrics.json` file and (if `--visualize`) prediction images under the specified `--out_dir`.

**Common options**
- **--device:** `auto`, `cpu`, or `cuda`. Use `auto` to prefer GPU if available.
- **--batch_size:** keep small (2–8) for limited GPU memory.
- **--num_workers:** number of dataloader workers (0 on Windows if issues occur).
- **--score_thresh:** prediction score threshold for visualization (default 0.5).
- **--visualize:** add this flag to save prediction visualizations (when implemented).

**Outputs**
- Checkpoints and session data are saved under the directory specified by `--out_dir` (default `./sessions`). The trainer writes `checkpoint_last.pth` and `checkpoint_best.pth`.

**Key files**
- Source and helpers:
  - [args.py](args.py)
  - [dataset.py](dataset.py)
  - [model.py](model.py)
  - [utils.py](utils.py)
  - [trainer.py](trainer.py)
  - [main.py](main.py)

**Notes & next steps**
- `evaluate.py` is implemented and computes TP/FP/FN, precision, recall, F1 and mean IoU; visualizations are optional via `--visualize`.
