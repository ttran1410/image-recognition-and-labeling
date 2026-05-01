# Image Recognition & Object Labeling - Quick Start

This repository contains a modular PyTorch object detection pipeline with separate training and evaluation entry points. The README covers setup, dataset preparation, training, and the final evaluation demo.

**Prerequisites**
- **Python:** 3.8+ recommended
- **Virtual env:** create one and activate it
- **Dependencies:** see `requirements.txt`

**Install**
- Create and activate a virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
```

- Upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

- Install PyTorch:

GPU install (NVIDIA, recommended). If you want training to run on GPU, install a CUDA-enabled PyTorch build. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU-only install:

```bash
pip install torch torchvision torchaudio
```

Apple Silicon / MPS install:

```bash
pip install torch torchvision torchaudio
```

Use `--device mps` to force Apple Metal, or `--device auto` to let the script pick `mps` when CUDA is unavailable and MPS is available.

- Install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

- Verify that PyTorch can see your GPU before training:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu detected')"
```

Expected for GPU setup:
- `torch.version.cuda` is not `None`
- `torch.cuda.is_available()` returns `True`

Expected for MPS setup:

```bash
python -c "import torch; print(torch.__version__); print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

- The command prints `True`
- `python main.py ... --device auto` should report `Using device: mps`

On Apple Silicon, `torch`, `torchvision`, and `torchaudio` are typically installed from PyPI as shown above. If you hit an unsupported-op error on `mps`, rerun with `--device cpu`.

**Prepare dataset**
- If you need to regenerate CSV splits, run `data_preparation.py`.
- The script writes `dataset.csv`, `train.csv`, `val.csv`, and `test.csv` under `data/csvs`.
- The split is deterministic and uses a fixed seed.
- The current split is approximately `70%` train, `15%` validation, and `15%` test.
- The `test.csv` split is intended to stay untouched until the final evaluation demo.

**Smoke test (recommended first run)**
- Runs a single epoch on CPU with a very small batch to verify end-to-end behavior and checkpoint saving.

```bash
python main.py --epochs 1 --batch_size 2 --train_csv data/csvs/train.csv \
  --val_csv data/csvs/val.csv --device cpu --out_dir sessions/smoke_test --num_workers 0
```

- GPU smoke test:

```bash
python main.py --epochs 1 --batch_size 2 --train_csv data/csvs/train.csv \
  --val_csv data/csvs/val.csv --device cuda --out_dir sessions/gpu_smoke_test --num_workers 0
```

**Full training (example)**

- `--device auto` prefers CUDA, then MPS, and falls back to CPU.
- If you want the run to fail instead of silently using CPU, use `--device cuda`.

```bash
python main.py --epochs 10 --batch_size 8 --train_csv data/csvs/train.csv \
  --val_csv data/csvs/val.csv --device auto --out_dir sessions/run1 --num_workers 8
```

**Common options**
- **--device:** `auto`, `cpu`, `cuda`, or `mps`. Use `auto` to prefer the best available device.
- **--batch_size:** keep small (8-64) for limited GPU memory.
- **--num_workers:** number of dataloader workers (0 on Windows if issues occur).
- **--out_dir / --output_dir:** where checkpoints, logs, and visual outputs are saved.

**Outputs**
- Checkpoints and session data are saved under the directory specified by `--out_dir` (default `./sessions`).
- The trainer currently writes `best_model.pth`.
- Each finished training run also writes `training_log.txt` and `learning_curve.png` in the session folder.
- Final evaluation writes annotated prediction images and a contact sheet into `evaluation_outputs/` inside the session folder.

**Final evaluation**

Run the separate evaluation script on the untouched test split after training:

```bash
python evaluate.py --checkpoint sessions/run1/best_model.pth --test_csv data/csvs/test.csv \
  --device auto --max_images 10
```

This saves annotated predictions to `sessions/run1/evaluation_outputs/` by default, along with a contact sheet for quick review.

**Key files**
- Source and helpers:
  - [args.py](args.py)
  - [data_preparation.py](data_preparation.py)
  - [dataset.py](dataset.py)
  - [evaluate.py](evaluate.py)
  - [model.py](model.py)
  - [plot_learning_curve.py](plot_learning_curve.py)
  - [utils.py](utils.py)
  - [trainer.py](trainer.py)
  - [main.py](main.py)
