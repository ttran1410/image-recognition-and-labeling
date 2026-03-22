# Image Recognition & Object Labeling - Quick Start

This repository trains a small PyTorch object detector on the provided dataset. The README contains quick setup and run commands to start training and perform a smoke test.

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

- Alternatively, using `conda`:

```bash
conda create -n ai_proj python=3.9 -y
conda activate ai_proj
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Note: plain `torch==...` in `requirements.txt` may install a CPU-only build on Windows, so PyTorch is documented separately here as a platform-specific dependency.
On Apple Silicon, `torch`, `torchvision`, and `torchaudio` are typically installed from PyPI as shown above. If you hit an unsupported-op error on `mps`, rerun with `--device cpu`.

**Prepare dataset**
- If you need to regenerate CSV splits, run `data_preparation.py`. The code expects CSVs under `data/csvs` by default.

**Smoke test (recommended first run)**
- Runs a single epoch on CPU with a very small batch to verify end-to-end behavior and checkpoint saving.

```bash
python main.py --epochs 1 --batch_size 2 --train_csv data/csvs/train_data.csv \
  --val_csv data/csvs/val_data.csv --device cpu --out_dir sessions/smoke_test --num_workers 0
```

- GPU smoke test:

```bash
python main.py --epochs 1 --batch_size 2 --train_csv data/csvs/train_data.csv \
  --val_csv data/csvs/val_data.csv --device cuda --out_dir sessions/gpu_smoke_test --num_workers 0
```

**Full training (example)**

- `--device auto` prefers CUDA, then MPS, and falls back to CPU.
- If you want the run to fail instead of silently using CPU, use `--device cuda`.

```bash
python main.py --epochs 10 --batch_size 8 --train_csv data/csvs/train_data.csv \
  --val_csv data/csvs/val_data.csv --device auto --out_dir sessions/run1 --num_workers 8
```

**Common options**
- **--device:** `auto`, `cpu`, or `cuda`. Use `auto` to prefer GPU if available.
- **--batch_size:** keep small (8-64) for limited GPU memory.
- **--num_workers:** number of dataloader workers (0 on Windows if issues occur).

**Outputs**
- Checkpoints and session data are saved under the directory specified by `--out_dir` (default `./sessions`).
- The trainer currently writes `best_model.pth`.
- Each finished training run also writes `training_log.txt` and `learning_curve.png` in the session folder.

**Key files**
- Source and helpers:
  - [args.py](args.py)
  - [dataset.py](dataset.py)
  - [model.py](model.py)
  - [utils.py](utils.py)
  - [trainer.py](trainer.py)
  - [main.py](main.py)
