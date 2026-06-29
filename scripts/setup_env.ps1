# Hand2Body environment setup (Windows / PowerShell).
# System Python is 3.14 (no torch wheels). We use a uv-managed Python 3.12 venv.
# GPU: RTX 5080 Laptop (Blackwell) -> requires the cu128 torch build.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# 1. uv (installs into the system Python; harmless if already present)
python -m pip install --quiet uv

# 2. Python 3.12 venv
python -m uv venv --python 3.12 .venv

# 3. core deps + editable install
python -m uv pip install --python .venv -e ".[dev]"

# 4. torch for Blackwell (cu128) — ~2.6 GB download
python -m uv pip install --python .venv --index-url https://download.pytorch.org/whl/cu128 torch

# 5. training extras (smplx, trimesh, tqdm, tensorboard)
python -m uv pip install --python .venv -e ".[train]"

# 6. sanity
$env:PYTHONPATH = (Get-Location).Path
& ".venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
& ".venv\Scripts\python.exe" -m pytest -q

Write-Host "`nDone. Activate with:  .venv\Scripts\Activate.ps1"
