# Installation

## Recommended Conda environment

```bash
conda env create -f environment.yml
conda activate imas
pip install -e .
```

`environment.yml` uses Python 3.11 and a CUDA 12.1 PyTorch environment as a
portable starting point. Change or remove `pytorch-cuda` to match the target
machine. For CPU-only utility and unit-test use, install the CPU PyTorch build.

## Pip installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[full]"
```

## Development tools

```bash
pip install -e ".[full,dev]"
pytest -q
python -m build
```
