# Reproducibility

This project is designed for transparent, repeatable computer vision experiments.

## Splitting Strategy

The default split is group-aware:

```text
group = subset + "_" + id
```

This reduces the risk that images from the same mouse identity appear in both training and validation sets. Avoid random image-level splits unless you are intentionally testing a baseline and clearly document the leakage risk.

## Seeds

The CLI scripts set seeds for:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA

Some GPU kernels can still be nondeterministic. Use `--deterministic` when you prefer reproducibility over speed.

## Device Selection

The project automatically chooses the best available PyTorch device:

1. CUDA
2. Apple MPS
3. CPU

For NVIDIA GPUs on Windows, install a CUDA-enabled PyTorch wheel. Example:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Recommended Smoke Test

Run a short, non-pretrained training pass before long experiments:

```powershell
python scripts/train.py --epochs 1 --batch-size 8 --image-size 128 --no-pretrained --limit 80
```

This confirms imports, labels, image loading, model construction, device transfer, and checkpoint writing.

## Checkpoint Metadata

Training checkpoints include:

- model state
- label mapping
- epoch
- validation metric
- training configuration

Store generated checkpoints under `outputs/models/`; this directory is ignored by Git.
