# Contributing

Thank you for improving this project. Keep changes small, readable, and reproducible.

## Development Guidelines

- Put reusable logic in `src/`.
- Keep notebooks thin and explanatory.
- Do not commit datasets, checkpoints, generated figures, or local junctions.
- Prefer explicit imports over wildcard imports.
- Add docstrings and type hints for public functions.
- Run the setup check and a smoke training pass before sharing changes.

## Validation Commands

```powershell
python scripts/check_setup.py
python scripts/train.py --epochs 1 --batch-size 8 --image-size 128 --no-pretrained --limit 80
python -m compileall src scripts
```

## Data Policy

The repository is code-only. Dataset files belong under `dataset/mouse_dataset/` locally and are ignored by Git. Document any new expected files in `dataset/README.md` and `docs/dataset.md`.
