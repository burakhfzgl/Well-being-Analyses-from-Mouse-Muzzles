# Dataset Layout

Keep Mouse Grimace Faces data locally (not committed):

```text
dataset/mouse_dataset/
├── MouseGrimaceFaces_main.csv
├── MouseGrimaceFaces_mgs.csv
├── images_mgs/
├── images_perfect/
└── images_perfect_organized/   # built by: python main.py --mode prepare
```

```bash
python main.py --mode prepare
python main.py --mode check
```
