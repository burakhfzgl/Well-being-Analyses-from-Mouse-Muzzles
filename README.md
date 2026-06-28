
1. Project overview
2. Directory structure
3. Setup
   pip install -r requirements.txt
4. Data preparation
   python -m src.data.build_labels
5. Training
   python -m src.training.training_modified
6. Evaluation
   python -m src.evaluation.experiment_modified
7. Visualization
   python -m src.visualization.saliency_map


## Project Structure
```text
PROJECT_CV/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── mouse-mgs-2026-06-04/
│   │   └── mouse_dataset/
│   └── processed/
│       └── valid_images/
│
├── src/
│   ├── data/
│   │   ├── build_labels.py
│   │   └── filter_valid_images.py
│   ├── models/
│   │   ├── model.py
│   │   └── mouseDataset.py
│   ├── training/
│   │   ├── training.py
│   │   └── training_modified.py
│   ├── evaluation/
│   │   ├── experiment.py
│   │   ├── experiment_modified.py
│   │   └── generate_mouse_dataset_report.py
│   └── visualization/
│       ├── gradcam_functions.py
│       ├── saliency_map.py
│       ├── visualization_functions.py
│       └── imag_brick_wall.py
│
├── notebooks/
│   ├── data_analyse.ipynb
│   ├── black_box_cut.ipynb
│   ├── well_being_classification.ipynb
│   ├── DLC_Training.ipynb
│   └── final_report_requirements_notes.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── gradcam/
│   └── models/
│       ├── best_full_image_model.pt
│       └── best_muzzle_image_model_v1.pt
│
├── docs/
│   └── paper/
│
└── Makefile# cv_software
```