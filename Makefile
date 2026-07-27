PYTHON ?= python

.PHONY: install install-cuda check prepare dry-run part1 part2 summarize diagrams tables figures gradcam curves

install:
	$(PYTHON) -m pip install -r requirements.txt

install-cuda:
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) scripts/data_processing/check_setup.py

prepare:
	$(PYTHON) scripts/data_processing/prepare_organized_dataset.py

dry-run:
	$(PYTHON) scripts/training/run_article_experiments.py --part all --dry-run

part1:
	$(PYTHON) scripts/training/run_article_experiments.py --part part1

part2:
	$(PYTHON) scripts/training/run_article_experiments.py --part part2

summarize:
	$(PYTHON) scripts/evaluation/summarize_article_experiments.py

diagrams:
	$(PYTHON) scripts/analysis/generate_report_diagrams.py

tables:
	$(PYTHON) scripts/analysis/generate_result_table_figures.py

figures:
	$(PYTHON) scripts/analysis/generate_qualitative_visualizations.py --part all

gradcam:
	$(PYTHON) scripts/analysis/generate_gradcam_visualizations.py

curves:
	$(PYTHON) scripts/analysis/generate_training_curves.py