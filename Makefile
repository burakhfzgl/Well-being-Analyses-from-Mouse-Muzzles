PYTHON ?= python

.PHONY: install check prepare dry-run part1 part2 summarize diagrams tables figures gradcam curves

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) main.py --mode check

prepare:
	$(PYTHON) main.py --mode prepare

dry-run:
	$(PYTHON) main.py --mode train --part all --dry-run

part1:
	$(PYTHON) main.py --mode train --part part1

part2:
	$(PYTHON) main.py --mode train --part part2

summarize:
	$(PYTHON) main.py --mode summarize

diagrams:
	$(PYTHON) main.py --mode diagrams

tables:
	$(PYTHON) main.py --mode tables

figures:
	$(PYTHON) main.py --mode figures

gradcam:
	$(PYTHON) main.py --mode gradcam

curves:
	$(PYTHON) main.py --mode curves
