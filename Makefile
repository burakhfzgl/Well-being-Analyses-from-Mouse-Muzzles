PYTHON ?= python

.PHONY: install install-cuda check train smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

install-cuda:
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) scripts/check_setup.py

smoke:
	$(PYTHON) scripts/train.py --epochs 1 --batch-size 8 --image-size 128 --no-pretrained --limit 80

train:
	$(PYTHON) scripts/train.py