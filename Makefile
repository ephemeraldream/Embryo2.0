.PHONY: *
VENV=.venv
PYTHON=$(VENV)/bin/python3
DEVICE=gpu


venv:
	#python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

pre_commit_install:
	@echo "=== Installing pre-commit ==="
	$(PYTHON) -m pre_commit install


install_all: venv
	@echo "=== Installing common dependencies ==="
	$(PYTHON) -m pip install -r requirements/requirements-$(DEVICE).txt

	make pre_commit_install

run_training:
	$(PYTHON) -m src.train


install_cuda:
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


