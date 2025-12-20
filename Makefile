.PHONY: setup dev smoke quickstart eval results test

VENV ?= .venv
PYTHON ?= python
PY := $(VENV)/bin/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

dev: setup
	$(PY) -m pip install -e '.[dev]'

smoke:
	$(PY) -m beqcritic.smoke

quickstart:
	PYTHON=$(PY) bash scripts/run_quickstart.sh

eval:
	$(PY) -m beqcritic.evaluate_ab \
	  --candidates runs/quickstart/proofnetverif_test_candidates.jsonl \
	  --selections-a runs/quickstart/proofnetverif_test_selection_selfbleu.jsonl --a-name selfbleu \
	  --selections-b runs/quickstart/proofnetverif_test_selection_beqcritic.jsonl --b-name beqcritic \
	  --bootstrap 5000 --seed 0 \
	  --timing runs/quickstart/timing.txt

results: quickstart
	$(PY) scripts/generate_results.py --run-dir runs/quickstart --output results/results.md --bootstrap 2000 --seed 0

test: dev
	$(PY) -m pytest -q
