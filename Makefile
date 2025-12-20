.PHONY: setup smoke quickstart eval test

VENV ?= .venv
PY := $(VENV)/bin/python

setup:
	python -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

smoke:
	$(PY) -m beqcritic.smoke

quickstart:
	bash scripts/run_quickstart.sh

eval:
	$(PY) -m beqcritic.evaluate_ab \
	  --candidates runs/quickstart/proofnetverif_test_candidates.jsonl \
	  --selections-a runs/quickstart/proofnetverif_test_selection_selfbleu.jsonl --a-name selfbleu \
	  --selections-b runs/quickstart/proofnetverif_test_selection_beqcritic.jsonl --b-name beqcritic \
	  --bootstrap 5000 --seed 0 \
	  --timing runs/quickstart/timing.txt

test:
	$(PY) -m unittest discover -s tests -q
