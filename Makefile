.PHONY: setup dev smoke quickstart eval results test verify verify-report paper-rollup paper-check source-hashes

VENV ?= .venv
PYTHON ?= python
PY := $(VENV)/bin/python
REPRO_REPORT ?= runs/reproducibility_report.json
NLVERIFIER_RESULTS ?= results
NLVERIFIER_ROLLUP_JSON ?= results/nlverifier_paper_metrics.json
NLVERIFIER_ROLLUP_MD ?= results/nlverifier_paper_metrics.md
NLVERIFIER_TABLE_TEX ?= paper/generated/nlverifier_main_table.tex

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
	  --timing runs/quickstart/timing.txt

results: quickstart
	$(PY) scripts/generate_results.py --run-dir runs/quickstart --output results/results.md

test: dev
	$(PY) -m pytest -q

verify: dev
	$(PY) scripts/verify_reproducibility.py

verify-report: dev
	$(PY) scripts/verify_reproducibility.py --report-json $(REPRO_REPORT)

paper-rollup:
	$(PYTHON) scripts/summarize_nlverifier_paper_metrics.py \
	  --results-dir $(NLVERIFIER_RESULTS) \
	  --output-json $(NLVERIFIER_ROLLUP_JSON) \
	  --output-md $(NLVERIFIER_ROLLUP_MD) \
	  --output-tex $(NLVERIFIER_TABLE_TEX)

paper-check:
	$(PYTHON) scripts/summarize_nlverifier_paper_metrics.py \
	  --results-dir $(NLVERIFIER_RESULTS) \
	  --output-json $(NLVERIFIER_ROLLUP_JSON) \
	  --output-md $(NLVERIFIER_ROLLUP_MD) \
	  --output-tex $(NLVERIFIER_TABLE_TEX) \
	  --check

source-hashes:
	$(PYTHON) scripts/summarize_nlverifier_paper_metrics.py \
	  --output-json $(NLVERIFIER_ROLLUP_JSON) \
	  --verify-source-hashes
