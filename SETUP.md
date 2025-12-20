# Setup (reproducible)

Tested with Python `>=3.10` (see `pyproject.toml`).

## 1) Create a virtualenv and install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Optional (BEq+ paper metric; requires `lean-interact`):

```bash
python -m pip install -e '.[beqplus]'
```

## 2) Smoke check

```bash
python -m beqcritic.smoke
```

## 3) Run the end-to-end Quickstart

```bash
bash scripts/run_quickstart.sh
```

Outputs are written under `runs/quickstart/`.

## Convenience targets

```bash
make setup
make smoke
make quickstart
make eval
make test
```
