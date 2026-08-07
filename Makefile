PYTHON_BIN ?= python3.12
PYTHON ?= .venv/bin/python
RUN_DIR ?= artifacts/runs/sample-smoke
SMOKE_CONFIG ?= configs/smoke.toml
PUBLIC_CONFIG ?= configs/public_sample.toml

.PHONY: setup download-sample validate-data validate-public-data smoke test reproduce-sample verify-run report dashboard lint typecheck check

setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --locked --extra dev; \
	else \
		"$(PYTHON_BIN)" -m venv .venv; \
		.venv/bin/python -m pip install --upgrade pip; \
		.venv/bin/pip install -e ".[dev]"; \
	fi

download-sample:
	$(PYTHON) -m microstructure.cli ingest --config $(PUBLIC_CONFIG)

validate-data:
	$(PYTHON) -m microstructure.cli validate --config $(SMOKE_CONFIG)

validate-public-data:
	$(PYTHON) -m microstructure.cli validate --config $(PUBLIC_CONFIG)

smoke:
	$(PYTHON) -m microstructure.cli reproduce --config $(SMOKE_CONFIG) --run-dir $(RUN_DIR)

test:
	$(PYTHON) -m pytest

reproduce-sample: smoke

verify-run:
	$(PYTHON) -m microstructure.cli verify --run-dir $(RUN_DIR)

report:
	$(PYTHON) -m microstructure.cli report --run-dir $(RUN_DIR)

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py -- --run-dir $(RUN_DIR)

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src/microstructure

check: lint typecheck test smoke
