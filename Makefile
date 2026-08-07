PYTHON_BIN ?= python3.12
PYTHON ?= .venv/bin/python
RUN_DIR ?= artifacts/runs/sample-smoke
PUBLIC_RUN_DIR ?= artifacts/runs/binance-public-sample
SMOKE_CONFIG ?= configs/smoke.toml
PUBLIC_CONFIG ?= configs/public_sample.toml
PUBLIC_INGESTION_MANIFEST ?=
PUBLIC_INGESTION_MANIFEST_SHA256 ?=

.PHONY: setup download-sample validate-data validate-public-data smoke check-smoke test reproduce-sample reproduce-public-sample verify-run verify-public-run report report-public dashboard dashboard-public lint typecheck check

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

check-smoke:
	@check_root="$$(mktemp -d)"; \
	trap 'rm -rf "$$check_root"' EXIT; \
	$(PYTHON) -m microstructure.cli reproduce --config $(SMOKE_CONFIG) --run-dir "$$check_root/run"

test:
	$(PYTHON) -m pytest

reproduce-sample: smoke

reproduce-public-sample:
	@test -n "$(PUBLIC_INGESTION_MANIFEST)" || { echo "PUBLIC_INGESTION_MANIFEST is required"; exit 2; }
	@test -n "$(PUBLIC_INGESTION_MANIFEST_SHA256)" || { echo "PUBLIC_INGESTION_MANIFEST_SHA256 is required"; exit 2; }
	$(PYTHON) -m microstructure.cli reproduce --config $(PUBLIC_CONFIG) --run-dir $(PUBLIC_RUN_DIR) --ingestion-manifest $(PUBLIC_INGESTION_MANIFEST) --ingestion-manifest-sha256 $(PUBLIC_INGESTION_MANIFEST_SHA256)

verify-run:
	$(PYTHON) -m microstructure.cli verify --run-dir $(RUN_DIR)

verify-public-run:
	$(PYTHON) -m microstructure.cli verify --run-dir $(PUBLIC_RUN_DIR)

report:
	$(PYTHON) -m microstructure.cli report --run-dir $(RUN_DIR)

report-public:
	$(PYTHON) -m microstructure.cli report --run-dir $(PUBLIC_RUN_DIR)

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py -- --run-dir $(RUN_DIR)

dashboard-public:
	$(PYTHON) -m streamlit run dashboard/app.py -- --run-dir $(PUBLIC_RUN_DIR)

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src/microstructure

check: lint typecheck test check-smoke
