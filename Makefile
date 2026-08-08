PYTHON_BIN ?= python3.12
PYTHON ?= .venv/bin/python
RUN_DIR ?= artifacts/runs/sample-smoke
PUBLIC_RUN_DIR ?= artifacts/runs/binance-public-sample
SMOKE_CONFIG ?= configs/smoke.toml
PUBLIC_CONFIG ?= configs/public_sample.toml
PUBLIC_INGESTION_MANIFEST ?=
PUBLIC_INGESTION_MANIFEST_SHA256 ?=
M8_CONFIG ?= configs/m8_multidate_trade_study.toml
M8_DATA_ROOT ?= data/m8
M8_RUN_DIR ?= artifacts/runs/binance-m8-multidate
M8_RAW_MANIFEST ?=
M8_RAW_MANIFEST_SHA256 ?=
M8_L2_CONFIG ?= configs/m8_l2_capture_study.toml
M8_L2_ANALYSIS_CONFIG ?= configs/m8_l2_analysis.toml
M8_L2_DATA_ROOT ?= data/m8_l2
M8_L2_SESSION_DATE ?=
M8_L2_BUNDLE_DIR ?=
M8_L2_TRAIN_BUNDLE_DIR ?=
M8_L2_TRAIN_MANIFEST_SHA256 ?=
M8_L2_TRAIN_CHECKSUMS_SHA256 ?=
M8_L2_VALIDATION_BUNDLE_DIR ?=
M8_L2_VALIDATION_MANIFEST_SHA256 ?=
M8_L2_VALIDATION_CHECKSUMS_SHA256 ?=
M8_L2_DEVELOPMENT_LOCK_DIR ?=
M8_L2_DEVELOPMENT_LOCK_SHA256 ?=
M8_L2_PRIMARY_BUNDLE_DIR ?=
M8_L2_PRIMARY_MANIFEST_SHA256 ?=
M8_L2_PRIMARY_CHECKSUMS_SHA256 ?=
M8_L2_REPLICATION_BUNDLE_DIR ?=
M8_L2_REPLICATION_MANIFEST_SHA256 ?=
M8_L2_REPLICATION_CHECKSUMS_SHA256 ?=
M8_L2_RUN_DIR ?= artifacts/runs/binance-m8-live-l2
M8_L2_RUN_MANIFEST_SHA256 ?=
M8_L2_RUN_CHECKSUMS_SHA256 ?=
M8_L2_REPORT_DIR ?= artifacts/runs/binance-m8-live-l2-reports

.PHONY: setup download-sample download-m8 capture-m8-l2-session verify-m8-l2-session lock-m8-l2-development verify-m8-l2-development-lock reproduce-m8-l2 verify-m8-l2-run report-m8-l2 validate-data validate-public-data smoke check-smoke test reproduce-sample reproduce-public-sample reproduce-m8 verify-run verify-public-run verify-m8-run report report-public report-m8 dashboard dashboard-public lint typecheck check

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

download-m8:
	$(PYTHON) -m microstructure.cli acquire-m8 --config "$(M8_CONFIG)" --output-root "$(M8_DATA_ROOT)"

capture-m8-l2-session:
	@test -n "$(M8_L2_SESSION_DATE)" || { echo "M8_L2_SESSION_DATE is required"; exit 2; }
	@status=0; $(PYTHON) -m microstructure.cli capture-m8-l2-session --config "$(M8_L2_CONFIG)" --date "$(M8_L2_SESSION_DATE)" --output-root "$(M8_L2_DATA_ROOT)" || status=$$?; if [ "$$status" -eq 1 ]; then exit 0; fi; exit "$$status"

verify-m8-l2-session:
	@test -n "$(M8_L2_BUNDLE_DIR)" || { echo "M8_L2_BUNDLE_DIR is required"; exit 2; }
	$(PYTHON) -m microstructure.cli verify-m8-l2-session --config "$(M8_L2_CONFIG)" --bundle-dir "$(M8_L2_BUNDLE_DIR)"

lock-m8-l2-development:
	@test -n "$(M8_L2_TRAIN_BUNDLE_DIR)" || { echo "M8_L2_TRAIN_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_MANIFEST_SHA256)" || { echo "M8_L2_TRAIN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" || { echo "M8_L2_TRAIN_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_BUNDLE_DIR)" || { echo "M8_L2_VALIDATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_MANIFEST_SHA256)" || { echo "M8_L2_VALIDATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" || { echo "M8_L2_VALIDATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_DIR)" || { echo "M8_L2_DEVELOPMENT_LOCK_DIR is required"; exit 2; }
	@status=0; $(PYTHON) -m microstructure.cli lock-m8-l2-development --capture-config "$(M8_L2_CONFIG)" --analysis-config "$(M8_L2_ANALYSIS_CONFIG)" --train-bundle-dir "$(M8_L2_TRAIN_BUNDLE_DIR)" --train-manifest-sha256 "$(M8_L2_TRAIN_MANIFEST_SHA256)" --train-checksums-sha256 "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" --validation-bundle-dir "$(M8_L2_VALIDATION_BUNDLE_DIR)" --validation-manifest-sha256 "$(M8_L2_VALIDATION_MANIFEST_SHA256)" --validation-checksums-sha256 "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" --lock-dir "$(M8_L2_DEVELOPMENT_LOCK_DIR)" || status=$$?; if [ "$$status" -eq 1 ]; then exit 0; fi; exit "$$status"

verify-m8-l2-development-lock:
	@test -n "$(M8_L2_TRAIN_BUNDLE_DIR)" || { echo "M8_L2_TRAIN_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_MANIFEST_SHA256)" || { echo "M8_L2_TRAIN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" || { echo "M8_L2_TRAIN_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_BUNDLE_DIR)" || { echo "M8_L2_VALIDATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_MANIFEST_SHA256)" || { echo "M8_L2_VALIDATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" || { echo "M8_L2_VALIDATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_DIR)" || { echo "M8_L2_DEVELOPMENT_LOCK_DIR is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" || { echo "M8_L2_DEVELOPMENT_LOCK_SHA256 is required"; exit 2; }
	@status=0; $(PYTHON) -m microstructure.cli verify-m8-l2-development-lock --capture-config "$(M8_L2_CONFIG)" --analysis-config "$(M8_L2_ANALYSIS_CONFIG)" --train-bundle-dir "$(M8_L2_TRAIN_BUNDLE_DIR)" --train-manifest-sha256 "$(M8_L2_TRAIN_MANIFEST_SHA256)" --train-checksums-sha256 "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" --validation-bundle-dir "$(M8_L2_VALIDATION_BUNDLE_DIR)" --validation-manifest-sha256 "$(M8_L2_VALIDATION_MANIFEST_SHA256)" --validation-checksums-sha256 "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" --lock-dir "$(M8_L2_DEVELOPMENT_LOCK_DIR)" --development-lock-sha256 "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" || status=$$?; if [ "$$status" -eq 1 ]; then exit 0; fi; exit "$$status"

reproduce-m8-l2:
	@test -n "$(M8_L2_TRAIN_BUNDLE_DIR)" || { echo "M8_L2_TRAIN_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_MANIFEST_SHA256)" || { echo "M8_L2_TRAIN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" || { echo "M8_L2_TRAIN_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_BUNDLE_DIR)" || { echo "M8_L2_VALIDATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_MANIFEST_SHA256)" || { echo "M8_L2_VALIDATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" || { echo "M8_L2_VALIDATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_DIR)" || { echo "M8_L2_DEVELOPMENT_LOCK_DIR is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" || { echo "M8_L2_DEVELOPMENT_LOCK_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_BUNDLE_DIR)" || { echo "M8_L2_PRIMARY_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_MANIFEST_SHA256)" || { echo "M8_L2_PRIMARY_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_CHECKSUMS_SHA256)" || { echo "M8_L2_PRIMARY_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_BUNDLE_DIR)" || { echo "M8_L2_REPLICATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_MANIFEST_SHA256)" || { echo "M8_L2_REPLICATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_CHECKSUMS_SHA256)" || { echo "M8_L2_REPLICATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@status=0; $(PYTHON) -m microstructure.cli reproduce-m8-l2 --capture-config "$(M8_L2_CONFIG)" --analysis-config "$(M8_L2_ANALYSIS_CONFIG)" --train-bundle-dir "$(M8_L2_TRAIN_BUNDLE_DIR)" --train-manifest-sha256 "$(M8_L2_TRAIN_MANIFEST_SHA256)" --train-checksums-sha256 "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" --validation-bundle-dir "$(M8_L2_VALIDATION_BUNDLE_DIR)" --validation-manifest-sha256 "$(M8_L2_VALIDATION_MANIFEST_SHA256)" --validation-checksums-sha256 "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" --development-lock-dir "$(M8_L2_DEVELOPMENT_LOCK_DIR)" --development-lock-sha256 "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" --primary-bundle-dir "$(M8_L2_PRIMARY_BUNDLE_DIR)" --primary-manifest-sha256 "$(M8_L2_PRIMARY_MANIFEST_SHA256)" --primary-checksums-sha256 "$(M8_L2_PRIMARY_CHECKSUMS_SHA256)" --replication-bundle-dir "$(M8_L2_REPLICATION_BUNDLE_DIR)" --replication-manifest-sha256 "$(M8_L2_REPLICATION_MANIFEST_SHA256)" --replication-checksums-sha256 "$(M8_L2_REPLICATION_CHECKSUMS_SHA256)" --run-dir "$(M8_L2_RUN_DIR)" || status=$$?; if [ "$$status" -eq 1 ]; then exit 0; fi; exit "$$status"

verify-m8-l2-run:
	@test -n "$(M8_L2_TRAIN_BUNDLE_DIR)" || { echo "M8_L2_TRAIN_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_MANIFEST_SHA256)" || { echo "M8_L2_TRAIN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" || { echo "M8_L2_TRAIN_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_BUNDLE_DIR)" || { echo "M8_L2_VALIDATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_MANIFEST_SHA256)" || { echo "M8_L2_VALIDATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" || { echo "M8_L2_VALIDATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_DIR)" || { echo "M8_L2_DEVELOPMENT_LOCK_DIR is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" || { echo "M8_L2_DEVELOPMENT_LOCK_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_BUNDLE_DIR)" || { echo "M8_L2_PRIMARY_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_MANIFEST_SHA256)" || { echo "M8_L2_PRIMARY_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_CHECKSUMS_SHA256)" || { echo "M8_L2_PRIMARY_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_BUNDLE_DIR)" || { echo "M8_L2_REPLICATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_MANIFEST_SHA256)" || { echo "M8_L2_REPLICATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_CHECKSUMS_SHA256)" || { echo "M8_L2_REPLICATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_RUN_MANIFEST_SHA256)" || { echo "M8_L2_RUN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_RUN_CHECKSUMS_SHA256)" || { echo "M8_L2_RUN_CHECKSUMS_SHA256 is required"; exit 2; }
	@status=0; $(PYTHON) -m microstructure.cli verify-m8-l2-run --capture-config "$(M8_L2_CONFIG)" --analysis-config "$(M8_L2_ANALYSIS_CONFIG)" --train-bundle-dir "$(M8_L2_TRAIN_BUNDLE_DIR)" --train-manifest-sha256 "$(M8_L2_TRAIN_MANIFEST_SHA256)" --train-checksums-sha256 "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" --validation-bundle-dir "$(M8_L2_VALIDATION_BUNDLE_DIR)" --validation-manifest-sha256 "$(M8_L2_VALIDATION_MANIFEST_SHA256)" --validation-checksums-sha256 "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" --development-lock-dir "$(M8_L2_DEVELOPMENT_LOCK_DIR)" --development-lock-sha256 "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" --primary-bundle-dir "$(M8_L2_PRIMARY_BUNDLE_DIR)" --primary-manifest-sha256 "$(M8_L2_PRIMARY_MANIFEST_SHA256)" --primary-checksums-sha256 "$(M8_L2_PRIMARY_CHECKSUMS_SHA256)" --replication-bundle-dir "$(M8_L2_REPLICATION_BUNDLE_DIR)" --replication-manifest-sha256 "$(M8_L2_REPLICATION_MANIFEST_SHA256)" --replication-checksums-sha256 "$(M8_L2_REPLICATION_CHECKSUMS_SHA256)" --run-dir "$(M8_L2_RUN_DIR)" --run-manifest-sha256 "$(M8_L2_RUN_MANIFEST_SHA256)" --run-checksums-sha256 "$(M8_L2_RUN_CHECKSUMS_SHA256)" || status=$$?; if [ "$$status" -eq 1 ]; then exit 0; fi; exit "$$status"

report-m8-l2:
	@test -n "$(M8_L2_TRAIN_BUNDLE_DIR)" || { echo "M8_L2_TRAIN_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_MANIFEST_SHA256)" || { echo "M8_L2_TRAIN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" || { echo "M8_L2_TRAIN_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_BUNDLE_DIR)" || { echo "M8_L2_VALIDATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_MANIFEST_SHA256)" || { echo "M8_L2_VALIDATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" || { echo "M8_L2_VALIDATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_DIR)" || { echo "M8_L2_DEVELOPMENT_LOCK_DIR is required"; exit 2; }
	@test -n "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" || { echo "M8_L2_DEVELOPMENT_LOCK_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_BUNDLE_DIR)" || { echo "M8_L2_PRIMARY_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_MANIFEST_SHA256)" || { echo "M8_L2_PRIMARY_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_PRIMARY_CHECKSUMS_SHA256)" || { echo "M8_L2_PRIMARY_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_BUNDLE_DIR)" || { echo "M8_L2_REPLICATION_BUNDLE_DIR is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_MANIFEST_SHA256)" || { echo "M8_L2_REPLICATION_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_REPLICATION_CHECKSUMS_SHA256)" || { echo "M8_L2_REPLICATION_CHECKSUMS_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_RUN_MANIFEST_SHA256)" || { echo "M8_L2_RUN_MANIFEST_SHA256 is required"; exit 2; }
	@test -n "$(M8_L2_RUN_CHECKSUMS_SHA256)" || { echo "M8_L2_RUN_CHECKSUMS_SHA256 is required"; exit 2; }
	@status=0; $(PYTHON) -m microstructure.cli report-m8-l2 --capture-config "$(M8_L2_CONFIG)" --analysis-config "$(M8_L2_ANALYSIS_CONFIG)" --train-bundle-dir "$(M8_L2_TRAIN_BUNDLE_DIR)" --train-manifest-sha256 "$(M8_L2_TRAIN_MANIFEST_SHA256)" --train-checksums-sha256 "$(M8_L2_TRAIN_CHECKSUMS_SHA256)" --validation-bundle-dir "$(M8_L2_VALIDATION_BUNDLE_DIR)" --validation-manifest-sha256 "$(M8_L2_VALIDATION_MANIFEST_SHA256)" --validation-checksums-sha256 "$(M8_L2_VALIDATION_CHECKSUMS_SHA256)" --development-lock-dir "$(M8_L2_DEVELOPMENT_LOCK_DIR)" --development-lock-sha256 "$(M8_L2_DEVELOPMENT_LOCK_SHA256)" --primary-bundle-dir "$(M8_L2_PRIMARY_BUNDLE_DIR)" --primary-manifest-sha256 "$(M8_L2_PRIMARY_MANIFEST_SHA256)" --primary-checksums-sha256 "$(M8_L2_PRIMARY_CHECKSUMS_SHA256)" --replication-bundle-dir "$(M8_L2_REPLICATION_BUNDLE_DIR)" --replication-manifest-sha256 "$(M8_L2_REPLICATION_MANIFEST_SHA256)" --replication-checksums-sha256 "$(M8_L2_REPLICATION_CHECKSUMS_SHA256)" --run-dir "$(M8_L2_RUN_DIR)" --run-manifest-sha256 "$(M8_L2_RUN_MANIFEST_SHA256)" --run-checksums-sha256 "$(M8_L2_RUN_CHECKSUMS_SHA256)" --output-dir "$(M8_L2_REPORT_DIR)" || status=$$?; if [ "$$status" -eq 1 ]; then exit 0; fi; exit "$$status"

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

reproduce-m8:
	@test -n "$(M8_RAW_MANIFEST)" || { echo "M8_RAW_MANIFEST is required"; exit 2; }
	@test -n "$(M8_RAW_MANIFEST_SHA256)" || { echo "M8_RAW_MANIFEST_SHA256 is required"; exit 2; }
	$(PYTHON) -m microstructure.cli reproduce-m8 --config "$(M8_CONFIG)" --run-dir "$(M8_RUN_DIR)" --raw-manifest "$(M8_RAW_MANIFEST)" --raw-manifest-sha256 "$(M8_RAW_MANIFEST_SHA256)"

verify-run:
	$(PYTHON) -m microstructure.cli verify --run-dir $(RUN_DIR)

verify-public-run:
	$(PYTHON) -m microstructure.cli verify --run-dir $(PUBLIC_RUN_DIR)

verify-m8-run:
	@test -n "$(M8_RAW_MANIFEST)" || { echo "M8_RAW_MANIFEST is required"; exit 2; }
	@test -n "$(M8_RAW_MANIFEST_SHA256)" || { echo "M8_RAW_MANIFEST_SHA256 is required"; exit 2; }
	$(PYTHON) -m microstructure.cli verify-m8 --config "$(M8_CONFIG)" --run-dir "$(M8_RUN_DIR)" --raw-manifest "$(M8_RAW_MANIFEST)" --raw-manifest-sha256 "$(M8_RAW_MANIFEST_SHA256)"

report:
	$(PYTHON) -m microstructure.cli report --run-dir $(RUN_DIR)

report-public:
	$(PYTHON) -m microstructure.cli report --run-dir $(PUBLIC_RUN_DIR)

report-m8:
	@test -n "$(M8_RAW_MANIFEST)" || { echo "M8_RAW_MANIFEST is required"; exit 2; }
	@test -n "$(M8_RAW_MANIFEST_SHA256)" || { echo "M8_RAW_MANIFEST_SHA256 is required"; exit 2; }
	$(PYTHON) -m microstructure.cli report-m8 --config "$(M8_CONFIG)" --run-dir "$(M8_RUN_DIR)" --raw-manifest "$(M8_RAW_MANIFEST)" --raw-manifest-sha256 "$(M8_RAW_MANIFEST_SHA256)"

dashboard:
	$(PYTHON) -m streamlit run dashboard/app.py -- --run-dir $(RUN_DIR)

dashboard-public:
	$(PYTHON) -m streamlit run dashboard/app.py -- --run-dir $(PUBLIC_RUN_DIR)

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src/microstructure

check: lint typecheck test check-smoke
