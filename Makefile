PYTHON ?= python
MPIEXEC ?= mpiexec
MPI_FLAGS ?= --use-hwthread-cpus
MPI_N ?= 8
OMP_NUM_THREADS ?= 1
NUMEXPR_NUM_THREADS ?= 1

.PHONY: help install test unit lint typecheck check smoke smoke-v01 smoke-v02 smoke-v03 smoke-v04 \
        quick-study-v02 quick-study-v04 \
        study-v02 study-v03 study-v04 validate-results validate-v01 validate-v02 validate-v03 results \
        push-results clean-results

help:
	@printf '%s\n' \
	  'make install          install editable package and test dependencies' \
	  'make test             run unit tests, Ruff and mypy' \
	  'make unit             run pytest only' \
	  'make lint             run Ruff checks' \
	  'make typecheck        run mypy' \
	  'make smoke            run tiny Dedalus V0.1-V0.4 smoke tests' \
	  'make study-v02        run V0.2 time/grid convergence study' \
	  'make study-v03        run V0.3 membrane time/grid convergence study' \
	  'make study-v04        run V0.4 hydrated cathode convergence study' \
	  'make quick-study-v02  run short V0.2 MPI pipeline check' \
	  'make quick-study-v04  run short V0.4 MPI/comparison pipeline check' \
	  'make results          validate existing output/ and output-electrochem/' \
	  'make push-results     commit only results/ and push them to GitHub' \
	  'make clean-results    remove generated validation reports'

install:
	$(PYTHON) -m pip install -e '.[test]'

test check: unit lint typecheck

unit:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy src tests scripts

smoke: smoke-v01 smoke-v02 smoke-v03 smoke-v04

smoke-v01:
	rm -rf .test-output/v01
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	  pemfc-cathode-3d --nx 8 --ny 8 --nz 24 --stop-time 0.0002 \
	  --output-dir .test-output/v01
	$(PYTHON) scripts/validate_results.py --model v01 \
	  --input .test-output/v01 --output results/smoke-v01.json

smoke-v02:
	rm -rf .test-output/v02
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	  pemfc-cathode-electrochem-3d --nx 8 --ny 8 --nz 24 \
	  --stop-time 0.00005 --max-dt 0.000001 \
	  --output-dir .test-output/v02
	$(PYTHON) scripts/validate_results.py --model v02 \
	  --input .test-output/v02 --output results/smoke-v02.json

smoke-v03:
	rm -rf .test-output/v03
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	  pemfc-membrane-1d --nz 32 --stop-time 0.01 --max-dt 0.0001 \
	  --output-dir .test-output/v03
	$(PYTHON) scripts/validate_results.py --model v03 \
	  --input .test-output/v03 --output results/smoke-v03.json

smoke-v04:
	rm -rf .test-output/v04
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	  pemfc-cathode-hydrated-3d --nx 8 --ny 8 --nz 24 \
	  --stop-time 0.00005 --max-dt 0.000001 \
	  --output-dir .test-output/v04
	$(PYTHON) scripts/validate_results.py --model v04 \
	  --input .test-output/v04 --output results/smoke-v04.json

study-v02:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v02_study.py

quick-study-v02:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v02_study.py \
	  --grid 8x8x24 --initial-stop-time 0.0002 --max-stop-time 0.0002 \
	  --time-tol 1 --grid-tol 1 \
	  --work-dir .quick-study-output/v02 \
	  --output results/quick-v02-study.json

study-v03:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	  $(PYTHON) scripts/run_v03_study.py

study-v04:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v04_study.py

quick-study-v04:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v04_study.py \
	  --grid 8x8x24 --initial-stop-time 0.0002 --max-stop-time 0.0002 \
	  --time-tol 1 --grid-tol 1 \
	  --work-dir .quick-study-output/v04 \
	  --output results/quick-v04-study.json \
	  --reference-v02 results/quick-v02-study.json

validate-results: validate-v01 validate-v02 validate-v03

results: validate-results

validate-v01:
	@test -d output || { echo 'output/ missing; run V0.1 first'; exit 1; }
	$(PYTHON) scripts/validate_results.py --model v01 --input output \
	  --output results/v01-latest.json

validate-v02:
	@test -d output-electrochem || { echo 'output-electrochem/ missing; run V0.2 first'; exit 1; }
	$(PYTHON) scripts/validate_results.py --model v02 --input output-electrochem \
	  --output results/v02-latest.json

validate-v03:
	@test -d output-membrane || { echo 'output-membrane/ missing; run V0.3 first'; exit 1; }
	$(PYTHON) scripts/validate_results.py --model v03 --input output-membrane \
	  --output results/v03-latest.json

push-results:
	@test -d results || { echo 'results/ missing'; exit 1; }
	git add results
	@if git diff --cached --quiet -- results; then \
	  echo 'No result changes to commit.'; \
	else \
	  git commit --only -m "Add PEMFC validation results" -- results; \
	fi
	git push

clean-results:
	rm -rf .test-output results/*.json
