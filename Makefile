PYTHON ?= python
MPIEXEC ?= mpiexec
MPI_FLAGS ?= --use-hwthread-cpus
MPI_N ?= 8
OMP_NUM_THREADS ?= 1
NUMEXPR_NUM_THREADS ?= 1
JOBS ?= 8

.PHONY: help install test unit lint typecheck check smoke smoke-v01 smoke-v02 smoke-v03 smoke-v04 \
        quick-study-v02 quick-study-v04 quick-overnight-v04 overnight-v04 analyze-v04-rh plot-v04-rh \
        quick-v05 study-v05 quick-sweep-v05 sweep-v05 quick-grid-v05 grid-v05 \
        quick-v06 study-v06 quick-ka-v06 ka-v06 quick-grid-v06 grid-v06 quick-rha-v06 rha-v06 \
        quick-v07 study-v07 quick-volume-v07 volume-v07 quick-h2-v07 h2-v07 quick-purge-v07 purge-v07 \
        quick-transient-purge-v07 transient-purge-v07 quick-scaling-v07 scaling-v07 \
        quick-purge-duration-v07 purge-duration-v07 quick-n2-v07 n2-v07 \
        quick-n2-sensitivity-v07 n2-sensitivity-v07 \
        quick-h2-feedback-v07 h2-feedback-v07 \
        quick-n2-pressure-v07 n2-pressure-v07 \
        quick-n2-literature-v07 n2-literature-v07 \
        quick-n2-state-v07 n2-state-v07 \
        quick-n2-water-v07 n2-water-v07 \
        quick-n2-water-volume-v07 n2-water-volume-v07 \
        quick-n2-catalano-v07 n2-catalano-v07 \
        quick-n2-catalano-profile-v07 n2-catalano-profile-v07 \
        quick-n2-catalano-transport-v07 n2-catalano-transport-v07 \
        quick-n2-catalano-transport-sensitivity-v07 n2-catalano-transport-sensitivity-v07 \
        analyze-n2-membrane-profiles-v07 analyze-n2-motupally-v07 \
        quick-n2-catalano-motupally-v07 quick-n2-catalano-motupally-33-v07 analyze-n2-catalano-motupally-lookup-v07 n2-catalano-motupally-v07 \
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
	  'make quick-overnight-v04 preflight the overnight RH-voltage campaign' \
	  'make overnight-v04    run fixed-grid V0.4 RH-voltage overnight campaign' \
	  'make analyze-v04-rh   analyze RH-dependent polarization and sensitivity' \
	  'make plot-v04-rh      generate PNG/PDF figures from V0.4 RH analysis' \
	  'make quick-v05        preflight V0.5 cathode-membrane coupling' \
	  'make study-v05        run converged V0.5 cathode-membrane coupling' \
	  'make quick-sweep-v05  preflight one V0.5 RH-voltage sweep case' \
	  'make sweep-v05        run 3x3 V0.5 RH-voltage campaign' \
	  'make quick-grid-v05   preflight V0.5 spatial-convergence driver' \
	  'make grid-v05         run 3-point V0.5 spatial convergence study' \
	  'make quick-v06        preflight V0.6 finite anode-transfer coupling' \
	  'make study-v06        run nominal V0.6 finite anode-transfer coupling' \
	  'make quick-ka-v06     preflight one V0.6 k_a sensitivity case' \
	  'make ka-v06           run V0.6 k_a sensitivity on three regimes' \
	  'make quick-grid-v06   preflight V0.6 spatial-convergence driver' \
	  'make grid-v06         run V0.6 dry/high-load spatial convergence' \
	  'make quick-rha-v06    preflight one V0.6 RH_anode sensitivity case' \
	  'make rha-v06          run V0.6 RH_anode sensitivity on three regimes' \
	  'make quick-v07        preflight V0.7 dynamic anode water volume' \
	  'make study-v07        run V0.7 dynamic anode water-volume study' \
	  'make quick-volume-v07 preflight V0.7 anode-volume sensitivity' \
	  'make volume-v07       run V0.7 anode-volume sensitivity campaign' \
	  'make quick-h2-v07     preflight V0.7 dead-end H2 inventory' \
	  'make h2-v07           run V0.7 dead-end H2 inventory study' \
	  'make quick-purge-v07  preflight V0.7 discrete purge model' \
	  'make purge-v07        run V0.7 charge-triggered purge study' \
	  'make quick-transient-purge-v07 preflight finite-duration purge' \
	  'make transient-purge-v07 run finite-duration pressure-driven purge' \
	  'make quick-scaling-v07 preflight full-cell active-area scaling' \
	  'make scaling-v07       run full-cell active-area scaling study' \
	  'make quick-purge-duration-v07 preflight dynamic-clock purge durations' \
	  'make purge-duration-v07 compare 0.2 s and 0.5 s purge durations' \
	  'make quick-n2-v07      preflight reduced N2 crossover model' \
	  'make n2-v07            run reduced N2 crossover campaign' \
	  'make quick-n2-sensitivity-v07 preflight N2 crossover sensitivity' \
	  'make n2-sensitivity-v07 run N2 crossover sensitivity campaign' \
	  'make quick-h2-feedback-v07 preflight H2 dilution feedback study' \
	  'make h2-feedback-v07   run H2 dilution feedback sensitivity' \
	  'make quick-n2-pressure-v07 preflight pressure-driven N2 crossover' \
	  'make n2-pressure-v07   run pressure-driven N2 crossover campaign' \
	  'make quick-n2-literature-v07 preflight literature N2 permeability' \
	  'make n2-literature-v07 run literature-anchored N2 permeability study' \
	  'make quick-n2-state-v07 preflight state-dependent N2 permeability' \
	  'make n2-state-v07      run state-dependent N2 permeability study' \
	  'make quick-n2-water-v07 preflight water-content N2 permeability' \
	  'make n2-water-v07      run water-content N2 permeability study' \
	  'make quick-n2-water-volume-v07 preflight water-volume N2 permeability' \
	  'make n2-water-volume-v07 run water-volume N2 permeability study' \
	  'make quick-n2-catalano-v07 preflight Catalano-calibrated N2 study' \
	  'make n2-catalano-v07   run Catalano-calibrated N2 study' \
	  'make quick-n2-catalano-profile-v07 preflight mean-state vs through-plane Catalano comparison' \
	  'make n2-catalano-profile-v07 run mean-state vs through-plane Catalano comparison' \
	  'make quick-n2-catalano-transport-v07 preflight linear vs V0.6 membrane-profile comparison' \
	  'make n2-catalano-transport-v07 run linear vs V0.6 membrane-profile comparison' \
	  'make quick-n2-catalano-transport-sensitivity-v07 preflight D_water/k_a sensitivity' \
	  'make n2-catalano-transport-sensitivity-v07 run full D_water/k_a sensitivity' \
	  'make analyze-n2-membrane-profiles-v07 diagnose Pe/Bi and N2 resistance profiles' \
	  'make analyze-n2-motupally-v07 compare constant and Motupally water diffusivity' \
	  'make quick-n2-catalano-motupally-v07 preflight constant-D vs Motupally crossover' \
	  'make quick-n2-catalano-motupally-33-v07 rerun quick with 33x33 lookup' \
	  'make analyze-n2-catalano-motupally-lookup-v07 compare 21x21 and 33x33 lookups' \
	  'make n2-catalano-motupally-v07 run constant-D vs Motupally crossover' \
	  'make results          validate existing output/ and output-electrochem/' \
	  'make push-results     commit only results/ and push them to GitHub' \
	  'make clean-results    remove generated validation reports'

install:
	$(PYTHON) -m pip install -e '.[test]'

test check: unit lint typecheck

unit:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check --fix --unsafe-fixes src tests scripts
	@if ! git diff --quiet -- src tests scripts; then \
		git add src tests scripts; \
		git commit -m "Ruff fixes"; \
	else \
		echo "Ruff: no changes to commit"; \
	fi

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


quick-overnight-v04:
	rm -rf .quick-overnight-output/v04
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v04_overnight.py \
	  --nx 8 --ny 8 --nz 24 --stop-time 0.0002 --max-dt 0.000001 \
	  --rh-values 0.5 --voltages 0.768 \
	  --work-dir .quick-overnight-output/v04 \
	  --json-output results/quick-overnight-v04.json \
	  --csv-output results/quick-overnight-v04.csv

overnight-v04:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v04_overnight.py


analyze-v04-rh:
	$(PYTHON) scripts/analyze_v04_rh_sweep.py


plot-v04-rh:
	$(PYTHON) scripts/plot_v04_rh_sweep.py


quick-v05:
	rm -rf .quick-v05
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v05_coupled.py \
	  --nx 8 --ny 8 --nz 24 --membrane-nz 33 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 3 --current-rtol 1 --potential-atol 1 \
	  --work-dir .quick-v05 --output results/quick-v05-coupled.json

study-v05:
	rm -rf .v05-coupling
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v05_coupled.py


quick-sweep-v05:
	rm -rf .quick-v05-sweep
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v05_rh_voltage_sweep.py \
	  --rh-values 0.5 --voltages 0.768 \
	  --nx 8 --ny 8 --nz 24 --membrane-nz 65 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 10 --current-rtol 0.01 --potential-atol 0.0001 \
	  --work-dir .quick-v05-sweep \
	  --output-json results/quick-v05-sweep.json \
	  --output-csv results/quick-v05-sweep.csv

sweep-v05:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v05_rh_voltage_sweep.py


quick-grid-v05:
	rm -rf .quick-v05-grid
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v05_grid_convergence.py \
	  --quick --membrane-nz 65 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 10 --current-rtol 0.01 --potential-atol 0.0001 \
	  --work-dir .quick-v05-grid \
	  --output-json results/quick-v05-grid-convergence.json \
	  --output-csv results/quick-v05-grid-convergence.csv

grid-v05:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v05_grid_convergence.py


quick-v06:
	rm -rf .quick-v06
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' \
	  $(PYTHON) scripts/run_v06_coupled.py \
	  --nx 8 --ny 8 --nz 24 --membrane-nz 65 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 10 --current-rtol 0.01 --potential-atol 0.0001 \
	  --mpi-n 2 --work-dir .quick-v06 \
	  --output results/quick-v06-coupled.json

study-v06:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' \
	  $(PYTHON) scripts/run_v06_coupled.py \
	  --mpi-n $(MPI_N) \
	  --output results/v06-coupled-membrane.json


quick-ka-v06:
	rm -rf .quick-v06-ka
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v06_ka_sensitivity.py \
	  --regimes nominal --k-values 1e-6 \
	  --nx 8 --ny 8 --nz 24 --membrane-nz 65 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 10 --current-rtol 0.01 --potential-atol 0.0001 \
	  --work-dir .quick-v06-ka \
	  --output-json results/quick-v06-ka-sensitivity.json \
	  --output-csv results/quick-v06-ka-sensitivity.csv

ka-v06:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v06_ka_sensitivity.py


quick-grid-v06:
	rm -rf .quick-v06-grid
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v06_grid_convergence.py \
	  --quick --regimes dry_high_load --membrane-nz 65 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 10 --current-rtol 0.01 --potential-atol 0.0001 \
	  --mpi-n 2 --work-dir .quick-v06-grid \
	  --output-json results/quick-v06-grid-convergence.json \
	  --output-csv results/quick-v06-grid-convergence.csv

grid-v06:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v06_grid_convergence.py \
	  --regimes dry_high_load --mpi-n $(MPI_N)


quick-rha-v06:
	rm -rf .quick-v06-rha
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=2 \
	  $(PYTHON) scripts/run_v06_rha_sensitivity.py \
	  --regimes nominal --anode-rh-values 0.5 \
	  --nx 8 --ny 8 --nz 24 --membrane-nz 65 \
	  --stop-time 0.0002 --max-dt 0.000001 --scalar-dt 0.00001 \
	  --max-coupling-iterations 10 --current-rtol 0.01 --potential-atol 0.0001 \
	  --mpi-n 2 --work-dir .quick-v06-rha \
	  --output-json results/quick-v06-rha-sensitivity.json \
	  --output-csv results/quick-v06-rha-sensitivity.csv

rha-v06:
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) NUMEXPR_NUM_THREADS=$(NUMEXPR_NUM_THREADS) \
	MPIEXEC=$(MPIEXEC) MPI_FLAGS='$(MPI_FLAGS)' MPI_N=$(MPI_N) \
	  $(PYTHON) scripts/run_v06_rha_sensitivity.py \
	  --mpi-n $(MPI_N)


quick-v07:
	$(PYTHON) scripts/run_v07_anode_volume.py \
	  --regimes nominal --stop-time 60 --dt 1 --write-every 5 \
	  --output-json results/quick-v07-anode-volume.json \
	  --output-csv results/quick-v07-anode-volume.csv

study-v07:
	$(PYTHON) scripts/run_v07_anode_volume.py


quick-volume-v07:
	$(PYTHON) -m scripts.run_v07_anode_volume_sensitivity \
	  --regimes nominal --volumes-ml 10 20 40 \
	  --horizon-per-ml 100 --dt 1 --write-every 5 \
	  --output-json results/quick-v07-anode-volume-sensitivity.json \
	  --output-csv results/quick-v07-anode-volume-sensitivity.csv

volume-v07:
	$(PYTHON) -m scripts.run_v07_anode_volume_sensitivity


quick-h2-v07:
	$(PYTHON) -m scripts.run_v07_anode_h2 \
	  --regimes nominal --stop-time 60 --dt 1 --write-every 5 \
	  --output-json results/quick-v07-anode-h2.json \
	  --output-csv results/quick-v07-anode-h2.csv

h2-v07:
	$(PYTHON) -m scripts.run_v07_anode_h2


quick-purge-v07:
	$(PYTHON) -m scripts.run_v07_anode_purge \
	  --regimes nominal --stop-time 200 --dt 0.1 --write-every 10 \
	  --output-json results/quick-v07-anode-purge.json \
	  --output-csv results/quick-v07-anode-purge.csv

purge-v07:
	$(PYTHON) -m scripts.run_v07_anode_purge


quick-transient-purge-v07:
	$(PYTHON) -m scripts.run_v07_anode_transient_purge \
	  --regimes nominal --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-anode-transient-purge.json \
	  --output-csv results/quick-v07-anode-transient-purge.csv

transient-purge-v07:
	$(PYTHON) -m scripts.run_v07_anode_transient_purge


quick-scaling-v07:
	$(PYTHON) -m scripts.run_v07_active_area_scaling \
	  --regimes nominal --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-active-area-scaling.json \
	  --output-csv results/quick-v07-active-area-scaling.csv

scaling-v07:
	$(PYTHON) -m scripts.run_v07_active_area_scaling


quick-purge-duration-v07:
	$(PYTHON) -m scripts.run_v07_purge_duration_comparison \
	  --regimes nominal --durations 0.2 0.5 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-purge-duration-comparison.json \
	  --output-csv results/quick-v07-purge-duration-comparison.csv

purge-duration-v07:
	$(PYTHON) -m scripts.run_v07_purge_duration_comparison


quick-n2-v07:
	$(PYTHON) -m scripts.run_v07_anode_nitrogen \
	  --regimes nominal --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-anode-nitrogen.json \
	  --output-csv results/quick-v07-anode-nitrogen.csv

n2-v07:
	$(PYTHON) -m scripts.run_v07_anode_nitrogen


quick-n2-sensitivity-v07:
	$(PYTHON) -m scripts.run_v07_n2_crossover_sensitivity \
	  --regimes nominal --fluxes 0 5e-7 1e-6 2e-6 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-crossover-sensitivity.json \
	  --output-csv results/quick-v07-n2-crossover-sensitivity.csv

n2-sensitivity-v07:
	$(PYTHON) -m scripts.run_v07_n2_crossover_sensitivity


quick-h2-feedback-v07:
	$(PYTHON) -m scripts.run_v07_h2_feedback_sensitivity \
	  --regimes nominal --fluxes 0 1e-6 5e-6 --exponents 0 0.5 1 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-h2-feedback-sensitivity.json \
	  --output-csv results/quick-v07-h2-feedback-sensitivity.csv

h2-feedback-v07:
	$(PYTHON) -m scripts.run_v07_h2_feedback_sensitivity


quick-n2-pressure-v07:
	$(PYTHON) -m scripts.run_v07_n2_pressure_crossover \
	  --regimes nominal --reference-fluxes 1e-6 5e-6 \
	  --feedback-exponents 0 1 --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-pressure-crossover.json \
	  --output-csv results/quick-v07-n2-pressure-crossover.csv

n2-pressure-v07:
	$(PYTHON) -m scripts.run_v07_n2_pressure_crossover


quick-n2-literature-v07:
	$(PYTHON) -m scripts.run_v07_n2_literature_permeability \
	  --regimes nominal --permeabilities-barrer 0.24 2.4 24 \
	  --feedback-exponents 0 1 --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-literature-permeability.json \
	  --output-csv results/quick-v07-n2-literature-permeability.csv

n2-literature-v07:
	$(PYTHON) -m scripts.run_v07_n2_literature_permeability


quick-n2-state-v07:
	$(PYTHON) -m scripts.run_v07_n2_state_permeability \
	  --regimes nominal --humidity-shape-exponents 1 2 4 \
	  --feedback-exponents 0 1 --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-state-permeability.json \
	  --output-csv results/quick-v07-n2-state-permeability.csv

n2-state-v07:
	$(PYTHON) -m scripts.run_v07_n2_state_permeability


quick-n2-water-v07:
	$(PYTHON) -m scripts.run_v07_n2_water_content_permeability \
	  --regimes nominal --feedback-exponents 0 1 --jobs $(JOBS) \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-water-content-permeability.json \
	  --output-csv results/quick-v07-n2-water-content-permeability.csv

n2-water-v07:
	$(PYTHON) -m scripts.run_v07_n2_water_content_permeability


quick-n2-water-volume-v07:
	$(PYTHON) -m scripts.run_v07_n2_water_volume_permeability \
	  --regimes nominal --feedback-exponents 0 1 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-water-volume-permeability.json \
	  --output-csv results/quick-v07-n2-water-volume-permeability.csv

n2-water-volume-v07:
	$(PYTHON) -m scripts.run_v07_n2_water_volume_permeability


quick-n2-catalano-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_calibrated \
	  --regimes nominal \
	  --water-partial-molar-volumes-cm3-mol 17.0 18.01528 \
	  --activation-energies-j-mol 49600 19830 \
	  --feedback-exponents 0 1 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-catalano-calibrated.json \
	  --output-csv results/quick-v07-n2-catalano-calibrated.csv

n2-catalano-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_calibrated


quick-n2-catalano-profile-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_profile_comparison \
	  --regimes nominal --feedback-exponents 0 1 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-catalano-profile-comparison.json \
	  --output-csv results/quick-v07-n2-catalano-profile-comparison.csv

n2-catalano-profile-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_profile_comparison


quick-n2-catalano-transport-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_membrane_transport \
	  --regimes nominal --feedback-exponents 0 1 --jobs $(JOBS) \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-catalano-membrane-transport-comparison.json \
	  --output-csv results/quick-v07-n2-catalano-membrane-transport-comparison.csv

n2-catalano-transport-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_membrane_transport --jobs $(JOBS)


quick-n2-catalano-transport-sensitivity-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_membrane_transport_sensitivity \
	  --regimes nominal wet_low_load --feedback-exponents 0 \
	  --diffusivities-m2-s 1e-10 2e-10 4e-10 \
	  --transfer-coefficients-m-s 1e-6 2e-6 4e-6 \
	  --jobs $(JOBS) --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-catalano-membrane-transport-sensitivity.json \
	  --output-csv results/quick-v07-n2-catalano-membrane-transport-sensitivity.csv

n2-catalano-transport-sensitivity-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_membrane_transport_sensitivity \
	  --jobs $(JOBS)


analyze-n2-membrane-profiles-v07:
	$(PYTHON) -m scripts.analyze_v07_n2_membrane_profiles


analyze-n2-motupally-v07:
	$(PYTHON) -m scripts.analyze_v07_motupally_diffusivity


quick-n2-catalano-motupally-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_motupally \
	  --regimes nominal wet_low_load --feedback-exponents 0 1 \
	  --jobs $(JOBS) --lookup-rh-points 21 --lookup-current-points 21 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-catalano-motupally-comparison.json \
	  --output-csv results/quick-v07-n2-catalano-motupally-comparison.csv

quick-n2-catalano-motupally-33-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_motupally \
	  --regimes nominal wet_low_load --feedback-exponents 0 1 \
	  --jobs $(JOBS) --lookup-rh-points 33 --lookup-current-points 33 \
	  --stop-time 200 --dt 0.01 --write-every 100 \
	  --output-json results/quick-v07-n2-catalano-motupally-comparison-33x33.json \
	  --output-csv results/quick-v07-n2-catalano-motupally-comparison-33x33.csv

analyze-n2-catalano-motupally-lookup-v07:
	$(PYTHON) -m scripts.analyze_v07_motupally_lookup_convergence

n2-catalano-motupally-v07:
	$(PYTHON) -m scripts.run_v07_n2_catalano_motupally \
	  --jobs $(JOBS) --lookup-rh-points 33 --lookup-current-points 33
