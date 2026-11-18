.PHONY: all simulate analyze figures test clean base-profile optimized-profile parallel-profile complexity benchmark timing memory optimized-validation parallel-validation stability-check performance-suite all-optimized simulate-optimized all-parallel-optimized simulate-parallel-optimized parallel profile

WORKERS ?= 2

test: 
	python -m pytest

all: simulate analyze figures

simulate: .simulate_complete

.simulate_complete: configuration.py src/simulation.py src/dgps.py src/methods.py
	python -m src.simulation
	rm -f .simulate.optimized_complete .simulate.parallel-optimized_complete .analyze_complete .figures_complete
	touch .simulate_complete

analyze: .analyze_complete

.analyze_complete: src/analyze.py results/raw/simulation_data.pkl
	python -m src.analyze
	touch .analyze_complete

figures: .figures_complete

.figures_complete: src/figures.py results/raw/simulation_data.pkl
	python -m src.figures
	touch .figures_complete

clean: 
	rm -f .simulate_complete .analyze_complete .figures_complete .simulate.optimized_complete .simulate.parallel-optimized_complete
	rm -rf  results
	if [ -d performance_results ]; then find performance_results -type f ! -name .gitkeep -delete; fi

profile: base-profile

base-profile:
	python -m performance_analysis.base_profiler

optimized-profile:
	python -m performance_analysis.optimized_profiler

parallel-profile:
	python -m performance_analysis.parallel_profiler

complexity: 
	python -m performance_analysis.complexity

benchmark: 
	python -m performance_analysis.benchmark_comparisons

timing:
	python -m performance_analysis.timing_comparisons

memory:
	python -m performance_analysis.memory_comparisons

optimized-validation:
	python -m pytest -s tests/test_regression.py

parallel-validation:
	python -m pytest -s tests/test_regression_parallel.py

stability-check:
	python -m performance_analysis.numerical_stability

performance-suite:
	$(MAKE) base-profile
	$(MAKE) optimized-profile
	$(MAKE) parallel-profile
	$(MAKE) complexity
	$(MAKE) benchmark
	$(MAKE) timing
	$(MAKE) memory
	$(MAKE) stability-check
	$(MAKE) optimized-validation
	$(MAKE) parallel-validation

all-optimized: simulate-optimized analyze figures

simulate-optimized: .simulate.optimized_complete

all-parallel-optimized: simulate-parallel-optimized analyze figures

simulate-parallel-optimized: .simulate.parallel-optimized_complete

#Always run when called
parallel: 
	python -m src_optimized.simulate_parallel --workers "$(WORKERS)"


.simulate.parallel-optimized_complete: configuration_optimized.py src_optimized/simulate_parallel.py src_optimized/optimized_simulation.py src_optimized/optimized_methods.py src/dgps.py
	python -m src_optimized.simulate_parallel --workers "$(WORKERS)"
	rm -f .simulate_complete .simulate.optimized_complete .analyze_complete .figures_complete
	touch .simulate.parallel-optimized_complete

.simulate.optimized_complete: configuration_optimized.py src_optimized/optimized_simulation.py src_optimized/optimized_methods.py src/dgps.py
	python -m src_optimized.optimized_simulation
	rm -f .simulate_complete .simulate.parallel-optimized_complete .analyze_complete .figures_complete
	touch .simulate.optimized_complete

