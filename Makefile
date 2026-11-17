.PHONY: all simulate analyze figures test clean base-profiler optimized-profiler complexity benchmark timing memory optimized-validation stability-check performance-suite all-optimized simulate-optimized

test: 
	python -m pytest

all: simulate analyze figures

simulate: .simulate_complete

.simulate_complete: configuration.py src/simulation.py src/dgps.py src/methods.py
	python -m src.simulation
	rm -f .simulate.optimized_complete .analyze_complete .figures_complete
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
	rm -f .simulate_complete .analyze_complete .figures_complete .simulate.optimized_complete
	rm -rf  results

base-profiler:
	python -m performance_analysis.base_profiler

optimized-profiler:
	python -m performance_analysis.optimized_profiler

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

stability-check:
	python -m performance_analysis.numerical_stability

performance-suite:
	$(MAKE) base-profiler
	$(MAKE) optimized-profiler
	$(MAKE) complexity
	$(MAKE) benchmark
	$(MAKE) timing
	$(MAKE) memory
	$(MAKE) stability-check
	$(MAKE) optimized-validation

all-optimized: simulate-optimized analyze figures

simulate-optimized: .simulate.optimized_complete

.simulate.optimized_complete: configuration_optimized.py src_optimized/optimized_simulation.py src_optimized/optimized_methods.py src/dgps.py
	python -m src_optimized.optimized_simulation
	rm -f .simulate_complete .analyze_complete .figures_complete
	touch .simulate.optimized_complete

