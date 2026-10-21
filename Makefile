.PHONY: all simulate analyze figures test clean test

test: 
	python -m pytest

all: .all_complete

.all_complete: simulate analyze figures

simulate: .simulate_complete

.simulate_complete: configuration.py src/simulation.py src/dgps.py src/methods.py
	python -m src.simulation
	touch .simulate_complete

analyze: .analyze_complete

.analyze_complete: .simulate_complete src/analyze.py
	python -m src.analyze
	touch .analyze_complete

figures: .figures_complete

.figures_complete: .simulate_complete src/figures.py
	python -m src.figures
	touch .figures_complete

clean: 
	rm -f .simulate_complete .analyze_complete .figures_complete .all_complete 
	rm -rf  results