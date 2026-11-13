.PHONY: env test unpack numbers paper
env:
	conda env create -f environment.yml && conda run -n cif pip install -e .
test:
	python -m pytest tests -q
unpack:
	python scripts/unpack_runs.py
numbers:
	python scripts/paper_numbers.py --check && python scripts/paper_tables.py && python scripts/paper_figures.py
paper:
	cd paper && latexmk -pdf main.tex
