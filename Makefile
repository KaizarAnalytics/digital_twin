install:
	pip install -r requirements.txt

quickscan:
	python scripts/run_quickscan.py

report:
	python scripts/build_report.py
