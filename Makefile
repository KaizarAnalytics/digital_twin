.PHONY: install dev test lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks]"

test:
	pytest -v

lint:
	ruff check src/
	ruff format --check src/

format:
	ruff format src/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
