.PHONY: setup test lint clean

setup:
	pip install -e ".[dev]" && pip-compile pyproject.toml -o requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ && mypy src/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist/ *.egg-info
	find . -name __pycache__ -exec rm -rf {} +
