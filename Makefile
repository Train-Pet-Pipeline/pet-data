-include ../pet-infra/shared/Makefile.include

.PHONY: setup test

setup:
	python -m pip install -e ".[dev]" && python -m piptools compile pyproject.toml -o requirements.txt

test:
	pytest tests/ -v
