-include ../pet-infra/shared/Makefile.include

.PHONY: setup test

setup:
	pip install -e ".[dev]" && pip-compile pyproject.toml -o requirements.txt

test:
	pytest tests/ -v
