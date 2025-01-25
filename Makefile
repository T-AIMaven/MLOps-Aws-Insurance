# Makefile

.PHONY: all req test lint format coverage clean

# Variables
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8

all: req lint format test coverage clean ## Run all tasks


req: ## Install the requirements
	$(PIP) install -r requirements.txt

test: ## Run tests with pytest
	$(PYTEST) -vv -s tests/

lint: ## Run linters (flake8, isort)
	$(FLAKE8) .
	$(ISORT) --check-only .

format: ## Run code formatter (black, isort)
	$(BLACK) .
	$(ISORT) .

coverage: ## Run tests with coverage
	$(PYTEST) --cov=tests --cov-report=term-missing --cov-report=html:coverage_report -vv -s tests/


clean: ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -f .coverage
