.DEFAULT_GOAL := help
.PHONY: help install install-dev format lint test test-cov check run train docker-build docker-run clean

# Invoke tools through `python -m` so they always run against this
# interpreter's environment rather than whatever the PATH resolves to.
PYTHON  ?= python3
PYTEST  := $(PYTHON) -m pytest
RUFF    := $(PYTHON) -m ruff
DATA    ?= data/heart-disease.csv
TARGET  ?= target
IMAGE   ?= robo-data-scientist
PORT    ?= 8501

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install runtime + development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

format: ## Apply formatting
	$(RUFF) format .
	$(RUFF) check --fix .

lint: ## Check formatting and lint (as CI does)
	$(RUFF) format --check .
	$(RUFF) check .

test: ## Run the test suite
	$(PYTEST)

test-cov: ## Run tests with a coverage report
	$(PYTEST) --cov=utils --cov-report=term-missing

check: lint test ## Run every check CI runs

run: ## Start the Streamlit app
	$(PYTHON) -m streamlit run app.py

train: ## Train on DATA/TARGET (override: make train DATA=my.csv TARGET=y)
	$(PYTHON) train.py --data $(DATA) --target $(TARGET)

docker-build: ## Build the container image
	docker build -t $(IMAGE) .

docker-run: ## Run the container on PORT (default 8501)
	docker run --rm -p $(PORT):8501 \
		-v $(PWD)/saved_models:/app/saved_models \
		$(IMAGE)

clean: ## Remove caches and generated artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov catboost_info
