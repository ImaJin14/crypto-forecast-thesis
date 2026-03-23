.PHONY: help setup data train eval tune ablation test lint clean

help:
	@echo ""
	@echo "  crypto-forecast-thesis — Available Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup      Install dependencies"
	@echo "  make data       Run full data collection + preprocessing pipeline"
	@echo "  make train      Train a model (MODEL=lstm ASSET=BTC HORIZON=1d)"
	@echo "  make tune       Run Optuna hyperparameter search"
	@echo "  make eval       Evaluate all models and generate KPI tables"
	@echo "  make ablation   Run feature ablation study"
	@echo "  make test       Run pytest test suite"
	@echo "  make lint       Run ruff + black formatting checks"
	@echo "  make clean      Remove __pycache__ and .pyc files"
	@echo ""

setup:
	pip install -r requirements.txt

data:
	python scripts/run_data_pipeline.py

train:
	python scripts/run_training.py --model $(MODEL) --asset $(ASSET) --horizon $(HORIZON)

tune:
	python scripts/run_tuning.py --model $(MODEL) --asset $(ASSET)

eval:
	python scripts/run_evaluation.py

ablation:
	python scripts/run_ablation.py

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
