# ML Model Serving Makefile

.PHONY: help install test lint format run build deploy clean

# Default target
help:
	@echo "ML Model Serving Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install dependencies"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linting"
	@echo "  make format      Format code"
	@echo "  make run         Run development server"
	@echo "  make build       Build Docker images"
	@echo "  make deploy      Deploy to Kubernetes"
	@echo "  make clean       Clean up generated files"
	@echo ""

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements/dev.txt
	pip install -r requirements/api.txt
	pip install -r requirements/training.txt
	@echo "Installation complete!"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=app --cov-report=html
	@echo "Tests completed!"

# Linting
lint:
	@echo "Running linting..."
	flake8 app/ tests/ training_pipeline/ --max-line-length=88 --ignore=E203,W503
	mypy app/ tests/ training_pipeline/ --ignore-missing-imports
	@echo "Linting completed!"

# Formatting
format:
	@echo "Formatting code..."
	black app/ tests/ training_pipeline/ scripts/ --line-length=88
	isort app/ tests/ training_pipeline/ scripts/ --profile=black
	@echo "Formatting completed!"

# Development
run:
	@echo "Starting development server..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker build
build:
	@echo "Building Docker images..."
	docker build -f docker/Dockerfile.api -t ml-model-api:latest .
	docker build -f docker/Dockerfile.training -t ml-training-pipeline:latest .
	@echo "Docker images built!"

# Kubernetes deployment
deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f kubernetes/api-deployment.yaml
	kubectl apply -f kubernetes/api-service.yaml
	kubectl apply -f kubernetes/training-job.yaml
	kubectl apply -f kubernetes/prometheus.yaml
	kubectl apply -f kubernetes/grafana.yaml
	@echo "Deployment completed!"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "htmlcov" -exec rm -rf {} +
	rm -rf .mypy_cache
	@echo "Clean up completed!"

# Database operations
db-init:
	@echo "Initializing database..."
	python -c "from app.db.session import init_db; import asyncio; asyncio.run(init_db())"

db-reset: clean db-init
	@echo "Database reset complete!"

# Training pipeline
train:
	@echo "Running training pipeline..."
	python -m training_pipeline.pipeline

# Monitoring
monitor:
	@echo "Starting monitoring..."
	python scripts/monitor_drift.py v1 --check-drift --check-stats

# Production deployment
prod-deploy: test lint build deploy
	@echo "Production deployment completed!"

# Development setup
dev-setup: install db-init
	@echo "Development environment setup complete!"