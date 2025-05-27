# File: Makefile
.PHONY: help build run stop export-model lint clean test logs

# Variables
IMAGE_NAME = hi-asr-api
CONTAINER_NAME = hi-asr-api-container
PYTHON = python3 # Or specify path to venv python

help:
	@echo "Makefile for Hindi ASR FastAPI Service"
	@echo ""
	@echo "Usage:"
	@echo "  make build          Build the Docker image."
	@echo "  make run            Run the Docker container in detached mode."
	@echo "  make run-interactive Run the Docker container in interactive mode (logs to console)."
	@echo "  make stop           Stop and remove the running Docker container."
	@echo "  make logs           Follow logs from the running container."
	@echo "  make export-model   Download NeMo model and export to ONNX (runs locally)."
	@echo "  make lint           Run linters (black, flake8) on the app code."
	@echo "  make clean          Remove __pycache__ directories and .pyc files."
	@echo "  make test           (Placeholder) Run tests."
	@echo ""

# Docker commands
build:
	@echo "Building Docker image $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME) .

run: stop
	@echo "Running Docker container $(CONTAINER_NAME) in detached mode..."
	docker run -d -p 8000:8000 --name $(CONTAINER_NAME) \
		-v $(shell pwd)/model:/home/appuser/app/model \
		-v $(shell pwd)/audio:/home/appuser/app/audio \
		$(IMAGE_NAME)
	@echo "Container $(CONTAINER_NAME) started. API at http://localhost:8000"

run-interactive: stop
	@echo "Running Docker container $(CONTAINER_NAME) in interactive mode..."
	docker run --rm -p 8000:8000 --name $(CONTAINER_NAME) \
		-v $(shell pwd)/model:/home/appuser/app/model \
		-v $(shell pwd)/audio:/home/appuser/app/audio \
		$(IMAGE_NAME)

stop:
	@echo "Stopping and removing container $(CONTAINER_NAME) if it exists..."
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)

logs:
	@echo "Following logs for container $(CONTAINER_NAME)..."
	docker logs -f $(CONTAINER_NAME)

# Model export
export-model:
	@echo "Exporting NeMo model to ONNX..."
	$(PYTHON) model/export_model.py

# Linting (Assumes black and flake8 are installed in your dev environment)
lint:
	@echo "Running linters..."
	$(PYTHON) -m black app/ model/
	$(PYTHON) -m flake8 app/ model/

# Clean Python cache
clean:
	@echo "Cleaning Python cache files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Placeholder for tests
test:
	@echo "Running tests (placeholder)..."
	# Example: $(PYTHON) -m pytest