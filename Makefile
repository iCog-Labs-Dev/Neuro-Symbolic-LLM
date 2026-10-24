.PHONY: help build test lint format smoke shell clean

DOCKER_CMD = docker run --rm -v "$$(PWD):/app" -v nsllm-hf-cache:/root/.cache/huggingface neuro-symbolic-llm

help:
	@echo "Available commands:"
	@echo "  make build         - Build the neuro-symbolic-llm Docker image"
	@echo "  make test          - Run pytest test suite in Docker"
	@echo "  make lint          - Run ruff code linter in Docker"
	@echo "  make format        - Format code with ruff and black in Docker"
	@echo "  make smoke         - Run Stage A1 smoke experiment in Docker"
	@echo "  make shell         - Start an interactive bash shell in Docker"

build:
	docker build -t neuro-symbolic-llm .

test:
	$(DOCKER_CMD) pytest

lint:
	$(DOCKER_CMD) ruff check .

format:
	$(DOCKER_CMD) ruff check --fix .
	$(DOCKER_CMD) black .

smoke:
	$(DOCKER_CMD) python experiments/run_stage_a1.py --config configs/stage_A/a1_lora_baseline.yaml

shell:
	docker run --rm -it -v "$$(PWD):/app" -v nsllm-hf-cache:/root/.cache/huggingface neuro-symbolic-llm bash
