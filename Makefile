.PHONY: up down lint test bench

up:
	docker compose up --build -d

down:
	docker compose down -v

lint:
	ruff check src/ tests/ scripts/

test:
	pytest tests/ -v --cov=src/neuroflow --cov-report=term-missing

bench:
	python scripts/evaluate_beir.py --dataset scifact --limit 150 --output docs/BENCHMARKS.md
