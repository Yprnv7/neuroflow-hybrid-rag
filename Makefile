.PHONY: up down lint test bench

up:
	docker compose up --build -d

down:
	docker compose down -v

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v --cov=src/neuroflow --cov-report=term-missing

bench:
	python scripts/evaluate_beir.py --output docs/BENCHMARKS.md
