.PHONY: setup up down health data ingest index bm25 search eval api test clean

PYTHON ?= python

setup:
	@bash scripts/setup_env.sh

up:
	docker compose up -d

down:
	docker compose down

health:
	$(PYTHON) scripts/healthcheck.py

data:
	$(PYTHON) scripts/download_docs.py

ingest:
	$(PYTHON) scripts/ingest.py

index:
	$(PYTHON) scripts/build_index.py --recreate

bm25:
	$(PYTHON) scripts/build_bm25.py

search:
	$(PYTHON) scripts/search_hybrid.py "how to validate query parameters"

eval:
	$(PYTHON) scripts/run_eval.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -v

clean:
	rm -rf data/processed/*
	rm -rf __pycache__ .pytest_cache .coverage
