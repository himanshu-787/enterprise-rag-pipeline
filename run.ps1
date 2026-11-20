# PowerShell runner for Enterprise RAG Pipeline tasks
param (
    [Parameter(Position=0)]
    [string]$Target = "help"
)

switch ($Target.ToLower()) {
    "setup" {
        & .\scripts\setup_env.ps1
    }
    "up" {
        docker compose up -d
    }
    "down" {
        docker compose down
    }
    "health" {
        python scripts/healthcheck.py
    }
    "data" {
        python scripts/download_docs.py
    }
    "ingest" {
        python scripts/ingest.py
    }
    "index" {
        python scripts/build_index.py --recreate
    }
    "bm25" {
        python scripts/build_bm25.py
    }
    "search" {
        python scripts/search_hybrid.py "how to validate query parameters"
    }
    "eval" {
        python scripts/run_eval.py
    }
    "api" {
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    }
    "test" {
        pytest -v
    }
    default {
        Write-Host "Usage: .\run.ps1 [setup|up|down|health|data|ingest|index|bm25|search|eval|api|test]" -ForegroundColor Cyan
    }
}
