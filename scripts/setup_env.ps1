Write-Host "=== Enterprise RAG Pipeline Environment Setup (Windows) ===" -ForegroundColor Cyan

# Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "[OK] $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
}

# Check Git
try {
    $gitVersion = git --version 2>&1
    Write-Host "[OK] $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Git is not installed or not in PATH." -ForegroundColor Yellow
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "[OK] $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[INFO] Docker not found in PATH. You can run Qdrant and Redis locally or via Docker Desktop." -ForegroundColor Yellow
}

# Setup .env
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
}

# Setup venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "`nTo activate environment and install dependencies:" -ForegroundColor Cyan
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  pip install -r requirements.txt"
Write-Host "=== Setup completed ===" -ForegroundColor Green
