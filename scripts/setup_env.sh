#!/usr/bin/env bash
set -e

echo "=== Enterprise RAG Pipeline Environment Setup ==="

# Check Python version
python_bin="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        python_bin="python"
    else
        echo "Error: Python 3 is not installed."
        exit 1
    fi
fi

py_ver=$($python_bin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[OK] Found Python $py_ver"

# Check Git
if ! command -v git &> /dev/null; then
    echo "Warning: git is not installed or not in PATH."
else
    echo "[OK] Found git $(git --version)"
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "[OK] Found docker $(docker --version)"
else
    echo "[INFO] Docker not found in PATH. Ensure Docker Desktop is installed for containerized Qdrant/Redis."
fi

# Copy .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    $python_bin -m venv .venv
fi

echo "To activate environment and install dependencies:"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "=== Setup check completed ==="
