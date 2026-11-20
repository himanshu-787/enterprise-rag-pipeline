#!/usr/bin/env python3
"""System health check script for infrastructure and data dependencies."""
import os
import sys
import socket
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import settings


def check_qdrant(url: str) -> bool:
    """Check if Qdrant REST API is reachable."""
    endpoints = [f"{url}/healthz", f"{url}/collections", url]
    for ep in endpoints:
        try:
            req = urllib.request.Request(ep, headers={"User-Agent": "Enterprise-RAG-Healthcheck"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status in (200, 404):
                    return True
        except Exception:
            continue
    return False


def check_redis(redis_url: str) -> bool:
    """Check if Redis port is listening."""
    try:
        # Parse redis://host:port
        clean = redis_url.replace("redis://", "")
        if "@" in clean:
            clean = clean.split("@")[-1]
        host, port_str = clean.split(":")
        port = int(port_str.split("/")[0])
        with socket.create_connection((host, port), timeout=2.0):
            return True
    except Exception:
        return False


def check_accelerator() -> str:
    """Check available hardware accelerators for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"CUDA GPU ({gpu_name})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
        else:
            return "CPU"
    except ImportError:
        return "CPU (PyTorch not imported)"


def check_dataset(raw_dir: str) -> tuple[int, int]:
    """Count files and total size in raw dataset directory."""
    p = Path(raw_dir)
    if not p.exists():
        return 0, 0
    md_files = list(p.rglob("*.md")) + list(p.rglob("*.markdown"))
    total_size = sum(f.stat().st_size for f in md_files)
    return len(md_files), total_size


def main():
    print("=" * 60)
    print("Enterprise RAG Pipeline - Infrastructure & System Healthcheck")
    print("=" * 60)

    # 1. Hardware Accelerator
    accel = check_accelerator()
    print(f"[*] Compute Device     : {accel}")

    # 2. Dataset
    doc_count, doc_bytes = check_dataset(settings.data_raw_dir)
    if doc_count > 0:
        print(f"[OK] Knowledge Base     : Found {doc_count} markdown documents ({doc_bytes / 1024:.1f} KB)")
    else:
        print(f"[WARN] Knowledge Base   : No documents found in '{settings.data_raw_dir}'. Run 'python scripts/download_docs.py'.")

    # 3. Qdrant
    qdrant_ok = check_qdrant(settings.qdrant_url)
    if qdrant_ok:
        print(f"[OK] Qdrant Vector DB   : Reachable at {settings.qdrant_url}")
    else:
        print(f"[INFO] Qdrant Vector DB : Not reachable at {settings.qdrant_url} (Standalone/In-Memory fallback available)")

    # 4. Redis
    redis_ok = check_redis(settings.redis_url)
    if redis_ok:
        print(f"[OK] Redis Cache        : Reachable at {settings.redis_url}")
    else:
        print(f"[INFO] Redis Cache      : Not reachable at {settings.redis_url} (In-Memory Semantic Cache fallback available)")

    print("-" * 60)
    print("Health check completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
