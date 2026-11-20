"""Document loader for technical markdown repositories."""
import os
import re
import hashlib
from pathlib import Path
from typing import List

from src.rag.ingestion.models import Document


def _extract_title(content: str, default_title: str) -> str:
    """Extract first H1 heading from markdown or use default."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            # Strip anchors like {#anchor}
            title = re.sub(r"\{#[^}]+\}", "", title).strip()
            if title:
                return title
    return default_title


def _generate_doc_id(rel_path: str) -> str:
    """Generate a clean, deterministic doc ID from normalized relative path."""
    normalized = rel_path.replace("\\", "/").strip("/")
    path_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    stem = Path(normalized).stem.replace(" ", "_").lower()
    return f"{stem}_{path_hash}"


def load_documents(raw_dir: str) -> List[Document]:
    """
    Recursively scan directory for Markdown files and return Document models.
    
    Args:
        raw_dir: Path to raw data directory.
        
    Returns:
        List of loaded Document models sorted by source path.
    """
    base_path = Path(raw_dir)
    if not base_path.exists():
        return []

    documents = []
    # Collect .md and .markdown files
    md_files = sorted(
        list(base_path.rglob("*.md")) + list(base_path.rglob("*.markdown")),
        key=lambda p: str(p).lower()
    )

    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="replace")

        rel_path = str(file_path.relative_to(base_path)).replace("\\", "/")
        stem_title = file_path.stem.replace("_", " ").replace("-", " ").title()
        title = _extract_title(content, stem_title)
        doc_id = _generate_doc_id(rel_path)

        documents.append(
            Document(
                id=doc_id,
                title=title,
                source_path=rel_path,
                content=content,
                metadata={
                    "file_size": file_path.stat().st_size,
                    "rel_path": rel_path,
                }
            )
        )

    return documents
