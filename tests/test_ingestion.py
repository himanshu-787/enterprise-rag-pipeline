"""Unit tests for document cleaning, loading, chunking, and ingestion pipeline."""
import json
from pathlib import Path
from src.rag.ingestion.models import Document, Chunk
from src.rag.ingestion.cleaning import clean_markdown
from src.rag.ingestion.loader import _extract_title, _generate_doc_id
from src.rag.ingestion.chunker import ContextualChunker
from src.rag.ingestion.pipeline import IngestionPipeline


def test_clean_markdown_frontmatter():
    raw = "---\ntitle: Sample\nauthor: Test\n---\n\n# Real Content\nParagraph here."
    cleaned = clean_markdown(raw)
    assert "---" not in cleaned
    assert "author: Test" not in cleaned
    assert "# Real Content" in cleaned
    assert "Paragraph here." in cleaned


def test_clean_markdown_html_comments():
    raw = "# Title\n<!-- Some internal comment -->\nVisible text."
    cleaned = clean_markdown(raw)
    assert "<!--" not in cleaned
    assert "Some internal comment" not in cleaned
    assert "Visible text." in cleaned


def test_clean_markdown_callouts():
    raw = "/// tip\nThis is a helpful tip.\n///\n\nNormal text."
    cleaned = clean_markdown(raw)
    assert "Tip:" in cleaned
    assert "This is a helpful tip." in cleaned
    assert "///" not in cleaned


def test_clean_markdown_links_and_images():
    raw = "Here is a [FastAPI tutorial](https://fastapi.tiangolo.com) and an image ![logo](https://fastapi.tiangolo.com/img/logo.png)."
    cleaned = clean_markdown(raw)
    assert "FastAPI tutorial" in cleaned
    assert "https://fastapi.tiangolo.com" not in cleaned
    assert "logo" in cleaned


def test_clean_markdown_preserves_code_fences():
    raw = """# Header

```python
# This is a python comment
def test_func():
    return [x for x in (1, 2)]
```
"""
    cleaned = clean_markdown(raw)
    assert "```python" in cleaned
    assert "# This is a python comment" in cleaned
    assert "def test_func():" in cleaned


def test_loader_title_and_id():
    content = "# My Custom Page {#anchor}\n\nSome text."
    title = _extract_title(content, "Default")
    assert title == "My Custom Page"

    doc_id1 = _generate_doc_id("tutorial/path_params.md")
    doc_id2 = _generate_doc_id("tutorial/path_params.md")
    assert doc_id1 == doc_id2
    assert "path_params" in doc_id1


def test_chunker_heading_paths_and_code_comments():
    chunker = ContextualChunker(max_tokens=200, overlap_tokens=20, min_chunk_chars=30)
    doc_content = """# Main Title

## Section One

Paragraph in section one with descriptive explanation of the component.

```python
# A python comment that should not be a heading
x = 42
```

### Subsection 1A

Detailed subsection paragraph explaining the implementation details and usage.
"""
    doc = Document(
        id="test_doc_1",
        title="Main Title",
        source_path="test.md",
        content=doc_content
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    paths = [c.heading_path for c in chunks]
    assert any("Section One" in p for p in paths)
    assert any("Subsection 1A" in p for p in paths)
    # Ensure "# A python comment" was NOT made into a heading path
    for p in paths:
        assert "A python comment" not in p


def test_chunker_overlap():
    chunker = ContextualChunker(max_tokens=30, overlap_tokens=15, min_chunk_chars=20)
    doc_content = """# Title

## Long Section

Short paragraph alpha with unique words A1 A2 A3 and some extra descriptive text.

Short paragraph beta with unique words B1 B2 B3 and some extra descriptive text.

Short paragraph gamma with unique words C1 C2 C3 and some extra descriptive text.
"""
    doc = Document(
        id="overlap_doc",
        title="Title",
        source_path="overlap.md",
        content=doc_content
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    # Check that adjacent chunks share context prefix
    for c in chunks:
        assert c.context_text.startswith("Document: Title | Section: Title > Long Section")


def test_chunker_oversized_code_split():
    chunker = ContextualChunker(max_tokens=50, overlap_tokens=10)
    # Code block with 40 lines
    code_lines = [f"    line_{i} = {i} * 2" for i in range(40)]
    doc_content = "# Big Code\n\n```python\n" + "\n".join(code_lines) + "\n```\n"
    doc = Document(
        id="code_doc",
        title="Big Code",
        source_path="code.md",
        content=doc_content
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert c.char_count <= chunker.max_chars + 100


def test_pipeline_jsonl_roundtrip(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc_file = raw_dir / "sample.md"
    doc_file.write_text("# Sample Page\n\n## Section\n\nContent for testing pipeline.\n", encoding="utf-8")

    out_file = tmp_path / "chunks.jsonl"
    pipeline = IngestionPipeline(max_tokens=150, overlap_tokens=20)
    stats = pipeline.run(raw_dir=str(raw_dir), output_file=str(out_file))

    assert stats["total_docs"] == 1
    assert stats["total_chunks"] >= 1
    assert Path(stats["output_file"]).exists()

    # Read back and validate JSONL schema
    with open(out_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == stats["total_chunks"]
        first_chunk = Chunk.model_validate_json(lines[0])
        assert first_chunk.doc_id.startswith("sample_")
        assert "Sample Page" in first_chunk.context_text
