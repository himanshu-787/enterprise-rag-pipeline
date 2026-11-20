"""Heading-aware contextual document chunker with code block preservation."""
import re
from typing import List, Tuple
from src.rag.ingestion.models import Document, Chunk


class ContextualChunker:
    """
    Context-aware Markdown chunker that preserves heading hierarchy and code blocks.
    
    Features:
    - Ignores '#' characters inside code fences (prevents python comments as headings)
    - Maintains hierarchical heading paths (e.g. "Tutorial > Query Parameters > Validation")
    - Packs blocks into chunks respecting token budget (chars // 4)
    - Adds sliding window overlap between consecutive chunks in the same section
    - Prepends document title and section path to chunk context_text
    - Generates stable, deterministic chunk IDs
    """

    def __init__(
        self,
        max_tokens: int = 350,
        overlap_tokens: int = 40,
        min_chunk_chars: int = 50,
    ):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * 4
        self.overlap_tokens = overlap_tokens
        self.overlap_chars = overlap_tokens * 4
        self.min_chunk_chars = min_chunk_chars

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Split a Document into an ordered list of contextual Chunks."""
        sections = self._split_into_sections(doc.content, doc.title)
        chunks: List[Chunk] = []
        chunk_index = 0

        for heading_path, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            blocks = self._split_into_blocks(section_text)
            packed_chunks = self._pack_blocks(blocks)

            for text_block in packed_chunks:
                text_block = text_block.strip()
                if len(text_block) < self.min_chunk_chars and chunks:
                    # Append very tiny trailing fragment to previous chunk if within budget
                    if len(chunks[-1].text) + len(text_block) <= self.max_chars:
                        chunks[-1].text = f"{chunks[-1].text}\n\n{text_block}"
                        chunks[-1].char_count = len(chunks[-1].text)
                        chunks[-1].token_count_est = chunks[-1].char_count // 4
                        chunks[-1].context_text = (
                            f"Document: {doc.title} | Section: {chunks[-1].heading_path}\n\n"
                            f"{chunks[-1].text}"
                        )
                        continue

                chunk_id = f"{doc.id}_c{chunk_index}"
                context_text = f"Document: {doc.title} | Section: {heading_path}\n\n{text_block}"
                char_count = len(text_block)
                token_count = char_count // 4

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc.id,
                        chunk_index=chunk_index,
                        title=doc.title,
                        heading_path=heading_path,
                        text=text_block,
                        context_text=context_text,
                        char_count=char_count,
                        token_count_est=token_count,
                        metadata={
                            "source_path": doc.source_path,
                            "heading_path": heading_path,
                        }
                    )
                )
                chunk_index += 1

        return chunks

    def _split_into_sections(self, content: str, doc_title: str) -> List[Tuple[str, str]]:
        """
        Split markdown content into sections based on headings (#, ##, ###),
        ignoring any '#' characters inside ``` code blocks.
        """
        lines = content.splitlines()
        in_code_block = False
        heading_stack: dict[int, str] = {}
        sections: List[Tuple[str, List[str]]] = []

        current_lines: List[str] = []
        current_path = doc_title

        for line in lines:
            stripped = line.strip()
            # Toggle code block
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue

            if not in_code_block and stripped.startswith("#"):
                match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
                if match:
                    level = len(match.group(1))
                    h_text = match.group(2).strip()

                    # Save current section if it has content
                    if current_lines:
                        sections.append((current_path, "\n".join(current_lines)))
                        current_lines = []

                    # Update heading stack
                    # Remove all headings deeper than or equal to current level
                    to_remove = [lvl for lvl in heading_stack if lvl >= level]
                    for lvl in to_remove:
                        del heading_stack[lvl]

                    heading_stack[level] = h_text
                    sorted_levels = sorted(heading_stack.keys())
                    current_path = " > ".join([heading_stack[lvl] for lvl in sorted_levels])
                    continue

            current_lines.append(line)

        if current_lines:
            sections.append((current_path, "\n".join(current_lines)))

        return [(path, text) for path, text in sections if text.strip()]

    def _split_into_blocks(self, section_text: str) -> List[str]:
        """
        Split section text into discrete paragraph and code blocks.
        """
        blocks = []
        # Match code blocks or double-newline paragraphs
        parts = re.split(r"(```[\s\S]*?```)", section_text)
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            if i % 2 == 1:
                # Code block
                blocks.append(part.strip())
            else:
                # Prose: split into paragraphs
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", part) if p.strip()]
                blocks.extend(paragraphs)

        # Handle oversized code blocks by splitting them on lines
        final_blocks = []
        for b in blocks:
            if len(b) > self.max_chars and b.startswith("```"):
                sub_blocks = self._split_oversized_code(b)
                final_blocks.extend(sub_blocks)
            elif len(b) > self.max_chars:
                # Long prose: split by sentence or line
                sub_blocks = self._split_oversized_prose(b)
                final_blocks.extend(sub_blocks)
            else:
                final_blocks.append(b)

        return final_blocks

    def _split_oversized_code(self, code_block: str) -> List[str]:
        """Split a code block that exceeds max characters along line boundaries."""
        lines = code_block.splitlines()
        lang_line = lines[0] if lines else "```"
        body_lines = lines[1:-1] if len(lines) > 2 else lines
        result = []
        curr: List[str] = []
        curr_len = 0

        for line in body_lines:
            if curr_len + len(line) + 1 > (self.max_chars - 20) and curr:
                chunk_str = f"{lang_line}\n" + "\n".join(curr) + "\n```"
                result.append(chunk_str)
                curr = []
                curr_len = 0
            curr.append(line)
            curr_len += len(line) + 1

        if curr:
            chunk_str = f"{lang_line}\n" + "\n".join(curr) + "\n```"
            result.append(chunk_str)

        return result

    def _split_oversized_prose(self, prose: str) -> List[str]:
        """Split oversized prose into sentences or line boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", prose)
        result = []
        curr: List[str] = []
        curr_len = 0

        for s in sentences:
            if curr_len + len(s) + 1 > self.max_chars and curr:
                result.append(" ".join(curr))
                curr = []
                curr_len = 0
            curr.append(s)
            curr_len += len(s) + 1

        if curr:
            result.append(" ".join(curr))

        return result

    def _pack_blocks(self, blocks: List[str]) -> List[str]:
        """
        Pack blocks into chunks up to max_chars with overlap from previous chunk.
        """
        if not blocks:
            return []

        chunks: List[str] = []
        curr_blocks: List[str] = []
        curr_chars = 0

        for block in blocks:
            block_len = len(block)
            if curr_chars + block_len + 2 > self.max_chars and curr_blocks:
                # Emit current chunk
                chunk_text = "\n\n".join(curr_blocks)
                chunks.append(chunk_text)

                # Calculate overlap: take tail blocks from current chunk
                overlap_candidates: List[str] = []
                overlap_accum = 0
                for prev in reversed(curr_blocks):
                    if overlap_accum + len(prev) + 2 <= self.overlap_chars:
                        overlap_candidates.insert(0, prev)
                        overlap_accum += len(prev) + 2
                    else:
                        break

                curr_blocks = overlap_candidates + [block]
                curr_chars = sum(len(b) for b in curr_blocks) + 2 * max(0, len(curr_blocks) - 1)
            else:
                curr_blocks.append(block)
                curr_chars += block_len + 2

        if curr_blocks:
            chunks.append("\n\n".join(curr_blocks))

        return chunks
