"""Text cleaning and normalization utilities for Markdown technical documentation."""
import re


def clean_markdown(text: str) -> str:
    """
    Clean raw markdown document text while preserving code blocks and technical details.
    
    Operations:
    1. Strip YAML frontmatter (--- ... ---)
    2. Strip HTML comments (<!-- ... -->)
    3. Strip MkDocs include directives ({* include ... *})
    4. Normalize MkDocs callouts (/// tip ... /// and !!! note ...)
    5. Strip heading anchor tags ({#anchor-name})
    6. Convert Markdown links [text](url) to 'text'
    7. Remove image tags ![alt](url)
    8. Normalize multiple consecutive blank lines
    """
    if not text:
        return ""

    # 1. Strip YAML frontmatter at start of document
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    # 2. Strip HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 3. Strip MkDocs include directives
    text = re.sub(r"\{\*\s*include\s+.*?\*\}", "", text)

    # 4. Normalize callout blocks (protect code blocks)
    # Convert '/// tip\n' or '/// note\n' to 'Tip:\n' or 'Note:\n', and strip closing '///'
    def replace_callout_start(match):
        kind = match.group(1).capitalize()
        title = match.group(2).strip() if match.group(2) else ""
        return f"{kind}: {title}\n" if title else f"{kind}:\n"

    text = re.sub(r"^\s*///\s*(\w+)(?:\s+(.*))?$", replace_callout_start, text, flags=re.MULTILINE)
    text = re.sub(r"^\s*///\s*$", "", text, flags=re.MULTILINE)

    # Convert '!!! note "Title"' or '!!! tip'
    def replace_admonition(match):
        kind = match.group(1).capitalize()
        title = match.group(2).strip().strip('"\'') if match.group(2) else ""
        return f"{kind}: {title}\n" if title else f"{kind}:\n"

    text = re.sub(r"^\s*!!!\s*(\w+)(?:\s+(.*))?$", replace_admonition, text, flags=re.MULTILINE)

    # 5. Strip heading anchors: e.g. "### Heading {#anchor-name}" -> "### Heading"
    text = re.sub(r"(#{1,6}\s+.*?)\s*\{#[^}]+\}", r"\1", text)

    # We must not strip links or images inside code blocks.
    # We split by code fences (```), clean outside, and reassemble.
    parts = re.split(r"(```[\s\S]*?```)", text)
    cleaned_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Code block: leave untouched
            cleaned_parts.append(part)
        else:
            # Prose text: remove images and convert links
            # Remove images: ![alt](url)
            p = re.sub(r"!\[(.*?)\]\([^)]+\)", r"[Image: \1]" if r"\1" else "", part)
            # Convert links: [anchor](url) -> anchor
            p = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", p)
            cleaned_parts.append(p)

    text = "".join(cleaned_parts)

    # Normalize excessive blank lines (more than 2 consecutive newlines -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
