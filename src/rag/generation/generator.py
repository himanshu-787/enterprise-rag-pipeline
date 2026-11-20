"""LLM answer generator with strict citations and hallucination guardrails."""
import re
import json
import urllib.request
from typing import List, Dict, Any, Tuple, Optional

from src.rag.config import settings
from src.rag.retrieval.models import RetrievedChunk

SYSTEM_PROMPT = """You are an expert technical documentation assistant.
Answer the user's question using ONLY the provided technical documentation context.
Follow these rules strictly:
1. Every factual statement must include a citation in the format [Doc {index}], referencing the context block index.
2. If the context does not contain enough information to answer the question, state: 'The provided technical documentation does not contain sufficient details to answer this question.'
3. Do not invent or assume parameters, functions, or configurations not present in the context.
4. Keep the explanation concise and include relevant code snippets if present in the context."""


class AnswerGenerator:
    """Generates answers with citations using LLM or deterministic fallback synthesizer."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        llm_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.hf_token = hf_token or settings.hf_token
        self.llm_model = llm_model or settings.llm_model
        self.openai_api_key = openai_api_key or settings.openai_api_key

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into indexed context blocks."""
        formatted_blocks = []
        for i, c in enumerate(chunks, start=1):
            block = (
                f"[Doc {i}] Title: {c.title} | Section: {c.heading_path}\n"
                f"{c.text.strip()}"
            )
            formatted_blocks.append(block)
        return "\n\n".join(formatted_blocks)

    def _extractive_synthesize(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Extractive synthesis fallback for offline or zero-API-key environments.
        Synthesizes response directly from top context passages with precise citations.
        """
        if not chunks:
            return "The provided technical documentation does not contain sufficient details to answer this question."

        top = chunks[0]
        # Check if top chunk contains code block
        code_blocks = re.findall(r"```[\s\S]*?```", top.text)
        has_code = bool(code_blocks)

        summary_lines = []
        for line in top.text.splitlines():
            line_s = line.strip()
            if line_s and not line_s.startswith("```") and not line_s.startswith("#"):
                summary_lines.append(line_s)
                if len(summary_lines) >= 2:
                    break

        first_point = " ".join(summary_lines) if summary_lines else top.text[:200]

        answer = (
            f"Based on the technical documentation, in {top.heading_path}: {first_point} [Doc 1]."
        )
        if len(chunks) > 1:
            second = chunks[1]
            sec_lines = [l.strip() for l in second.text.splitlines() if l.strip() and not l.startswith("```") and not l.startswith("#")]
            if sec_lines:
                answer += f"\n\nAdditionally, {sec_lines[0]} [Doc 2]."

        if has_code:
            answer += f"\n\nExample implementation from [Doc 1]:\n{code_blocks[0]}"

        return answer

    def _call_hf_api(self, prompt: str) -> Optional[str]:
        """Call Hugging Face Inference API if token is provided."""
        if not self.hf_token:
            return None
        url = f"https://api-inference.huggingface.co/models/{self.llm_model}"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512, "temperature": 0.1},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]["generated_text"]
        except Exception:
            pass
        return None

    def generate_answer(
        self,
        query: str,
        contexts: List[RetrievedChunk],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate answer with citations given query and retrieved contexts.
        
        Args:
            query: User query.
            contexts: List of retrieved context chunks.
            
        Returns:
            Tuple of (answer_string, list_of_citations).
        """
        if not contexts:
            return (
                "The provided technical documentation does not contain sufficient details to answer this question.",
                [],
            )

        citations = []
        for i, c in enumerate(contexts, start=1):
            citations.append({
                "citation_id": f"[Doc {i}]",
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "heading_path": c.heading_path,
                "score": c.score,
            })

        formatted_context = self._format_context(contexts)
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n\n"
            f"<|context|>\n{formatted_context}\n\n"
            f"<|user|>\n{query}\n\n"
            f"<|assistant|>\n"
        )

        hf_answer = self._call_hf_api(full_prompt)
        if hf_answer:
            return hf_answer.strip(), citations

        # Deterministic extractive synthesizer fallback
        answer = self._extractive_synthesize(query, contexts)
        return answer, citations
