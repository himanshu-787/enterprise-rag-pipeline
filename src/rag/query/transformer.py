"""Query transformation module: expansion, decomposition, and HyDE."""
import re
from typing import List, Optional

from src.rag.query.models import QueryTransformationResult

TECHNICAL_EXPANSIONS = {
    "auth": ["OAuth2PasswordBearer authentication", "security token header"],
    "authenticate": ["OAuth2PasswordBearer", "security token authentication"],
    "token": ["OAuth2 tokenUrl bearer token", "JWT access token"],
    "validate": ["Query validation min_length max_length pattern", "Pydantic Field constraints"],
    "validation": ["Query parameter string validation", "Pydantic BaseModel data validation"],
    "db": ["SQLAlchemy session yield get_db", "database connection lifecycle"],
    "database": ["SQLAlchemy SessionLocal yield db", "relational database session"],
    "session": ["SessionLocal yield db close", "database session dependency"],
    "cors": ["CORSMiddleware allow_origins allow_credentials", "cross-origin resource sharing"],
    "background": ["BackgroundTasks add_task", "asynchronous background execution"],
    "task": ["BackgroundTasks add_task notification", "asynchronous background tasks"],
    "error": ["HTTPException status_code detail", "custom exception_handler JSONResponse"],
    "exception": ["HTTPException status_code detail", "exception_handler Request"],
    "test": ["TestClient pytest httpx", "testing path operation functions"],
    "testing": ["TestClient app client.get", "pytest fixtures TestClient"],
    "dependency": ["Depends common_parameters", "dependency injection system"],
    "dependencies": ["sub-dependencies hierarchical graph", "Depends use_cache"],
}


class QueryTransformer:
    """
    Transforms user queries to optimize retrieval recall and precision.
    Capabilities:
    1. Multi-query expansion (technical phrasing and synonym mapping)
    2. Sub-query decomposition (splitting compound multi-intent questions)
    3. HyDE (Hypothetical Document Embeddings)
    """

    def __init__(self, enable_hyde: bool = True):
        self.enable_hyde = enable_hyde

    def expand_query(self, query: str) -> List[str]:
        """Generate technical and synonym variations of the query."""
        expansions = []
        tokens = re.findall(r"\w+", query.lower())

        matched_expansions = []
        for t in tokens:
            if t in TECHNICAL_EXPANSIONS:
                matched_expansions.extend(TECHNICAL_EXPANSIONS[t])

        for exp in matched_expansions[:3]:
            # Create augmented query variant
            augmented = f"{query} {exp}"
            if augmented not in expansions:
                expansions.append(augmented)

        return expansions

    def decompose_query(self, query: str) -> List[str]:
        """Decompose compound multi-intent queries into separate sub-queries."""
        # Split on conjunctions like " and how to ", " as well as ", " along with "
        split_pattern = r"\s+(?:and(?:\s+how\s+to|\s+also|\s+can\s+i)?|as\s+well\s+as|along\s+with)\s+"
        parts = re.split(split_pattern, query, flags=re.IGNORECASE)

        sub_queries = []
        if len(parts) > 1:
            for p in parts:
                p_clean = p.strip().rstrip("?").strip()
                if len(p_clean) > 8:
                    if not p_clean.lower().startswith(("how", "what", "where", "why", "can")):
                        p_clean = f"How to {p_clean}"
                    sub_queries.append(p_clean + "?")

        return sub_queries

    def generate_hyde(self, query: str) -> str:
        """
        Generate a hypothetical technical document response (HyDE).
        Simulates technical documentation prose and code examples matching the query.
        """
        clean_q = query.strip().rstrip("?")
        tokens = re.findall(r"\w+", clean_q.lower())

        code_snippet = "from fastapi import FastAPI\n\napp = FastAPI()\n"
        if "query" in tokens or "validate" in tokens:
            code_snippet += "from fastapi import Query\n\n@app.get('/items/')\ndef read_items(q: str = Query(..., min_length=3)):\n    return {'q': q}\n"
        elif "auth" in tokens or "token" in tokens or "security" in tokens:
            code_snippet += "from fastapi.security import OAuth2PasswordBearer\n\noauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')\n"
        elif "background" in tokens or "task" in tokens:
            code_snippet += "from fastapi import BackgroundTasks\n\ndef write_log(msg: str):\n    pass\n"
        elif "cors" in tokens:
            code_snippet += "from fastapi.middleware.cors import CORSMiddleware\n\napp.add_middleware(CORSMiddleware, allow_origins=['*'])\n"
        else:
            code_snippet += "@app.get('/')\ndef root():\n    return {'status': 'ok'}\n"

        hyde_text = (
            f"Documentation and Guide for: {clean_q}\n\n"
            f"In FastAPI, this feature is implemented using declarative type annotations "
            f"and framework components.\n\n"
            f"```python\n{code_snippet}```\n"
        )
        return hyde_text

    def transform(self, query: str) -> QueryTransformationResult:
        """
        Execute full transformation pipeline.
        
        Returns:
            QueryTransformationResult with original, expansions, sub-queries, and HyDE.
        """
        expansions = self.expand_query(query)
        sub_queries = self.decompose_query(query)
        hyde_doc = self.generate_hyde(query) if self.enable_hyde else None

        # Build deduplicated list of search queries
        all_q = [query]
        for q in expansions:
            if q not in all_q:
                all_q.append(q)
        for sq in sub_queries:
            if sq not in all_q:
                all_q.append(sq)

        return QueryTransformationResult(
            original_query=query,
            expanded_queries=expansions,
            sub_queries=sub_queries,
            hypothetical_document=hyde_doc,
            all_queries=all_q,
        )
