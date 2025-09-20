# ai/react_agent/tools.py
from langchain.tools import tool
from typing import Any
import json
from pathlib import Path

from ai.rag.run_retriever import run_retrieval

# Path to your Chroma vector store
VECTOR_DIR = Path("./ai/data/chroma_db").resolve()  

# Default reranker model (can be changed to any HF cross-encoder)
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@tool
def retrieve_from_vector_store(query: str, top_k: int = 5, rerank: bool = True) -> str:
    """
    Tool for retrieving top-k relevant chunks from the Chroma vector store with optional reranking.
    
    Returns a JSON string containing a list of results; each result is an object
    with `text` and `metadata` keys. The agent expects this tool to be the
    single source-of-truth for financial evidence.
    """
    result = run_retrieval(
        query=query,
        top_k=top_k,
        fetch_k=20,  # fetch more candidates before reranking
        persist_dir=str(VECTOR_DIR),
        collection_name="faq_collection", 
        embedding_backend="google",  # or "google" if you used Google embeddings
        embedding_model="embedding-001",  # or your chosen model
        reranker_model=DEFAULT_RERANKER,
        rerank=rerank
    )

    # If result is an error string, return it as JSON
    if isinstance(result, str):
        return json.dumps({"error": result})
    
    return json.dumps(result)
