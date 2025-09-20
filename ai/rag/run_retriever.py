from pathlib import Path
import traceback
import json
from ai.rag.get_embeddings_model import get_embeddings
from ai.rag.retriever import RAGRetriever

def run_retrieval(
    query: str,
    top_k: int = 2,
    fetch_k: int = 5,
    persist_dir: str = "./data/chroma_db",
    collection_name: str = "faq_collection",
    embedding_backend: str = "hf",
    embedding_model: str = "BAAI/bge-small-en",
    reranker_model: str = None,  # e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank: bool = True
):
    """
    Run a retrieval query against the persisted Chroma collection with optional reranking.

    Parameters:
        query: the search query
        top_k: how many results to return after reranking
        fetch_k: how many results to fetch initially before reranking
        reranker_model: cross-encoder model for reranking (if None, no reranking)
        rerank: whether to perform reranking
    """
    try:
        # Get embeddings
        emb = get_embeddings(backend=embedding_backend, model_name=embedding_model)
        print(persist_dir)
        # Initialize retriever with optional reranker
        retriever = RAGRetriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding=emb,
            reranker_model=reranker_model
        )
        
        # Retrieve results
        results = retriever.retrieve(query, top_k=top_k, fetch_k=fetch_k, rerank=rerank)
        return results

    except Exception as e:
        return f"Error during retrieval: {str(e)}\n{traceback.format_exc()}"

# Example usage:
# if __name__ == '__main__':
#     q = "How can I create an account?"
#     print(run_retrieval(q, top_k=3, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"))
