# ai/rag/retriever.py
import logging
from typing import List, Dict, Optional
from langchain_chroma import Chroma

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "faq_collection",
        embedding=None,
        reranker_model: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = embedding
        self.db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embedding, collection_name=self.collection_name)

        self.reranker = None
        if reranker_model and CrossEncoder is not None:
            logger.info("Loading cross-encoder reranker: %s", reranker_model)
            self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20, rerank: bool = True) -> List[Dict]:
        docs = self.db.similarity_search(query, k=fetch_k)

        if rerank and self.reranker is not None and docs:
            pairs = [[query, d.page_content] for d in docs]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
            docs = [d for d, _ in ranked]
        else:
            docs = docs[:top_k]

        results = [{"text": d.page_content, "metadata": d.metadata} for d in docs]
        return results
