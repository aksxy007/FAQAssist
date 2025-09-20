from typing import Any
from dotenv import load_dotenv

load_dotenv()


def get_embeddings(backend: str = "hf", model_name: str = "BAAI/bge-small-en", google_rest: bool = True) -> Any:
    backend = (backend or "hf").lower()

    if backend == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except Exception:
            raise

        if google_rest:
            return GoogleGenerativeAIEmbeddings(model=model_name, transport="rest")
        return GoogleGenerativeAIEmbeddings(model=model_name)

    elif backend == "hf":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception:
            from sentence_transformers import SentenceTransformer

            class SentenceTransformerWrapper:
                def __init__(self, name: str):
                    self.model = SentenceTransformer(name)

                def embed_documents(self, texts):
                    return self.model.encode(texts, convert_to_numpy=True)

                def embed_query(self, query: str):
                    return self.model.encode([query], convert_to_numpy=True)[0]

            return SentenceTransformerWrapper(model_name)

    else:
        raise ValueError("Unsupported embedding backend. Choose 'hf' or 'google'.")
