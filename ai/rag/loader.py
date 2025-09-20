# ai/rag/loader.py
import logging
from pathlib import Path
from typing import List
import numpy as np
from langchain.schema import Document
from langchain_chroma import Chroma
from ai.rag.chunker import FAQChunker  # use CSV chunker
from ai.rag.get_embeddings_model import get_embeddings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class RAGLoader:
    def __init__(
        self,
        data_dir: Path,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "faq_collection",
        embedding_backend: str = "hf",
        embedding_model: str = "BAAI/bge-small-en",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.chunker = FAQChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def collect_csvs(self) -> List[Path]:
        csvs = sorted(self.data_dir.rglob("*.csv"))
        logger.info("Found %d CSV files in %s", len(csvs), self.data_dir)
        return csvs

    def build_chroma_collection(self, persist: bool = True):
        embeddings = get_embeddings(
            backend=self.embedding_backend, model_name=self.embedding_model
        )
        logger.info(
            "Using embeddings backend='%s', model='%s'",
            self.embedding_backend,
            self.embedding_model,
        )

        all_docs: List[Document] = []
        csvs = self.collect_csvs()
        for csv in csvs:
            logger.info("Processing CSV: %s", csv)
            chunks = self.chunker.chunk_csv(str(csv))
            logger.info("  Extracted %d chunks from %s", len(chunks), csv.name)

            valid_chunks = [c for c in chunks if c.page_content.strip()]
            skipped = len(chunks) - len(valid_chunks)
            if skipped:
                logger.warning("  Skipped %d empty chunks in %s", skipped, csv.name)

            for c in valid_chunks:
                c.metadata.setdefault("file", csv.name)

            all_docs.extend(valid_chunks)

        logger.info("Total valid chunks to index: %d", len(all_docs))

        if all_docs:
            sample_texts = [d.page_content for d in all_docs[:2]]
            sample_vecs = embeddings.embed_documents(sample_texts)
            if isinstance(sample_vecs, np.ndarray):
                logger.info("Embeddings check passed, shape=%s", sample_vecs.shape)
            else:
                logger.warning("Embeddings check returned unexpected type: %s", type(sample_vecs))

        vect = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        logger.info("Chroma collection created with %d documents", len(all_docs))

        # if persist:
        #     vect.persist()
        #     logger.info(
        #         "Chroma persisted to %s (collection=%s)",
        #         self.persist_dir,
        #         self.collection_name,
        #     )

        return vect
