# ai/rag/chunker.py
import pandas as pd
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class FAQChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n", ".", "?", "!"]
        )

    def chunk_csv(self, csv_path: str) -> List[Document]:
        df = pd.read_csv(csv_path)
        docs = []

        for _, row in df.iterrows():
            q = str(row.get("question", "") or "").strip()
            a = str(row.get("answer", "") or "").strip()
            if not q or not a:
                continue

            if len(a) <= self.chunk_size:  # use your own var
                text = f"Q: {q}\nA: {a}"
                docs.append(Document(page_content=text, metadata={"question": q}))
            else:
                for i, chunk in enumerate(self.splitter.split_text(a)):
                    text = f"Q: {q}\nA: {chunk}"
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"question": q, "chunk_index": i},
                        )
                    )
        return docs
