from pathlib import Path
from ai.rag.loader import RAGLoader
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == '__main__':
    DATA_DIR = Path(os.getenv('DATA_DIR', './ai/data')).resolve()
    PERSIST_DIR = Path(os.getenv('CHROMA_DIR', './ai/data/chroma_db')).resolve()
    COLLECTION = os.getenv('CHROMA_COLLECTION', 'faq_collection')
    EMB_BACKEND = os.getenv('EMBED_BACKEND', 'google')
    EMB_MODEL = os.getenv('EMBED_MODEL', 'embedding-001')
    
    print(DATA_DIR)

    loader = RAGLoader(
        data_dir=DATA_DIR,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION,
        embedding_backend=EMB_BACKEND,
        embedding_model=EMB_MODEL,
    )

    vect = loader.build_chroma_collection(persist=True)
    print('Chroma vectorstore built and persisted')
