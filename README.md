# FAQ RAG

This project helps you **build and persist a Chroma vector database** from your local documents.  
It uses a configurable chunking strategy and embeddings backend to prepare your data for Retrieval-Augmented Generation (RAG).

---

## Features
- Load documents from a data directory
- Chunk text using `RecursiveCharacterTextSplitter`
- Generate embeddings using Google or other backends
- Persist vectors in ChromaDB for later retrieval

---

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### 2. Create and configure environment variables  
Create a `.env` file in the project root:

```env
DATA_DIR=./ai/data/raw
CHROMA_DIR=./ai/data/chroma_db
CHROMA_COLLECTION=faq_collection
EMBED_BACKEND=google
EMBED_MODEL=embedding-001
GOOGLE_API_KEY=<your-api-key>
GROQ_API_KEY=<your-api=key>
GROQ_MODEL=qwen/qwen3-32b
```

You can change these paths and values as needed.

---

## Installation

### Option A: Using pip
```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

pip install -r requirements.txt
```

### Option B: Using uv
```bash
uv venv
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)

uv pip install -r requirements.txt
```

---

## Usage
Run the vector store builder:

```bash
python build_vector_store.py
```

Expected output:
```
/absolute/path/to/ai/data/chroma_db
Chroma vectorstore built and persisted
```
```bash
streamlit run streamlit_main.py
```

Expected output:
```
Streamlit app running...
```

---

## Notes
- Place all your documents (CSV) in `ai/data`
- The vector store will be persisted under `ai/data/chroma_db`
- To rebuild with new documents, just re-run `build_vector_store.py`
- Tho a sample vector store is already kept in the `ai/data/`
---

## License
MIT License
