from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

emb = GoogleGenerativeAIEmbeddings(model="embedding-001", transport="rest")
vectors = emb.embed_documents(["Hello world", "Test text"])
print(vectors)