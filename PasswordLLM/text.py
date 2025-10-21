from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def load_vector_stores(vector_stores_path="vector_stores"):
    """Load all available vector stores"""
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={'device': 'cuda'}
    )
    
    vector_stores = {}
    for content_type in os.listdir(vector_stores_path):
        if content_type.startswith("5e_"):
            store_path = os.path.join(vector_stores_path, content_type)
            vector_stores[content_type.replace("_", " ")] = Chroma(
                persist_directory=store_path,
                embedding_function=embeddings
            )
    
    return vector_stores

stores = load_vector_stores()
store = stores["5e Classes"]
docs = store.similarity_search("fighter", k=2)

for i, doc in enumerate(docs):
    print(f"--- Chunk {i} ---")
    print(doc.page_content)

