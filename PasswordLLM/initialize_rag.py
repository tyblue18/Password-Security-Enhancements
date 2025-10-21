# initialize_rag.py
from rag_module import initialize_rag

# Initialize the RAG chain and save the vector store to disk
initialize_rag(save_path="vector_store")