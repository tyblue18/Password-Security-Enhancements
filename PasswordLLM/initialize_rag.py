# initialize_rag.py
from rag_module import initialize_rag

# Initialize the RAG chain and save the vectors to disk
if __name__ == "__main__":
    print(" Initializing Password Security vector stores...")
    print("This may take a few minutes...")
    initialize_rag(save_path="vector_stores")
    print("✅ Initialization complete!")
