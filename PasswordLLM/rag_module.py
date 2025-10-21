from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import glob
from collections import defaultdict

def initialize_password_stores(base_path="data", save_path="vector_stores"):
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={'device': 'cuda'}
    )

    os.makedirs(save_path, exist_ok=True)

    for content_dir in os.listdir(base_path):
        if not content_dir in ["common_password", "data_breach", "weak_passwords", "security_rules"]:
            continue

        content_type = content_dir
        dir_path = os.path.join(base_path, content_dir)
        print(f"[INFO] Processing {content_type}...")

        # Load text files manually for better reliability
        docs = []
        import glob
        
        # Load .txt files
        txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": txt_file}
                        ))
            except Exception as e:
                print(f"[ERROR] Could not load {txt_file}: {e}")
        
        # Load .csv files
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": csv_file}
                        ))
            except Exception as e:
                print(f"[ERROR] Could not load {csv_file}: {e}")
        
        print(f"[DEBUG] Found {len(docs)} documents in {content_type}")
        if len(docs) == 0:
            print(f"[WARNING] No documents found in {dir_path}")
            continue

        # Group by filename
        grouped_docs = defaultdict(list)
        for doc in docs:
            filename = os.path.basename(doc.metadata["source"])
            grouped_docs[filename].append(doc)

        final_docs = []

        for filename, file_docs in grouped_docs.items():
            title = os.path.splitext(filename)[0].replace("_", " ").strip().title()
            full_text = "\n\n".join([p.page_content for p in file_docs])

            # Skip empty documents
            if not full_text.strip():
                print(f"[WARNING] Skipping empty document: {filename}")
                continue

            metadata = {
                "title": title,
                "source_file": filename,
                "content_type": content_type,
                "category": content_type  # For password analysis categorization
            }

            final_docs.append(Document(page_content=full_text, metadata=metadata))
            print(f"[INFO] Added document: {title} ({len(full_text)} chars)")

        # Skip if no valid documents found
        if len(final_docs) == 0:
            print(f"[WARNING] No valid documents found for {content_type}, skipping vector store creation")
            continue

        # Save to Chroma vector store
        store_path = os.path.join(save_path, content_type.replace(" ", "_"))
        Chroma.from_documents(
            documents=final_docs,
            embedding=embeddings,
            persist_directory=store_path
        )

        print(f"[INFO] Created vector store for {content_type} with {len(final_docs)} documents")

def initialize_rag(save_path="vector_stores"):
    """Wrapper function for backward compatibility"""
    return initialize_password_stores(save_path=save_path)

if __name__ == "__main__":
    initialize_password_stores()
    print("done")

