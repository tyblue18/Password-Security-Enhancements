from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os, glob, torch, hashlib, shutil
from collections import defaultdict
from pathlib import Path

def _hash_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()

def _load_plain_files(dir_path: str):
    docs = []
    for pattern in ("*.txt", "*.csv"):
        for f in glob.glob(os.path.join(dir_path, pattern)):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    content = fh.read()
                if content.strip():
                    docs.append(Document(page_content=content, metadata={"source": f}))
            except Exception as e:
                print(f"[ERROR] Could not load {f}: {e}")
    return docs

def initialize_password_stores(base_path="data", save_path="vector_stores",
                               chunk_size=1000, chunk_overlap=150,
                               clean_existing=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    os.makedirs(save_path, exist_ok=True)
    allow = {"common_password", "data_breach", "weak_passwords", "security_rules"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )

    for content_dir in os.listdir(base_path):
        if content_dir not in allow:
            continue

        dir_path = os.path.join(base_path, content_dir)
        if not os.path.isdir(dir_path):
            continue

        print(f"[INFO] Processing {content_dir}...")

        raw_docs = _load_plain_files(dir_path)
        print(f"[DEBUG] Found {len(raw_docs)} raw documents in {content_dir}")
        if not raw_docs:
            print(f"[WARNING] No documents found in {dir_path}")
            continue

        grouped = defaultdict(list)
        for d in raw_docs:
            grouped[os.path.basename(d.metadata["source"])].append(d)

        final_docs = []
        for filename, file_docs in grouped.items():
            title = Path(filename).stem.replace("_", " ").strip().title()
            full_text = "\n\n".join(d.page_content for d in file_docs).strip()
            if not full_text:
                print(f"[WARNING] Skipping empty document: {filename}")
                continue

            base_meta = {
                "title": title,
                "source_file": filename,
                "content_type": content_dir,
                "category": content_dir,
            }

            # Split into chunks with metadata
            chunks = splitter.create_documents([full_text], metadatas=[base_meta])
            for i, ch in enumerate(chunks):
                ch.metadata["chunk_index"] = i
                ch.metadata["source"] = os.path.join(dir_path, filename)
                ch.page_content = ch.page_content.strip()
                ch.metadata["id"] = _hash_id(ch.metadata["source"], i)

            final_docs.extend(chunks)
            print(f"[INFO] {title}: split into {len(chunks)} chunks")

        if not final_docs:
            print(f"[WARNING] No valid chunks for {content_dir}, skipping")
            continue

        store_dir = os.path.join(save_path, content_dir.replace(" ", "_"))
        collection_name = f"pwd_{content_dir}"

        if clean_existing and os.path.exists(store_dir):
            print(f"[CLEAN] Removing existing store at {store_dir}")
            shutil.rmtree(store_dir, ignore_errors=True)

        ids = [d.metadata["id"] for d in final_docs]
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=store_dir,
        )
        vectorstore.add_documents(final_docs, ids=ids)
        vectorstore.persist()

        print(f"[INFO] Built vector store '{collection_name}' at {store_dir} with {len(final_docs)} chunks")

def initialize_rag(save_path="vector_stores"):
    return initialize_password_stores(save_path=save_path)

if __name__ == "__main__":
    initialize_password_stores()
    print("done")
