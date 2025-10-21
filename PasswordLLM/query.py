from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from difflib import get_close_matches
import contextlib
import time
import io
import os
import re


def detect_content_type(query: str) -> str:
    """Determine which vector stores to search based on password security question content"""
    query_lower = query.lower()
    
    def has_word(words):
        return any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in words)
    
    # Check for common password analysis
    if has_word([
        "common", "popular", "frequent", "top", "most used", "weak", "easy", 
        "simple", "dictionary", "list", "check password", "is this password"
    ]):
        return "common_password"
    
    # Check for breach-related queries
    if has_word([
        "breach", "leaked", "compromised", "stolen", "hacked", "pwned", 
        "haveibeenpwned", "data breach", "security incident"
    ]):
        return "data_breach"
    
    # Check for weak pattern analysis
    if has_word([
        "pattern", "keyboard", "sequential", "repeated", "qwerty", "123", 
        "weak pattern", "predictable", "substitution", "l33t"
    ]):
        return "weak_passwords"
    
    # Check for security rules and policies
    if has_word([
        "policy", "requirement", "rule", "standard", "compliance", "nist", 
        "pci", "iso", "strength", "criteria", "guidelines", "must contain"
    ]):
        return "security_rules"

    return None

import contextlib
import io

def silent_similarity_search(store, query, k=1000):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):  # Capture stdout silently
        return store.similarity_search(query, k=k)


def match_title_to_query(query: str, vector_stores: dict, content_type: str = None) -> str | None:
    query_lower = query.lower()
    known_titles = set()

    # If we have a content type, only search that store
    if content_type and content_type in vector_stores:
        stores_to_search = {content_type: vector_stores[content_type]}
        print("Searching {stores_to_search}")
    else:
        stores_to_search = vector_stores

    for store_name, store in stores_to_search.items():
        try:
            # Use similarity_search to pull more documents (including all titles)
            docs = silent_similarity_search(store, "dummy query", k=1000)
            for doc in docs:
                title = doc.metadata.get("title")
                if title:
                    known_titles.add(title.strip())
        except Exception as e:
            print(f"[DEBUG] Failed to extract metadata from {store_name}: {e}")

    #print(f"[DEBUG] Found {len(known_titles)} unique titles in {list(stores_to_search.keys())}")
    #print(f"[DEBUG] Known titles: {sorted(known_titles)}")

    # Try exact substring match first
    for title in known_titles:
        if re.search(rf'\b{re.escape(title.lower())}\b', query_lower):
            print(f"[DEBUG] Exact title match: {title}")
            return title

    # Try fuzzy match
    matches = get_close_matches(query_lower, [t.lower() for t in known_titles], n=1, cutoff=0.6)
    if matches:
        best_match = matches[0]
        for t in known_titles:
            if t.lower() == best_match:
                print(f"[DEBUG] Fuzzy title match: {t}")
                return t

    print("[DEBUG] No title match found.")
    return None


def load_vector_stores(vector_stores_path="vector_stores"):
    """Load all available vector stores"""
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={'device': 'cuda'}
    )
    
    vector_stores = {}
    password_categories = ["common_password", "data_breach", "weak_passwords", "security_rules"]
    
    for content_type in os.listdir(vector_stores_path):
        if content_type in password_categories:
            store_path = os.path.join(vector_stores_path, content_type)
            vector_stores[content_type] = Chroma(
                persist_directory=store_path,
                embedding_function=embeddings
            )
            print(f"[INFO] Loaded vector store: {content_type}")
    
    return vector_stores

def get_rag_response(query: str, vector_stores: dict) -> str:
    content_type = detect_content_type(query)
    title = match_title_to_query(query, vector_stores)  # New fuzzy matcher

    if content_type and content_type in vector_stores:
        print(f"Searching {content_type} vector store")
        if title:
            print(f"Filtering by title: {title}")
            docs = vector_stores[content_type].max_marginal_relevance_search(
                query, k=3, filter={"title": title}
            )
        else:
            docs = vector_stores[content_type].max_marginal_relevance_search(query, k=3)
    else:
        print("Searching all vector stores")
        docs = []
        for store_name, store in vector_stores.items():
            if title:
                store_docs = store.max_marginal_relevance_search(query, k=1, filter={"title": title})
            else:
                store_docs = store.max_marginal_relevance_search(query, k=1)
            docs.extend(store_docs)
            print(f"[DEBUG] Found {len(store_docs)} docs from {store_name}")

    # Limit context size to avoid overwhelming the model
    context_parts = []
    total_chars = 0
    max_context = 25000  # Increased to 25k characters for more context
    
    for doc in docs:
        if total_chars + len(doc.page_content) > max_context:
            # Truncate the last document to fit
            remaining = max_context - total_chars
            if remaining > 100:  # Only add if we have meaningful space left
                context_parts.append(doc.page_content[:remaining] + "...")
            break
        context_parts.append(doc.page_content)
        total_chars += len(doc.page_content)
    
    context = "\n\n".join(context_parts)
    print(f"[DEBUG] Retrieved {len(docs)} documents")
    print(f"[DEBUG] Context length: {len(context)} characters (limited to {max_context})")
    #print(context)

    prompt_template = PromptTemplate.from_template(
        """You are a password security analyzer. Use ONLY the provided context to answer questions.
        Do NOT use your general training knowledge. Base your answer ONLY on the security rules, 
        password lists, and breach data provided in the context below.
        
        When analyzing a specific password:
        1. Check if it appears in the common password lists in the context
        2. Evaluate it against the security rules provided in the context  
        3. Look for weak patterns mentioned in the context
        4. Provide a numerical score (1-10) based on the criteria in the context
        
        For the password 'faithful' specifically:
        - Length: 8 characters
        - Character types: lowercase letters only
        - Check against the security requirements in the context
        - Rate it 1-10 based on the rules provided
        
        Context: {context}
        
        Question: {question}
        
        Answer based ONLY on the context above:"""
    )
    
    rag_prompt = prompt_template.format(context=context, question=query)
    # return ChatOllama(model="llama3.2:1b").invoke(rag_prompt).content.strip()
    print("\nAnswer:", end=' ', flush=True)  # Begin output immediately
    llm = ChatOllama(model="llama3.2:1b", stream=True)
    
    for chunk in llm.stream(rag_prompt):
        print(chunk.content, end='', flush=True)
        #time.sleep(0.015)

    print()  # newline after completion

# Initialize and run
vector_stores = load_vector_stores()
print("Welcome to the Password Security Assistant! Type 'exit' to quit.")
print("Ask me about password strength, security rules, common passwords, or breach data.")
while True:
    question = input("\nAsk a password security question: ")
    if question.lower() == 'exit':
        print("Goodbye!")
        break
    response = get_rag_response(question, vector_stores)
    #print("\nAnswer:", response)