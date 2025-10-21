from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma



# Load the PDF document
loader = PyPDFLoader(r"C:\Users\clark\Desktop\UTK\CS 489\LLRPG\data\Player's Handbook.pdf")
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
print("Text split")

# Create embeddings and vector store
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
print("Vector store done")

# Load the chat model
model = ChatOllama(model="llama3.2")

# Define the prompt template
prompt_template = PromptTemplate.from_template(
    "You are an expert in D&D 5e. Use the following context to answer the question. "
    "If the context does not provide enough information, say 'I don't know.'\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

output_parser = StrOutputParser()

def rag_chain(question):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate response using the model
    prompt = prompt_template.format(context=context, question=question)
    response = model.invoke(prompt)

    return response.content

question = "Explain the effects of the fireball spell."
response_message = rag_chain(question)
print(response_message)