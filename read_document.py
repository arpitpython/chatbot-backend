# LangChain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Azure OpenAI
from openai import AzureOpenAI, AsyncAzureOpenAI
import tiktoken
import aiohttp
import asyncio

# PyPDF for pdf reader
import fitz 
import pytesseract
from PIL import Image

# Chroma DB
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# Faiss
import faiss
import pickle
import numpy as np

# Other
import os
import json
import time
import platform
from dotenv import load_dotenv
load_dotenv()


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

# TODO : refer updates
CHROMA_PATH = "./chroma_db"
FAISS_INDEX_PATH = "./faiss_documents/faiss_index.pkl"


if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = os.path.join(r'C:\Users\arpit.patel\AppData\Local\Programs\Tesseract-OCR', 'tesseract.exe')
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


def count_tokens(text, model="gpt-4o"):
    """Estimate token count for a given text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def load_chroma_store():
    chroma_store = Chroma(
            collection_name="documents", 
            persist_directory=CHROMA_PATH,
            embedding_function=OpenAIEmbeddings()
        )
    return chroma_store


def search_chroma(query_text, top_k=5):
    chroma_store = load_chroma_store()
    results = chroma_store.similarity_search(query_text, k=top_k)
    return results


def load_faiss_index(path=FAISS_INDEX_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("FAISS index not found. Ensure embeddings are stored first.")
    return None


def save_faiss_index(index, path=FAISS_INDEX_PATH):
    with open(path, "wb") as f:
        pickle.dump(index, f)


def search_faiss(query_embedding, top_k=5):
    """Search FAISS and return matching document content."""
    faiss_index = load_faiss_index()
    if faiss_index is None:
        print("FAISS index not found.")
        return []

    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)

    if not os.path.exists("./faiss_documents/document_texts.pkl"):
        print("Stored document texts not found.")
        return []

    with open("document_texts.pkl", "rb") as f:
        stored_documents = pickle.load(f)

    return [stored_documents[i] for i in indices[0] if i < len(stored_documents)]


def is_scanned_pdf(file_path):
    """Check if a PDF is scanned or contains selectable text."""
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            return False
    return True


def extract_text_from_pdf(file_path):
    """Extract text from a normal or scanned PDF."""
    doc = fitz.open(file_path)
    text = []

    if is_scanned_pdf(file_path):
        print(f"⚠️ Scanned PDF detected: {file_path}. Extracting using OCR...")
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_text = pytesseract.image_to_string(img)
            text.append(page_text)
    else:
        for page in doc:
            text.append(page.get_text("text"))

    return "\n".join(text).strip()


def load_pdf(input_files):
    """Load and process PDF files."""
    if not isinstance(input_files, list):
        input_files = [input_files]

    documents = []
    for file_path in input_files:
        try:
            text_content = extract_text_from_pdf(file_path)
            document = Document(page_content=text_content, metadata={"source": file_path})
            documents.append(document)
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    return documents


def clean_extra_whitespace(text):
    """Remove extra spaces and newlines."""
    return ' '.join(text.split())


def group_broken_paragraphs(text):
    """Fix broken paragraphs."""
    return text.replace("\n", " ").replace("\r", " ")


def split_documents(documents, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )

    return text_splitter.split_documents(documents)


async def generate_embeddings(input_text):
    """Generate embeddings using Azure OpenAI SDK (Faster than REST API)."""
    try:
        tokens = count_tokens(input_text)
        response = client.embeddings.create(
            model="text-embedding-ada-002",  
            input=[input_text]
        )
        return response.data[0].embedding

    except Exception as e:
        print(f"Embedding API Error: {e}")
        return None


async def store_embeddings_in_vector_db(documents, vstore='faiss'):
    """Store document embeddings in FAISS and ChromaDB using Azure OpenAI SDK."""
    tasks = [generate_embeddings(doc.page_content) for doc in documents]
    embeddings = await asyncio.gather(*tasks)

    embeddings = [emb for emb in embeddings if emb is not None]
    if not embeddings:
        print("No valid embeddings generated.")
        return

    np_embeddings = np.array(embeddings, dtype=np.float32)

    # Store in FAISS
    if vstore == 'faiss':
        faiss_index = faiss.IndexFlatL2(np_embeddings.shape[1])
        faiss_index.add(np_embeddings)
        save_faiss_index(faiss_index)

        # Save document texts for retrieval
        with open("document_texts.pkl", "wb") as f:
            pickle.dump([doc.page_content for doc in documents], f)

        print("Stored embeddings in FAISS and saved document texts.")

    # Store in ChromaDB
    elif vstore == 'chroma':
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        chroma_store = load_chroma_store()
        chroma_store.add_texts(texts=texts, metadata=metadata)
        print("Stored embeddings in ChromaDB")

    else:
        print("Unsupported vector store. Choose 'faiss' or 'chroma'.")


async def main(input_files):
    """Load, split, and store document embeddings."""
    documents = load_pdf(input_files)
    if not documents:
        return []
    
    docs_processed = split_documents(documents)
    await store_embeddings_in_vector_db(docs_processed)
    return docs_processed


async def query_faiss(user_query):
    query_embedding = await generate_embeddings(user_query)
    
    if query_embedding is None:
        print("Failed to generate query embedding.")
        return []

    return search_faiss(query_embedding)


async def ask_question(user_query, max_tokens=8000, max_input_tokens=24000):
    relevant_docs = await query_faiss(user_query)

    if not relevant_docs:
        return "No relevant documents found."

    context = "\n".join(relevant_docs)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer the following question concisely with key points:\n{user_query}"
    )
    
    token_count = count_tokens(prompt, model="gpt-4o")
    if token_count > max_input_tokens:
        print(f"⚠️ Input too long ({token_count} tokens). Truncating...")
        truncated_context = "\n".join(relevant_docs[:2])
        prompt = f"Context:\n{truncated_context}\n\nAnswer concisely:\n{user_query}"

    try:
        response = client.chat.completions.create(
            model="GPT-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Azure OpenAI Error: {e}")
        return "Error generating response."


if __name__ == "__main__":
    document_processing_start = time.time()
    
    input_files = ["my_documents/Insight Products - Quarterly Product Review Meeting.docx"]
    asyncio.run(main(input_files))

    document_processing_end = time.time()
    print(f"\n🔹 Document Execution Time: {document_processing_end - document_processing_start:.2f} seconds")
    
    query_start = time.time()
    query_text = "Which document is this? \n"
    response = asyncio.run(ask_question(query_text))

    query_end = time.time()
    
    print("\n🔹 User Query:", query_text)
    print("🔹 Answer from AI:\n", response)
    print(f"\n🔹 Query Execution Time: {query_end - query_start:.2f} seconds")
