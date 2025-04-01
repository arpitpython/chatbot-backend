# PyPDF for pdf reader
import fitz 
import pytesseract
import asyncio
import io
import os
import platform
from PIL import Image
from mimetypes import guess_type
from docx import Document as wordDocument
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Faiss
import faiss
import pickle
import numpy as np

from .models import *
from .serializers import *
from .bot_utilities import LanguageProcessor
from azure_blob_manager import AzureBlobManager


if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = os.path.join(r'C:\Users\arpit.patel\AppData\Local\Programs\Tesseract-OCR', 'tesseract.exe')
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    

def document_list():
    documents = UploadedDocument.objects.filter(is_deleted=False, bot_type="document")
    document_serializer = DocumentSerializer(documents, many=True)
    document_data = document_serializer.data
    
    for doc in document_data:
        if 'id' not in doc:
            doc['id'] = doc.get('pk', None)
        
        if 'name' not in doc and 'file_name' in doc:
            doc['name'] = doc['file_name']
        
        if 'type' not in doc and 'file_type' in doc:
            doc['type'] = doc['file_type']
            
        if 'size' not in doc and 'file_size' in doc:
            doc['size'] = doc['file_size']
            
        if 'azure_blob_path' in doc:
            azure_path = doc['azure_blob_path']
            file_url = AzureBlobManager(azure_path)
            azure_blob_url = file_url.generate_blob_sas_url()
            doc['url'] = azure_blob_url

    return document_data


async def read_document(azure_file_path, bot_type=None):
    azure_document = AzureBlobManager(azure_file_path)
    document_content = azure_document.download_blob(azure_file_path)
    mime_type, _ = guess_type(azure_file_path.lower())
    document_data = None
    
    if mime_type == 'text/plain':
        document_data = document_content.decode("utf-8")    
    elif mime_type == 'application/pdf':
        document_data = extract_text_from_pdf(document_content=document_content)
    elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                       "application/msword"]:
        document_data = extract_text_from_docx(document_content)
        
    if document_data and bot_type == 'document':
        pass
    else:
        return document_data    
    

async def save_document_data_faiss(document_data):
    documents = []
    document = Document(page_content=document_data)
    documents.append(document)
    
    docs_processed = split_documents(documents)
    await store_embeddings_in_vector_db(docs_processed)
    return True
    

def split_documents(documents, chunk_size=2000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
    )

    return text_splitter.split_documents(documents)


def extract_text_from_docx(docx_bytes):
    """Extract text from a Word (DOCX) file."""
    doc_stream = io.BytesIO(docx_bytes)
    doc = wordDocument(doc_stream)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(file_path=None, document_content=None):
    if not file_path and not document_content:
        return False
    
    """Extract text from a normal or scanned PDF."""
    if file_path:
        doc = fitz.open(file_path)
    else:
        doc = fitz.open(stream=document_content, filetype="pdf")

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


def is_scanned_pdf(file_path):
    """Check if a PDF is scanned or contains selectable text."""
    doc = fitz.open(file_path)
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            return False
    return True


async def store_embeddings_in_vector_db(documents):
    """Store document embeddings in FAISS using Azure OpenAI SDK."""
    llm = LanguageProcessor()
    tasks = [llm.generate_embeddings(doc.page_content) for doc in documents]
    embeddings = await asyncio.gather(*tasks)

    embeddings = [emb for emb in embeddings if emb is not None]
    if not embeddings:
        print("No valid embeddings generated.")
        return

    np_embeddings = np.array(embeddings, dtype=np.float32)

    # Store in FAISS
    faiss_index = faiss.IndexFlatL2(np_embeddings.shape[1])
    faiss_index.add(np_embeddings)
    save_faiss_index(faiss_index)

    # Save document texts for retrieval
    with open("document_texts.pkl", "wb") as f:
        pickle.dump([doc.page_content for doc in documents], f)

    print("Stored embeddings in FAISS and saved document texts.")
    return True


def load_faiss_index(input_file):
    faiss_file = f"/faiss_documents/{input_file}"
    if os.path.exists(faiss_file):
        with open(faiss_file, "rb") as f:
            return pickle.load(f)
    print("FAISS index not found. Ensure embeddings are stored first.")
    return None


def save_faiss_index(index, input_file):
    faiss_file = f"/faiss_documents/{input_file}"
    with open(faiss_file, "wb") as f:
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


