from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz
import pytesseract
from PIL import Image

import faiss
import numpy as np
import tiktoken
import asyncio
import os
import pickle
import time
import platform
import io

from azure_blob_manager import AzureBlobManager
from openai import AzureOpenAI, OpenAIError


class DocumentProcessor:
    def __init__(self, openai_client=None):
        """Initialize the document processor with optional OpenAI client."""
        self.openai_client = openai_client or AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        if platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = os.path.join(r'C:\Users\arpit.patel\AppData\Local\Programs\Tesseract-OCR', 'tesseract.exe')
        else:
            pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        
        self.chunk_size = 2000
        self.chunk_overlap = 200
        
    def count_tokens(self, text, model="gpt-4o"):
        """Estimate token count for a given text using tiktoken."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    def is_scanned_pdf(self, file_data):
        """Check if a PDF is scanned or contains selectable text."""
        doc = fitz.open(stream=file_data, filetype="pdf")
        for page in doc:
            text = page.get_text("text").strip()
            if text:
                return False
        return True
    
    def extract_text_from_pdf(self, file_data):
        """Extract text from a normal or scanned PDF."""
        doc = fitz.open(stream=file_data, filetype="pdf")
        text = []
        
        is_scanned = True
        for page in doc:
            page_text = page.get_text("text").strip()
            if page_text:
                is_scanned = False
                break
                
        if is_scanned:
            print("⚠️ Scanned PDF detected. Extracting using OCR...")
            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                text.append(page_text)
        else:
            for page in doc:
                text.append(page.get_text("text"))
        
        return "\n".join(text).strip()
    
    def split_documents(self, document_text, source_name):
        """Split a document into chunks."""
        document = Document(page_content=document_text, metadata={"source": source_name})
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            strip_whitespace=True
        )
        
        return text_splitter.split_documents([document])
    
    def generate_embeddings(self, text_chunks):
        """Generate embeddings for text chunks using Azure OpenAI."""
        embeddings = []
        
        for chunk in text_chunks:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[chunk.page_content]
                )
                embeddings.append(response.data[0].embedding)
            except OpenAIError as e:
                print(f"Embedding API Error for chunk: {e}")
        
        return embeddings
    
    def create_faiss_index(self, embeddings):
        """Create a FAISS index from the embeddings."""
        if not embeddings:
            return None
            
        np_embeddings = np.array(embeddings, dtype=np.float32)
        faiss_index = faiss.IndexFlatL2(np_embeddings.shape[1])
        faiss_index.add(np_embeddings)
        return faiss_index
    
    def process_document(self, file_data, file_name):
        """
        Process a document, split it into chunks, generate embeddings,
        and create a FAISS index.
        """
        start_time = time.time()
        print(f"Processing document: {file_name}")
        
        if file_name.lower().endswith('.pdf'):
            document_text = self.extract_text_from_pdf(file_data)
        else:
            document_text = file_data.decode('utf-8', errors='ignore')
        
        chunks = self.split_documents(document_text, file_name)
        print(f"Document split into {len(chunks)} chunks")
        
        embeddings = self.generate_embeddings(chunks)
        print(f"Generated {len(embeddings)} embeddings")
        
        faiss_index = self.create_faiss_index(embeddings)
        
        processing_time = time.time() - start_time
        print(f"Document processing completed in {processing_time:.2f} seconds")
        
        return {
            'chunks': chunks,
            'faiss_index': faiss_index,
            'processing_time': processing_time
        }
    
    def serialize_processing_results(self, results):
        """Serialize the processing results for storage."""
        chunks = results['chunks']
        faiss_index = results['faiss_index']
        
        chunks_data = [
            {
                'content': chunk.page_content,
                'metadata': chunk.metadata
            } for chunk in chunks
        ]
        
        faiss_index_data = pickle.dumps(faiss_index)
        
        return {
            'chunks_data': pickle.dumps(chunks_data),
            'faiss_index_data': faiss_index_data
        }
    
    def save_to_azure_blob(self, document_id, serialized_data):
        """Save the processed document data to Azure Blob Storage."""
        chunks_blob_path = f'docbot_processed/{document_id}_chunks.pkl'
        index_blob_path = f'docbot_processed/{document_id}_index.pkl'
        
        chunks_blob = AzureBlobManager(chunks_blob_path)
        chunks_blob.upload_to_blob(serialized_data['chunks_data'])
        
        index_blob = AzureBlobManager(index_blob_path)
        index_blob.upload_to_blob(serialized_data['faiss_index_data'])
        
        return {
            'chunks_path': chunks_blob_path,
            'index_path': index_blob_path
        }
    
    def load_from_azure_blob(self, document_id):
        """Load processed document data from Azure Blob Storage."""
        chunks_blob_path = f'docbot_processed/{document_id}_chunks.pkl'
        index_blob_path = f'docbot_processed/{document_id}_index.pkl'
        
        try:
            chunks_blob = AzureBlobManager(chunks_blob_path)
            chunks_data = chunks_blob.download_from_blob()
            chunks = pickle.loads(chunks_data)
            
            index_blob = AzureBlobManager(index_blob_path)
            index_data = index_blob.download_from_blob()
            faiss_index = pickle.loads(index_data)
            
            return {
                'chunks': chunks,
                'faiss_index': faiss_index
            }
        except Exception as e:
            print(f"Error loading processed data from Azure Blob: {e}")
            return None
    
    def query_document(self, document_id, query_text, top_k=5):
        """
        Query a processed document using the stored FAISS index.
        """
        processed_data = self.load_from_azure_blob(document_id)
        if not processed_data:
            return {"error": "Failed to load processed document data"}
        
        try:
            query_response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query_text]
            )
            query_embedding = query_response.data[0].embedding
        except OpenAIError as e:
            return {"error": f"Failed to generate query embedding: {str(e)}"}
        
        faiss_index = processed_data['faiss_index']
        chunks = processed_data['chunks']
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = faiss_index.search(query_vector, top_k)

        matched_chunks = [chunks[i]['content'] for i in indices[0] if i < len(chunks)]
        
        return {
            "matched_chunks": matched_chunks,
            "distances": distances[0].tolist()
        }


if __name__ == "__main__":
    processor = DocumentProcessor()
    