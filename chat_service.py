# chat_service.py

import asyncio
from .models import UploadedDocument
from .chat_models import ChatSession, ChatMessage
from .document_processor import process_document, query_document, generate_answer
import os

class DocumentChatService:
    """Service to handle document chat functionality"""
    
    @staticmethod
    async def process_document_if_needed(document_id):
        """Process document if it hasn't been processed yet"""
        # Check if FAISS index exists for this document
        from .document_processor import get_faiss_index_path
        
        faiss_index_path = get_faiss_index_path(document_id)
        if os.path.exists(faiss_index_path):
            # Document already processed
            return True
        
        try:
            # Get document from database
            document = UploadedDocument.objects.get(id=document_id, is_deleted=False)
            
            # Use the content stored in the database
            document_text = document.file_content
            
            # Process document and store embeddings
            success = await process_document(document_id, document_text)
            return success
            
        except UploadedDocument.DoesNotExist:
            print(f"Document with ID {document_id} not found or is deleted.")
            return False
        except Exception as e:
            print(f"Error processing document: {e}")
            return False
    
    @staticmethod
    async def get_or_create_chat_session(document_id, user_email):
        """Get existing chat session or create a new one"""
        try:
            # Try to get an existing active session
            document = UploadedDocument.objects.get(id=document_id, is_deleted=False)
            session = ChatSession.objects.filter(
                document=document,
                user_email=user_email,
                is_active=True
            ).first()
            
            if not session:
                # Create a new session
                session = ChatSession.objects.create(
                    document=document,
                    user_email=user_email,
                    session_name=f"Chat with {document.file_name}"
                )
                
            return session
            
        except UploadedDocument.DoesNotExist:
            print(f"Document with ID {document_id} not found or is deleted.")
            return None
        except Exception as e:
            print(f"Error creating chat session: {e}")
            return None
    
    @staticmethod
    async def handle_chat_message(session_id, message_text, user_email):
        """Handle a new chat message from the user"""
        try:
            # Get the chat session
            session = ChatSession.objects.get(id=session_id, user_email=user_email)
            document_id = session.document.id
            
            # Save user message
            user_message = ChatMessage.objects.create(
                session=session,
                role='user',
                content=message_text
            )
            
            # Update session timestamp
            session.save()  # This will update the updated_at field
            
            # Make sure document is processed
            is_processed = await DocumentChatService.process_document_if_needed(document_id)
            if not is_processed:
                error_response = "I'm sorry, but I couldn't process the document. Please try again later."
                ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=error_response
                )
                return error_response
            
            # Query document for relevant chunks
            context_chunks = await query_document(document_id, message_text, top_k=5)
            
            # Generate answer
            answer = await generate_answer(message_text, context_chunks)
            
            # Save assistant message
            assistant_message = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=answer
            )
            
            return answer
            
        except ChatSession.DoesNotExist:
            print(f"Chat session with ID {session_id} not found.")
            return "Chat session not found. Please start a new session."
        except Exception as e:
            print(f"Error handling chat message: {e}")
            return "I'm sorry, but I encountered an error processing your message."
        
        
        