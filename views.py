# Django
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.db.models import Max, Min, Count
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect
from django.middleware.csrf import get_token
from django.utils.text import Truncator
from django.utils import timezone
from django.db import transaction
from django.core.exceptions import ValidationError

# rest framework
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view

# other 
from dotenv import load_dotenv, find_dotenv
from uuid import UUID
from datetime import datetime
import time
import pytz
import json
import urllib.parse
import base64
import mimetypes

import logging
logger = logging.getLogger(__name__)

# same project files
from .serializers import LAChatMessageSerializer
from azure_blob_manager import AzureBlobManager
from .bot_utilities import *
from .models import *
from .serializers import *
from .utilities import *
from .document_processor import *


load_dotenv(find_dotenv())


@ensure_csrf_cookie
def load_document_bot(request):
    template = 'index.html'
    documents = UploadedDocument.objects.all()
    document_serializer = DocumentSerializer(documents, many=True)
    document_data = document_serializer.data
    context = {
        'documents': document_data
    }
    return render(request, template_name=template, context=context)


@ensure_csrf_cookie
def get_csrf_token(request):
    """
    Endpoint that sets and returns the CSRF token.
    This should be called before making any POST requests.
    """
    token = get_token(request)
    response = JsonResponse({"csrfToken": token})
    
    response.set_cookie(
        'csrftoken',
        token,
        max_age=86400,
        path='/',
        secure=False,
        httponly=False,
        samesite='Lax'
    )
    return response


@api_view(['GET'])
def get_documents(request):
    """
    API endpoint to get all documents
    """

    document_data = document_list()
    return JsonResponse({
        "success": True,
        "documents": document_data,
    })


@api_view(['POST'])
def upload_document(request):
    """
    API endpoint to upload a document - enhanced with preprocessing
    """
    try:
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return JsonResponse({
                'success': False,
                'message': 'No file uploaded.'
            }, status=400)
        
        bot_type = request.data.get('task_type')
        file_name = uploaded_file.name
        file_type = uploaded_file.content_type
        file_size = uploaded_file.size
        file_data = uploaded_file.read()
        
        azure_blob_path = f'docbot_files/{uploaded_file.name}'            
        azure_blob = AzureBlobManager(azure_blob_path)
        azure_blob.upload_to_blob(file_data)
        file_url = azure_blob.generate_blob_sas_url()
        
        # 2. Create database record
        document = UploadedDocument.objects.create(
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_url=azure_blob.blob.primary_endpoint,
            azure_blob_path=azure_blob_path,
            bot_type=bot_type,
            uploaded_by='arpit.patel@analytix.com'
        )
        
        try:
            processor = DocumentProcessor()
            processing_results = processor.process_document(file_data, file_name)
            serialized_data = processor.serialize_processing_results(processing_results)
            processor.save_to_azure_blob(str(document.id), serialized_data)
            
            processing_status = "success"
            processing_message = f"Document processed successfully in {processing_results['processing_time']:.2f} seconds"
        except Exception as processing_error:
            processing_status = "error"
            processing_message = f"Document uploaded but preprocessing failed: {str(processing_error)}"
            logger.error(f"Document processing error: {str(processing_error)}")
        
        return JsonResponse({
            'success': True,
            'message': 'Document uploaded successfully.',
            'id': document.id,
            'name': document.file_name,
            'type': document.file_type,
            'size': document.file_size,
            'created_at': document.created_at,
            'url': file_url,
            'processing': {
                'status': processing_status,
                'message': processing_message
            }
        }, status=201)
        
    except Exception as e:
        logger.error(f"Error in upload_document: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error uploading document: {str(e)}'
        }, status=500)
        

@api_view(['DELETE'])
def delete_document(request, documentId):
    document = UploadedDocument.objects.get(id=documentId)
    document.is_deleted = True
    document.save()
    
    LAChatMessage.objects.filter(document=document).update(delete_status=True) 
    
    azure_document = AzureBlobManager(document.azure_blob_path)
    azure_document.delete_blob()

    return JsonResponse({
        'success': True,
        'message': 'Document deleted successfully.',
    }, status=200)
    


@api_view(['POST'])    
def process_message(request):
    """
    Process message with optimized document handling
    """
    process_start_time = time.time()
    
    try:
        # Verify session is still valid
        session_status = verify_session_timeout(request)
        if not session_status:
            return Response({
                'success': False,
                'message': 'Your session has timed out. Please login again to continue.',
                'error': {
                    'summary': 'Session expired',
                    'details': 'Your session has timed out. Please login again to continue.',
                    'code': 'AUTH_EXPIRED',
                    'timestamp': datetime.now().isoformat()
                }
            }, status=status.HTTP_401_UNAUTHORIZED)
            
        indian_tz = pytz.timezone('Asia/Kolkata')
        user_timestamp = timezone.now().astimezone(indian_tz)
        
        if request.method == 'POST':
            try:
                data = request.data or request.body.decode('utf-8')
                bot_type = data.get('botType', "")
                input_text = data.get('message', "")
                user_input_text = data.get('message', "")
                document_id = data.get('documentId', None)
                session_id = data.get('sessionId', None)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"Failed to parse request body: {str(e)}")
                return Response({
                    'success': False,
                    'message': 'Invalid request format',
                    'error': {
                        'summary': 'Invalid request format',
                        'details': str(e),
                        'code': 'VALIDATION_ERROR'
                    }
                }, status=status.HTTP_400_BAD_REQUEST)
        
        document = None
        document_context = ""
        
        if document_id:
            try:
                document = UploadedDocument.objects.get(id=document_id)
                user_message = f'{user_input_text}\n\n{document.file_name}'
                
                if bot_type == 'document':
                    processor = DocumentProcessor()
                    query_results = processor.query_document(str(document_id), input_text)
                    
                    if "error" in query_results:
                        logger.error(f"Document query error: {query_results['error']}")
                        return Response({
                            'success': False,
                            'error': f"Document Processing Error: {query_results['error']}"
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
                    document_context = "\n\n".join(query_results["matched_chunks"])

                    input_text = f"""
                    User question: {input_text}
                    
                    Relevant document content:
                    {document_context}
                    
                    Based on the document content above, please answer the user's question.
                    """
                else:
                    document_azure_path = document.azure_blob_path
                    document_content = read_document(document_azure_path)
                    if document_content:
                        input_text += document_content
            except Exception as e:
                logger.error(f"Document processing error: {str(e)}")
                return Response({
                    'success': False,
                    'error': f"Document Processing Error: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            user_message = input_text
                
        if not bot_type or not input_text:
            missing_fields = {}
            if not bot_type:
                missing_fields['bot_type'] = 'This field is required'
            if not input_text:
                missing_fields['input_text'] = 'This field is required'
                                
            logger.error(f"Missing required fields: {missing_fields}")
            return Response({
                'success': False,
                'message': 'Missing required fields',
                'error': {
                    'summary': 'Missing required fields',
                    'details': missing_fields,
                    'code': 'VALIDATION_ERROR'
                }
            }, status=status.HTTP_400_BAD_REQUEST)

        input_count_token = count_tokens(input_text)
        MAX_TOKEN_LIMIT = 100000
        if input_count_token > MAX_TOKEN_LIMIT:
            return Response({
                'success': False,
                'error': f'Token limit exceeded! The maximum allowed limit is {MAX_TOKEN_LIMIT:,} tokens, '
                        f'but you have input {input_count_token:,} tokens.'
            }, status=status.HTTP_400_BAD_REQUEST)
         
        valid_bot_types = ["grammar", "email", "meeting_insights", "document"]
        if bot_type not in valid_bot_types:
            logger.error(f"Invalid task type: {bot_type}")
            return Response({
                'success': False,
                'message': f'Invalid Task Type - {bot_type}, It should be one of the {valid_bot_types}',  
                'error': {
                    'summary': 'Invalid TaskType',
                    'message': f'Invalid Task Type - {bot_type}, It should be one of the {valid_bot_types}',  
                    'code': 'VALIDATION_ERROR'
                }
            }, status=status.HTTP_400_BAD_REQUEST)

        # Get user information
        login_user_email = 'arpit.patel'
        login_token = 'arpit#4444'        
        if not login_user_email or not login_token:
            return Response({
                'success': False,
                'message': 'Login Email or Login Token Missing',
                'error': {
                    'summary': 'Authentication required',
                    'details': 'Login Email or Login Token Missing',
                    'code': 'AUTH_ERROR'
                }
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Create initial chat message
        with transaction.atomic():
            chat_message = LAChatMessage.objects.create(
                user_email=login_user_email,
                session_id=session_id,
                bot_type=bot_type,
                delete_status=False,
                user_message=user_message,
                document=document if document else None,
                user_timestamp=user_timestamp,
                # Initialize with safe default values to prevent null constraint errors
                bot_response="",
                output_tokens=0,
                input_tokens=0,
                cached_content_token_count=0,
                total_cost=0,
                processing_error=""
            )
            
            chat_id = chat_message.id

        # Process the message with LLM
        answer = ""
        tokens = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_content_token_count": 0,
            "total_cost": 0
        }
        llm_response_time = timedelta(seconds=0)
        error = None
        
        try:
            llm_start_time = time.time()
            llm_response = process_text_by_task(bot_type, input_text, login_token)
            
            answer = llm_response.get("answer", "")
            tokens = llm_response.get("usage", {})
            error = llm_response.get("error")
            
            # Handle potential missing or invalid data
            if not isinstance(tokens, dict):
                tokens = {
                    "input_tokens": 0,
                    "output_tokens": 0, 
                    "cached_content_token_count": 0,
                    "total_cost": 0
                }
            
            # Calculate LLM response time
            llm_execution_time = tokens.get('llm_response_time', time.time() - llm_start_time)
            llm_response_time = timedelta(seconds=int(llm_execution_time))
            
            # Handle LLM errors
            if error:
                logger.error(f"LLM Error in process_message: {error}")
                with transaction.atomic():
                    chat = LAChatMessage.objects.get(id=chat_id)
                    chat.query_response_status = False
                    chat.processing_error = str(error)
                    chat.llm_response_time = llm_response_time
                    chat.save()
                
                return Response({
                    'success': False,
                    'message': f'LLM Error: {error}',
                    'error': {
                        'summary': 'Unexpected response from LLM',
                        'details': f'LLM Error: {error}',
                        'code': 'LLM_ERROR'
                    }
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            error = str(e)
            logger.error(f"Error in process_message LLM call: {error}")
            
            with transaction.atomic():
                chat = LAChatMessage.objects.get(id=chat_id)
                chat.query_response_status = False
                chat.processing_error = error
                chat.llm_response_time = llm_response_time
                chat.save()
                
            return Response({
                'success': False,
                'message': f'Processing Error: {error}',
                'error': {
                    'summary': 'Error processing request',
                    'details': error,
                    'code': 'PROCESSING_ERROR'
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Update chat message with bot response
        bot_timestamp = timezone.now().astimezone(indian_tz)
        chat_title = Truncator(input_text).chars(90)
        
        # Ensure all values are valid to prevent null constraint errors
        safe_tokens = {
            "output_tokens": int(tokens.get("output_tokens", 0) or 0),
            "input_tokens": int(tokens.get("input_tokens", 0) or 0),
            "cached_content_token_count": int(tokens.get("cached_content_token_count", 0) or 0),
            "total_cost": float(tokens.get("total_cost", 0) or 0)
        }
        
        try:
            with transaction.atomic():
                chat = LAChatMessage.objects.get(id=chat_id)
                chat.bot_response = str(answer) if answer else ''
                chat.chat_title = chat_title
                chat.bot_timestamp = bot_timestamp
                chat.output_tokens = safe_tokens["output_tokens"]
                chat.input_tokens = safe_tokens["input_tokens"]
                chat.cached_content_token_count = safe_tokens["cached_content_token_count"]
                chat.total_cost = safe_tokens["total_cost"]
                chat.query_response_status = True
                chat.llm_response_time = llm_response_time
                chat.save()
        
        except ValidationError as ve:
            logger.error(f"Validation error updating chat: {str(ve)}")
            return Response({
                'success': False,
                'message': f'Validation error: {str(ve)}',
                'error': {
                    'summary': 'Data validation error',
                    'details': str(ve),
                    'code': 'VALIDATION_ERROR'
                }
            }, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            logger.error(f"Error updating chat record: {str(e)}")
            return Response({
                'success': False,
                'message': f'Database error: {str(e)}',
                'error': {
                    'summary': 'Database error',
                    'details': str(e),
                    'code': 'DB_ERROR'
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get updated chat history
        user_chat_data = filter_chat_data(request, bot_type)
        if user_chat_data is False:
            user_chat_data = []
            
        serializer = LAChatMessageSerializer(user_chat_data, many=True)
        chat_history = sorted(
            serializer.data, 
            key=lambda x: datetime.fromisoformat(x['created_at'].replace('Z', '+00:00')), 
            reverse=True
        ) if serializer.data else []
        
        execution_time = (time.time() - process_start_time)
        query_processing_time = timedelta(seconds=int(execution_time))
        
        # Update processing time
        try:
            LAChatMessage.objects.filter(id=chat_id).update(query_processing_time=query_processing_time)
        except Exception as e:
            logger.warning(f"Failed to update processing time: {str(e)}")
        
        return Response({
            "success": True,
            "chat_history": chat_history,
            "bot_response": chat.bot_response or error or "",
            "processing_time": execution_time
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"Unhandled error in process_message: {str(e)}")
        return Response({
            'success': False,
            'message': 'An unexpected error occurred',
            'error': {
                'summary': 'Server error',
                'details': str(e),
                'code': 'SERVER_ERROR'
            }
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
def filter_chat_data(request, bot_type, history_setting=True):   
    if request.user.is_authenticated:
        login_user_email = request.user.email
        login_token = request.session.session_key
        request.session['user_data'] = {
            'email': login_user_email,
            'login_key': login_token
        }
        request.session.modified = True
    else:
        login_user_email = "arpit.patel"
        login_token = "arpit#4444"
        
        if not login_user_email or not login_token:
            return False
           
    user_chat_history = LAChatMessage.objects.filter(
        user_email=login_user_email, 
        bot_type=bot_type, 
        delete_status=False,
        query_response_status=True
    ).select_related()
    
    if not history_setting:
        user_chat_history = user_chat_history.filter(session_id=login_token)
                
    return user_chat_history


def get_chat_sessions(request):
    """
    Get all chat sessions for a specific bot type (bot_type)
    
    Query parameters:
    - bot_type: The type of bot (e.g., 'grammar', 'email', 'meeting_insights', 'document')
    
    Returns:
    - JSON response with session data
    """
    bot_type = request.GET.get('bot_type')
    
    if not bot_type:
        return JsonResponse({
            'success': False,
            'error': 'Missing bot_type parameter'
        }, status=400)
    
    # Group by session_id to get unique sessions
    sessions = LAChatMessage.objects.filter(
        bot_type=bot_type,
        delete_status=False
    ).values('session_id').annotate(
        created_at=Min('created_at'),
        last_activity=Max('updated_at'),
        message_count=Count('id'),
        title=Max('chat_title')
    )
    
    # For each session, get the first user message as preview
    session_data = []
    for session in sessions:
        # Get the first user message for preview
        first_message = LAChatMessage.objects.filter(
            session_id=session['session_id'],
            delete_status=False
        ).order_by('created_at').first()
        
        preview = None
        if first_message:
            preview = first_message.user_message[:50] + ('...' if len(first_message.user_message) > 50 else '')
        
        session_data.append({
            'id': session['session_id'],
            'title': session['title'] or 'Untitled Conversation',
            'created_at': session['created_at'],
            'last_activity': session['last_activity'],
            'message_count': session['message_count'],
            'preview': preview
        })
    
    return JsonResponse({
        'success': True,
        'sessions': session_data
    })


def get_messages_by_sessionid(request, bot_type, session_id):
    """
    Get all messages for a specific session
    
    URL parameters:
    - session_id: The ID of the chat session
    
    Returns:
    - JSON response with messages data
    """
    if not session_id or not bot_type:
        return JsonResponse({
            'success': False,
            'error': 'Missing required parameter : Bot Type or Session ID'
        }, status=400)
    
    try:
        decoded_session_id = urllib.parse.unquote(session_id)
    except:
        decoded_session_id = session_id    
    
    # Get all messages for this session
    messages_query = LAChatMessage.objects.filter(
        session_id=decoded_session_id,
        bot_type=bot_type,
        delete_status=False
    ).order_by('created_at')
    
    # If no messages found
    # if not messages_query.exists():
    #     return JsonResponse({
    #         'success': False,
    #         'error': 'Session not found or has no messages'
    #     }, status=404)
    
    # Format messages for the frontend
    messages = []
    for msg in messages_query:
        messages.append({
            'id': str(msg.id) + '_user',
            'content': msg.user_message,
            'sender': 'user',
            'timestamp': msg.user_timestamp,
            'sessionId': msg.session_id
        })
        
        # Add bot response
        messages.append({
            'id': str(msg.id) + '_bot',
            'content': msg.bot_response or f'Processing Error : {msg.processing_error}',
            'sender': 'bot',
            'timestamp': msg.bot_timestamp,
            'sessionId': msg.session_id
        })
    
    # Get task type (bot type) from the first message
    bot_type = messages_query.first().bot_type if messages_query.exists() else None
    
    return JsonResponse({
        'success': True,
        'messages': messages,
        'bot_type': bot_type,
        'session_id': session_id
    })


@api_view(['DELETE'])
def delete_session(request, bot_type, session_id):
    """
    Delete a chat session and all its messages
    """
    try:
        decoded_session_id = urllib.parse.unquote(session_id)
        
        messages = LAChatMessage.objects.filter(
            session_id=decoded_session_id,
            bot_type=bot_type
        )
        
        if not messages.exists():
            return JsonResponse({
                'success': True,
                'message': 'Session not found or already deleted.',
                'was_empty': True
            })
        
        # Soft delete by setting delete_status to True
        message_count = messages.update(delete_status=True)
        
        return JsonResponse({
            'success': True,
            'message': f'Session deleted successfully. {message_count} messages affected.',
        })        
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete session: {str(e)}'
        }, status=500)
        
        

@api_view(['DELETE'])
def delete_message(request, bot_type, message_id):
    """
    Delete a specific message (both user question and bot response)
    
    URL parameters:
    - bot_type: Type of bot
    - message_id: ID of the message to delete
    
    Returns:
    - JSON response indicating success or failure
    """
    try:
        # Find the message by its ID
        try:
            # Remove the _user or _bot suffix if present
            original_id = message_id.split('_')[0]
            message = LAChatMessage.objects.get(
                id=original_id,
                bot_type=bot_type
            )
        except LAChatMessage.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Message not found'
            }, status=404)
        
        # Soft delete by setting delete_status to True
        message.delete_status = True
        message.save()
        
        return JsonResponse({
            'success': True,
            'message': 'Message deleted successfully',
        })
        
    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to delete message: {str(e)}'
        }, status=500)


