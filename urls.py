from django.urls import path
from . import views

urlpatterns = [
    # View to serve the main application
    path('', views.load_document_bot, name='load_document_bot'),
    
    # API endpoint to get CSRF token
    path('api/csrf-token/', views.get_csrf_token, name='get_csrf_token'),
    
    # Document API endpoints
    path('api/load_chat_history/', views.get_documents, name='get_documents'),
    path('api/chat_sessions/', views.get_chat_sessions, name='chat_sessions'),
    path('api/chat_messages/<bot_type>/<session_id>/', views.get_messages_by_sessionid, name='chat_session_id'),
    path('api/get_documents/', views.get_documents, name='get_documents'),
    path('api/upload_document/', views.upload_document, name='upload_document'),
    path('api/delete_document/<uuid:documentId>/', views.delete_document, name='delete_document'),
    path('api/process_message/', views.process_message, name='process_message'),
    
    # New endpoints for session and message deletion
    path('api/delete_session/<bot_type>/<session_id>/', views.delete_session, name='delete_session'),
    path('api/delete_message/<bot_type>/<message_id>/', views.delete_message, name='delete_message'),
]


