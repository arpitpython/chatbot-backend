from django.utils import timezone
from django.db import models
from django.contrib.auth.models import User
import uuid


class DocumentGroup(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, default="personal")
    is_public = models.BooleanField(default=False)


# Create your models here.
class UploadedDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file_name = models.CharField(max_length=255)
    file_type = models.CharField(max_length=100)
    file_size = models.IntegerField()  # Size in bytes
    # file_content = models.TextField()
    file_url = models.URLField()  # URL to Azure Blob Storage
    azure_blob_path = models.CharField(max_length=500)  # Path within Azure Blob Storage
    # uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_documents')
    uploaded_by = models.EmailField(max_length=255, blank=True)
    bot_type = models.CharField(max_length=100, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False) 
    is_public = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    
class DocumentPermission(models.Model):
    document = models.ForeignKey('UploadedDocument', on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)    
    

class LAChatMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_email = models.CharField(max_length=124)
    document = models.ForeignKey('UploadedDocument', on_delete=models.DO_NOTHING, null=True, default=None)
    session_id = models.CharField(max_length=100)
    chat_id = models.CharField(max_length=100)
    bot_type = models.CharField(max_length=50)
    chat_title = models.CharField(max_length=100, null=True)
    delete_status = models.BooleanField(default=False)
    user_message = models.TextField()
    user_timestamp = models.DateTimeField(default=timezone.now)
    bot_response = models.TextField()
    bot_timestamp = models.DateTimeField(default=timezone.now)
    output_tokens = models.IntegerField(default=0)
    input_tokens = models.IntegerField(default=0)
    cached_content_token_count = models.IntegerField(default=0)
    total_cost = models.DecimalField(max_digits=6, decimal_places=3, default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    query_response_status = models.BooleanField(default=True)
    llm_response_time = models.DurationField(null=True, blank=True)
    query_processing_time = models.DurationField(null=True, blank=True)
    processing_error = models.TextField()

    def __str__(self):
        return f"Chat with {self.user_email} at {self.user_timestamp}"    
    
    
    