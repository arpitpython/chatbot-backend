from rest_framework import serializers
from .models import *


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedDocument
        fields = '__all__'


class DocumentPermissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentPermission
        fields = '__all__'


class DocumentGroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentGroup
        fields = '__all__'


class LAChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = LAChatMessage
        fields = '__all__'
        read_only_fields = ('user', 'bot_timestamp')