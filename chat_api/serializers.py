from rest_framework import serializers
from .models import ChatMessage

class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'question', 'answer', 'timestamp']
        read_only_fields = ['id', 'timestamp']

class ChatRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000)
    
    def validate_question(self, value):
        if not value.strip():
            raise serializers.ValidationError("La pregunta no puede estar vacía")
        return value.strip()

class ChatResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    timestamp = serializers.DateTimeField()