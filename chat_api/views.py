from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
from .serializers import ChatRequestSerializer, ChatResponseSerializer
from .models import ChatMessage
from .rag_service import rag_service

@api_view(['POST'])
def chat_message(request):
    """
    Endpoint para procesar mensajes del chat
    """
    serializer = ChatRequestSerializer(data=request.data)
    
    if serializer.is_valid():
        question = serializer.validated_data['question']
        
        try:
            # Obtener respuesta del sistema RAG
            answer = rag_service.get_answer(question)
            
            # Guardar en base de datos
            chat_message = ChatMessage.objects.create(
                question=question,
                answer=answer
            )
            
            # Preparar respuesta
            response_data = {
                'answer': answer,
                'timestamp': timezone.now()
            }
            
            response_serializer = ChatResponseSerializer(response_data)
            return Response(response_serializer.data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {'error': 'Error interno del servidor'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def chat_history(request):
    """
    Endpoint para obtener historial de chat
    """
    messages = ChatMessage.objects.all()[:50]  # Últimos 50 mensajes
    data = []
    
    for msg in messages:
        data.append({
            'question': msg.question,
            'answer': msg.answer,
            'timestamp': msg.timestamp
        })
    
    return Response(data, status=status.HTTP_200_OK)