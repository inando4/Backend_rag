from django.db import models
from django.utils import timezone

class ChatMessage(models.Model):
    question = models.TextField()
    answer = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Q: {self.question[:50]}..."
