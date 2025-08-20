import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict  # AGREGAR ESTA LÍNEA
from groq import Groq
from django.conf import settings
from django.utils import timezone

class RAGService:
    def __init__(self):
        # Rutas de archivos
        self.base_path = os.path.join(settings.BASE_DIR, 'data')
        self.json_path = os.path.join(self.base_path, 'dataset_rag_matriculas_mejorado.json')
        self.index_path = os.path.join(self.base_path, 'index.faiss')
        
        # Inicializar modelo de embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Inicializar Groq
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Cargar datos y crear índice
        self.documents = self.load_documents()
        self.index = self.load_or_create_index()
    
    def load_documents(self):
        """Cargar documentos desde JSON"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró {self.json_path}")
            return []
    
    def load_or_create_index(self):
        """Cargar índice existente o crear uno nuevo"""
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        else:
            return self.create_index()
    
    def create_index(self):
        """Crear índice FAISS"""
        if not self.documents:
            return None
        
        # Crear embeddings - CORREGIDO: usar 'content' en lugar de 'contenido'
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.model.encode(texts)
        
        # Crear índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        
        # Normalizar embeddings para similitud coseno
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Guardar índice
        os.makedirs(self.base_path, exist_ok=True)
        faiss.write_index(index, self.index_path)
        
        return index
    
    def keyword_search(self, query, documents):
        """Búsqueda por palabras clave"""
        query_words = query.lower().split()
        keyword_scores = defaultdict(float)
        
        for i, doc in enumerate(documents):
            content = doc['content'].lower()
            score = 0
            
            # Buscar coincidencias exactas de frases
            for word in query_words:
                if word in content:
                    score += content.count(word) * 2  # Mayor peso para coincidencias exactas
            
            # Buscar frases completas
            if ' '.join(query_words) in content:
                score += 10  # Peso alto para frases completas
                
            keyword_scores[i] = score
            
        return keyword_scores
    
    def search_documents(self, query, top_k=3):
        """Búsqueda híbrida: semántica + palabras clave"""
        if not self.index or not self.documents:
            return []
        
        # Búsqueda semántica
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k * 2)  # Obtener más candidatos
        
        # Búsqueda por palabras clave
        keyword_scores = self.keyword_search(query, self.documents)
        
        # Combinar puntuaciones
        combined_results = []
        for i, score in enumerate(scores[0]):
            if score > 0.1:  # Umbral más bajo
                doc_idx = indices[0][i]
                semantic_score = float(score)
                keyword_score = keyword_scores.get(doc_idx, 0)
                
                # Combinar puntuaciones (puedes ajustar los pesos)
                combined_score = (semantic_score * 0.6) + (keyword_score * 0.4)
                
                combined_results.append({
                    'documento': self.documents[doc_idx],
                    'score': combined_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score
                })
        
        # Ordenar por puntuación combinada
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filtrar y devolver top_k
        return combined_results[:top_k]
    
    def generate_response(self, query, context_docs):
        """Generar respuesta usando Groq"""
        try:
            # DEBUG: Verificar estructura de datos
            print(f"Estructura de context_docs: {context_docs[0] if context_docs else 'Vacío'}")
            
            # Construir contexto - CORREGIDO: usar 'content' en lugar de 'contenido'
            context = "\n\n".join([doc['documento']['content'] for doc in context_docs])
            
            # Prompt para Groq
            prompt = f"""Eres un asistente virtual de la Universidad Nacional de San Agustín (UNSA). 
            Responde de manera clara y precisa basándote únicamente en el siguiente contexto sobre normativas universitarias.

            CONTEXTO:
            {context}

            PREGUNTA: {query}

            INSTRUCCIONES:
            - Responde solo con información del contexto proporcionado
            - Si no tienes información suficiente, indica que no puedes responder esa consulta específica
            - Mantén un tono profesional y amigable
            - Sé conciso pero completo en tu respuesta

            RESPUESTA:"""
            
            # Llamada a Groq
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error en Groq API: {e}")
            return "Lo siento, no puedo procesar tu consulta en este momento. Por favor, intenta más tarde."
    
    def get_answer(self, question):
        """Método principal para obtener respuesta"""
        # Buscar documentos relevantes
        relevant_docs = self.search_documents(question)
        
        if not relevant_docs:
            return "No encontré información relevante para tu consulta. Por favor, reformula tu pregunta o consulta sobre temas como matrículas, convalidaciones, reservas, o reactualización."
        
        # Generar respuesta
        return self.generate_response(question, relevant_docs)

# Instancia global del servicio
rag_service = RAGService()