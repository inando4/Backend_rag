import json
import os
import faiss
import numpy as np
import re
import unicodedata
import socket
import time
import logging
import requests
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from groq import Groq
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Rutas de archivos
        self.base_path = os.path.join(settings.BASE_DIR, 'data')
        self.json_path = os.path.join(self.base_path, 'dataset_v2.json')
        self.index_path = os.path.join(self.base_path, 'index.faiss')
        
        # Inicializar modelo de embeddings
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Configuración LLM (local o API)
        self.llm_strategy = os.getenv('LLM_STRATEGY', 'local')  # 'local', 'api', 'hybrid'
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Inicializar Groq (solo si se usa API)
        if self.llm_strategy in ['api', 'hybrid']:
            self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Cargar datos y crear índice
        self.documents = self.load_documents()
        self.index = self.load_or_create_index()
        
        logger.info(f"✅ RAG Service iniciado con estrategia: {self.llm_strategy}")
    
    def load_documents(self):
        """Cargar documentos desde JSON"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Error: No se encontró {self.json_path}")
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
        
        # Crear embeddings
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
        """Búsqueda por palabras clave mejorada con sinónimos"""
        
        def normalize(text):
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            return text.lower()
        
        query_normalized = normalize(query)
        query_words = query_normalized.split()
        
        # Sinónimos del dominio AMPLIADOS
        synonyms = {
            'matricula': ['matricula', 'inscripcion', 'registro'],
            'convalidacion': ['convalidacion', 'validacion', 'reconocimiento'],
            'excepcion': ['excepcion', 'especial', 'extraordinaria'],
            'requisitos': ['requisitos', 'documentos', 'expediente'],
            'cronograma': ['cronograma', 'fecha', 'fechas', 'calendario', 'plazo', 'cuando', 'cuanto'],  # ✅ Ampliado
            'reserva': ['reserva', 'suspension', 'pausa'],
            'reactualizacion': ['reactualizacion', 'reactivacion', 'renovacion'],
            'presentar': ['presentar', 'entregar', 'donde', 'lugar'],  # ✅ Nuevo
            'expediente': ['expediente', 'tramite', 'solicitud', 'documento']  # ✅ Nuevo
        }
        
        # Expandir query con sinónimos
        expanded_words = set(query_words)
        for word in query_words:
            for key, syn_list in synonyms.items():
                if word in syn_list:
                    expanded_words.update(syn_list)
        
        # ✅ DETECTAR PREGUNTAS SOBRE FECHAS/LUGARES
        date_question = any(w in query_normalized for w in ['cuando', 'fecha', 'fechas', 'plazo', 'cronograma'])
        place_question = any(w in query_normalized for w in ['donde', 'lugar', 'presentar', 'entregar'])
        
        keyword_scores = defaultdict(float)
        
        for i, doc in enumerate(documents):
            content_normalized = normalize(doc['content'])
            score = 0
            
            # Buscar palabras expandidas
            for word in expanded_words:
                if word in content_normalized:
                    count = content_normalized.count(word)
                    score += count * 2  # ✅ Aumentado de 1.5 a 2
            
            # Buscar frases completas (mayor peso)
            if query_normalized in content_normalized:
                score += 30  # ✅ Aumentado de 20 a 30
            
            # ✅ BONUS EXTRA para documentos con fechas si se pregunta por fechas
            if date_question:
                # Detectar patrones de fechas en el contenido
                date_patterns = [
                    r'\d{1,2}\s+de\s+\w+',  # "17 de marzo"
                    r'del\s+\d{1,2}\s+al\s+\d{1,2}',  # "del 17 al 28"
                    r'\d{1,2}\s*[-/]\s*\d{1,2}',  # "17-28" o "17/28"
                ]
                
                for pattern in date_patterns:
                    if re.search(pattern, content_normalized):
                        score += 50  # ✅ BONUS MASIVO para documentos con fechas
                        break
                
                # Bonus por campos estructurados
                if 'fecha_relevante' in doc and doc['fecha_relevante']:
                    score += 40
                
                if 'actividad_cronograma' in doc and doc['actividad_cronograma']:
                    score += 30
            
            # ✅ BONUS EXTRA para documentos con lugares si se pregunta por lugares
            if place_question:
                place_keywords = ['escuela', 'oficina', 'caja', 'lugar', 'presentar', 'entregar']
                for kw in place_keywords:
                    if kw in content_normalized:
                        score += 20
                
                if 'lugar_pago' in doc and doc['lugar_pago']:
                    score += 35
            
            # Bonus por keywords del documento
            if 'keywords' in doc:
                for kw in doc.get('keywords', []):
                    if normalize(kw) in query_normalized:
                        score += 10  # ✅ Aumentado de 5 a 10
            
            # ✅ Bonus por categoría relevante
            if 'categoria_principal' in doc:
                categoria = normalize(doc['categoria_principal'])
                if any(w in categoria for w in expanded_words):
                    score += 15
            
            keyword_scores[i] = score
            
        return keyword_scores
    
    def search_documents(self, query, top_k=5):
        """Búsqueda híbrida: semántica + palabras clave"""
        if not self.index or not self.documents:
            return []
        
        # Búsqueda semántica
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k * 3)
        
        # Búsqueda por palabras clave
        keyword_scores = self.keyword_search(query, self.documents)
        
        # ✅ Detectar si es una pregunta sobre fechas/lugares
        query_lower = query.lower()
        is_date_query = any(w in query_lower for w in ['cuando', 'fecha', 'fechas', 'plazo', 'cronograma'])
        is_place_query = any(w in query_lower for w in ['donde', 'lugar', 'presentar', 'entregar'])
        
        # Combinar puntuaciones
        combined_results = []
        for i, score in enumerate(scores[0]):
            if score > 0.2:  # ✅ Umbral reducido de 0.3 a 0.2
                doc_idx = indices[0][i]
                semantic_score = float(score)
                keyword_score = keyword_scores.get(doc_idx, 0)
                
                # ✅ Ajustar pesos dinámicamente según el tipo de pregunta
                if is_date_query or is_place_query:
                    # Dar más peso a keywords cuando se pregunta por fechas/lugares
                    combined_score = (semantic_score * 0.4) + (keyword_score * 0.6)
                else:
                    # Peso normal
                    combined_score = (semantic_score * 0.7) + (keyword_score * 0.3)
                
                combined_results.append({
                    'documento': self.documents[doc_idx],
                    'score': combined_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score
                })
        
        # Ordenar por puntuación combinada
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # ✅ Logging mejorado
        logger.info(f"📊 Query type - Fechas: {is_date_query}, Lugares: {is_place_query}")
        logger.info(f"📊 Recuperados {len(combined_results[:top_k])} documentos para: {query[:50]}...")
        
        for i, doc in enumerate(combined_results[:top_k], 1):
            logger.info(f"  {i}. Score: {doc['score']:.2f} (Sem: {doc['semantic_score']:.2f}, KW: {doc['keyword_score']:.2f})")
        
        return combined_results[:top_k]
    
    def _ensure_ollama_running(self):
        """Verificar que Ollama esté corriendo"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 11434))
            sock.close()
            
            if result == 0:
                logger.info("✅ Ollama está corriendo")
                return True
            else:
                logger.warning("⚠️ Ollama no está corriendo")
                logger.info("💡 Inicia Ollama con: ollama serve")
                return False
                
        except Exception as e:
            logger.error(f"Error verificando Ollama: {e}")
            return False
    
    def _build_prompt(self, query, context):
        """Construir prompt optimizado con extracción forzada de fechas"""
        return f"""Eres un asistente especializado en normativas académicas de la Universidad Nacional de San Agustín (UNSA).

        CONTEXTO RELEVANTE:
        {context}

        Pregunta: {query}

        IMPORTANTE:
        - Si el contexto menciona fechas, cópialas EXACTAMENTE como aparecen
        - Si el contexto menciona lugares, nómbralos específicamente
        - Si el contexto menciona costos, inclúyelos
        - NO inventes información
        - NO uses plantillas como "[día] de [mes]"
        - Sé directo y claro

        Respuesta:"""
        
        
    def _validate_dates_in_response(self, response, context):
        """Validar que las fechas mencionadas existan en el contexto"""
        import re
        
        # Extraer fechas de la respuesta (formato "DD de mes")
        date_patterns = [
            r'\d{1,2}\s+de\s+\w+',  # "17 de marzo"
            r'del\s+\d{1,2}\s+al\s+\d{1,2}',  # "del 17 al 28"
        ]
        
        response_dates = []
        for pattern in date_patterns:
            response_dates.extend(re.findall(pattern, response.lower()))
        
        # Verificar que cada fecha esté en el contexto
        context_lower = context.lower()
        hallucinated_dates = []
        
        for date in response_dates:
            if date not in context_lower:
                hallucinated_dates.append(date)
                logger.warning(f"⚠️ FECHA ALUCINADA DETECTADA: '{date}' no está en el contexto")
        
        if hallucinated_dates:
            logger.error(f"❌ El LLM inventó fechas: {hallucinated_dates}")
            logger.info("🔄 Regenerando respuesta con instrucciones más estrictas...")
            return False
        
        return True
        # Verificar que Ollama esté corriendo
        if not self._ensure_ollama_running():
            raise Exception("Ollama no está disponible. Ejecuta: ollama serve")
        
        try:
            logger.info("🤖 Generando respuesta con Ollama...")
            start_time = time.time()
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 1500,
                        "top_p": 0.8
                    }
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Tiempo de generación: {elapsed:.2f}s")
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                
                # Limpiar respuesta
                answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                answer = re.sub(r'<[^>]+>', '', answer)
                
                return answer.strip()
            else:
                raise Exception(f"Ollama error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error en Ollama: {e}")
            raise
        
    
    
    def _generate_with_groq(self, prompt):
        """Generar respuesta con Groq API"""
        try:
            logger.info("☁️ Generando respuesta con Groq API...")
            start_time = time.time()
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1,
                top_p=0.9
            )
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️ Tiempo de generación: {elapsed:.2f}s")
            
            answer = response.choices[0].message.content.strip()
            
            # Limpiar respuesta
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'<[^>]+>', '', answer)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error en Groq: {e}")
            raise
    
    def _generate_with_ollama(self, prompt, context=None, max_retries=2):
        """Generar respuesta con Ollama (local) con validación"""
        
        # Verificar que Ollama esté corriendo
        if not self._ensure_ollama_running():
            raise Exception("Ollama no está disponible. Ejecuta: ollama serve")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Generando respuesta con Ollama (intento {attempt + 1}/{max_retries})...")
                start_time = time.time()
                
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0 if attempt > 0 else 0.1,  # ✅ Más determinístico en reintentos
                            "num_predict": 1500,
                            "top_p": 0.8  # ✅ Reducir creatividad
                        }
                    },
                    timeout=120
                )
                
                elapsed = time.time() - start_time
                logger.info(f"⏱️ Tiempo de generación: {elapsed:.2f}s")
                
                if response.status_code == 200:
                    answer = response.json()['response'].strip()
                    
                    # Limpiar respuesta
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                    answer = re.sub(r'<[^>]+>', '', answer)
                    answer = answer.strip()
                    
                    # ✅ Validar fechas si tenemos contexto
                    if context and self._validate_dates_in_response(answer, context):
                        return answer
                    elif not context:
                        return answer
                    else:
                        # Si la validación falla, intentar de nuevo con prompt más estricto
                        if attempt < max_retries - 1:
                            logger.warning("⚠️ Reintentando con instrucciones más estrictas...")
                            prompt = prompt.replace(
                                "INSTRUCCIONES CRÍTICAS",
                                "⚠️ ADVERTENCIA: Tu respuesta anterior contenía fechas incorrectas. INSTRUCCIONES CRÍTICAS"
                            )
                            continue
                        else:
                            logger.error("❌ Máximo de reintentos alcanzado. Devolviendo respuesta sin validar.")
                            return answer
                else:
                    raise Exception(f"Ollama error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error en Ollama: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return answer
    
    def generate_response(self, query, context_docs):
        """Generar respuesta con estrategia configurable"""
        try:
            # ✅ Construir contexto ENRIQUECIDO con separadores más claros
            context_parts = []
            
            for idx, doc_wrapper in enumerate(context_docs, 1):
                doc = doc_wrapper['documento']
                
                # Construir bloque de información del documento
                doc_lines = []
                
                # ✅ Agregar ENCABEZADO del documento
                header = f"DOCUMENTO {idx}"
                if doc.get('id_chunk'):
                    header += f" [{doc['id_chunk']}]"
                if doc.get('categoria_principal'):
                    header += f" - {doc['categoria_principal']}"
                if doc.get('sub_categoria'):
                    header += f" > {doc['sub_categoria']}"
                
                doc_lines.append(header)
                doc_lines.append("-" * 50)
                
                # ✅ Agregar METADATA estructurada ANTES del contenido
                if doc.get('actividad_cronograma'):
                    doc_lines.append(f"📌 Actividad: {doc['actividad_cronograma']}")
                
                if doc.get('fecha_relevante'):
                    doc_lines.append(f"📅 FECHAS: {doc['fecha_relevante']}")
                
                if doc.get('lugar_pago'):
                    doc_lines.append(f"📍 Lugar de pago: {doc['lugar_pago']}")
                
                if doc.get('tasa_soles'):
                    doc_lines.append(f"💰 Costo: S/ {doc['tasa_soles']}")
                
                # ✅ Agregar línea divisoria
                if any([doc.get('actividad_cronograma'), doc.get('fecha_relevante'), 
                       doc.get('lugar_pago'), doc.get('tasa_soles')]):
                    doc_lines.append("")
                
                # ✅ Agregar CONTENIDO
                doc_lines.append(f"📄 Información: {doc['content']}")
                
                context_parts.append("\n".join(doc_lines))
            
            # Unir todos los documentos
            context = "\n\n" + "="*70 + "\n\n".join([""] + context_parts) + "\n\n" + "="*70
            
            # DEBUG: Ver qué contexto se envía
            logger.info("=" * 80)
            logger.info("📄 CONTEXTO ENRIQUECIDO ENVIADO AL LLM:")
            logger.info(context[:1000] + "..." if len(context) > 1000 else context)
            logger.info("=" * 80)
            
            prompt = self._build_prompt(query, context)
            
            # Seleccionar estrategia
            if self.llm_strategy == 'local':
                return self._generate_with_ollama(prompt, context=context)
            
            elif self.llm_strategy == 'api':
                return self._generate_with_groq(prompt)
            
            else:  # hybrid
                try:
                    return self._generate_with_ollama(prompt, context=context)
                except Exception as e:
                    logger.warning(f"⚠️ Ollama falló ({e}), usando Groq API...")
                    return self._generate_with_groq(prompt)
        
        except Exception as e:
            logger.error(f"Error en generación: {e}")
            return "Lo siento, no puedo procesar tu consulta en este momento. Por favor, intenta más tarde."
        
    def get_answer(self, question):
        """Método principal para obtener respuesta"""
        logger.info(f"🔍 Nueva consulta: {question}")
        
        # Buscar documentos relevantes
        relevant_docs = self.search_documents(question)
        
        if not relevant_docs:
            return "No encontré información relevante para tu consulta. Por favor, reformula tu pregunta o consulta sobre temas como matrículas, convalidaciones, reservas, o reactualización."
        
        # Log documentos recuperados con título generado
        for i, doc in enumerate(relevant_docs, 1):
            doc_data = doc['documento']
            
            # ✅ Construir título descriptivo multicampo
            title_parts = []
            
            if doc_data.get('categoria_principal'):
                title_parts.append(doc_data['categoria_principal'])
            
            if doc_data.get('actividad_cronograma'):
                title_parts.append(f"({doc_data['actividad_cronograma']})")
            elif doc_data.get('sub_categoria'):
                title_parts.append(f"({doc_data['sub_categoria']})")
            
            if doc_data.get('id_chunk'):
                title_parts.append(f"[{doc_data['id_chunk']}]")
            
            title = " ".join(title_parts) if title_parts else "Documento sin identificador"
            
            logger.info(f"  {i}. {title}")
            logger.info(f"      Score: {doc['score']:.3f} (Sem: {doc['semantic_score']:.2f}, KW: {doc['keyword_score']:.2f})")
        
        # Generar respuesta
        answer = self.generate_response(question, relevant_docs)
        logger.info(f"✅ Respuesta generada: {len(answer)} caracteres")
        
        return answer


# Instancia global del servicio
rag_service = RAGService()