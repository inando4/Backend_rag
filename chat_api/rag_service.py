import json
import os
import faiss
import re
import unicodedata
import socket
import time
import logging
import requests
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from django.conf import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Rutas de archivos
        self.base_path = os.path.join(settings.BASE_DIR, 'data')
        self.json_path = os.path.join(self.base_path, 'dataset_v2.json')
        self.index_path = os.path.join(self.base_path, 'index.faiss')
        
        # Inicializar modelo de embeddings
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # ✅ Configuración LLM (SOLO LOCAL)
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b-instruct')
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Cargar datos y crear índice
        self.documents = self.load_documents()
        self.index = self.load_or_create_index()
        
        logger.info(f"✅ RAG Service iniciado")
        logger.info(f"   Modelo Ollama: {self.ollama_model}")
    
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
            'cronograma': ['cronograma', 'fecha', 'fechas', 'calendario', 'plazo', 'cuando', 'cuanto'],
            'reserva': ['reserva', 'suspension', 'pausa'],
            'reactualizacion': ['reactualizacion', 'reactivacion', 'renovacion'],
            'presentar': ['presentar', 'entregar', 'donde', 'lugar'],
            'expediente': ['expediente', 'tramite', 'solicitud', 'documento'],
            'criterios': ['criterios', 'requisitos', 'condiciones', 'exigencias'],
            'academicos': ['academicos', 'academicas', 'educativos', 'curriculares'],
            'creditaje': ['creditaje', 'creditos', 'credito', 'unidades'],
            'contenidos': ['contenidos', 'contenido', 'temas', 'silabo', 'programa'],
            'similitud': ['similitud', 'equivalencia', 'parecido', 'semejanza'],
            'institutos': ['institutos', 'instituto', 'cetpro', 'senati', 'sencico'],
            'restricciones': ['restricciones', 'limitaciones', 'prohibiciones', 'no se puede', 'no se permite'],
            'pueden': ['pueden', 'puede', 'se puede', 'es posible', 'permiten'],
            'costo': ['costo', 'precio', 'pago', 'tasa', 'tarifa', 'cuanto cuesta', 'cuanto es', 'valor', 'monto'],
            'modalidad': ['modalidad', 'tipo', 'categoria', 'ordinario', 'profesional', 'ceprunsa', 'traslado'],
            'validar': ['validar', 'validacion', 'confirmar', 'confirmacion'],
            'obligatorio': ['obligatorio', 'obligatoriamente', 'debe', 'requerido', 'necesario'],
            'finalizar': ['finalizar', 'terminar', 'culminar', 'concluir', 'completar'],
            'constancia': ['constancia', 'documento', 'comprobante', 'certificado'],
            'imprimir': ['imprimir', 'descargar', 'obtener'],
            'acto': ['acto', 'accion', 'procedimiento', 'proceso', 'tramite'],
            'formal': ['formal', 'oficial', 'administrativo'],
            'acredita': ['acredita', 'certifica', 'avala', 'valida', 'reconoce'],
            'condicion': ['condicion', 'estado', 'situacion', 'calidad'],
            'definicion': ['que es', 'cual es', 'define', 'definicion', 'concepto', 'significa'],
            'ocurre': ['ocurre', 'pasa', 'sucede', 'acontece', 'resulta'],
            'dejan': ['dejan', 'abandonan', 'no se matriculan', 'no matricularse', 'dejar de'],
            'pierden': ['pierden', 'perder', 'perdida', 'perderse'],
            'postular': ['postular', 'volver a postular', 'postulacion', 'admision'],
            'adicionales': ['adicionales', 'extras', 'extra', 'mas', 'de mas'],
            'otorga': ['otorga', 'da', 'concede', 'permite', 'autoriza'],
            'pendiente': ['pendiente', 'sin aprobar', 'reprobado', 'desaprobado'],
            'contactar': ['contactar', 'comunicarse', 'escribir', 'enviar mensaje'],
            'correo': ['correo', 'email', 'correo electronico', 'direccion electronica'],
            'talleres': ['talleres', 'taller', 'extracurriculares', 'extracurricular', 'actividades complementarias'],
            'inscribirse': ['inscribirse', 'inscripcion', 'registrarse', 'registro', 'matricularse'],
            # ✅ NUEVOS SINÓNIMOS PARA MATRÍCULA POR EXCEPCIÓN
            'egresar': ['egresar', 'culminar', 'terminar', 'finalizar carrera'],
            'paralelo': ['paralelo', 'simultaneo', 'al mismo tiempo', 'juntas', 'juntos'],
            'faltan': ['faltan', 'me faltan', 'solo me faltan', 'quedan']
        }
        
        # Expandir query con sinónimos
        expanded_words = set(query_words)
        for word in query_words:
            for key, syn_list in synonyms.items():
                if word in syn_list:
                    expanded_words.update(syn_list)
        
        # Detectar tipo de pregunta
        date_question = any(w in query_normalized for w in ['cuando', 'fecha', 'fechas', 'plazo', 'cronograma'])
        place_question = any(w in query_normalized for w in ['donde', 'lugar', 'presentar', 'entregar'])
        academic_question = any(w in query_normalized for w in ['criterios', 'requisitos', 'academico', 'creditaje', 'contenido', 'similitud'])
        
        restriction_question = (
            any(w in query_normalized for w in ['se pueden', 'se puede', 'puedo', 'permiten', 'permite', 'instituto', 'restriccion', 'prohibido', 'no se'])
            and 'matricula por excepcion' not in query_normalized
            and 'excepcion' not in query_normalized
        )
        
        cost_question = any(w in query_normalized for w in ['cuanto', 'cuesta', 'costo', 'precio', 'pago', 'tasa', 'tarifa', 'valor', 'monto', 's/'])
        
        validation_question = any(w in query_normalized for w in [
            'validar', 'validacion', 'finalizar', 'terminar', 'culminar', 
            'obligatorio', 'obligatoriamente', 'debe', 'despues de registrar',
            'al finalizar', 'al terminar', 'constancia'
        ])
        
        definition_question = any(phrase in query_normalized for phrase in [
            'que es', 'cual es', 'que significa', 'define', 'definicion',
            'concepto', 'se considera', 'se entiende por',
            'acto formal', 'acto voluntario', 'acredita la condicion'
        ])
        
        payment_procedure_question = any(w in query_normalized for w in [
            'varios recibos', 'un solo recibo', 'un recibo', 'varios pagos', 
            'puedo pagar', 'como pago', 'forma de pago', 'procedimiento de pago',
            'en cuotas', 'en partes', 'fraccionado'
        ])
        
        amount_question = any(w in query_normalized for w in [
            'cuanto cuesta', 'cuanto es', 'cual es el costo', 'cual es el precio',
            'monto', 'valor', 's/', 'soles'
        ])
        
        consequence_question = any(phrase in query_normalized for phrase in [
            'que ocurre', 'que pasa', 'que sucede',
            'dejan de matricularse', 'no se matriculan', 'no matricularse',
            'mas de tres', 'mas de 3', 'despues de tres', 'luego de tres',
            'pierden', 'perder la condicion', 'volver a postular'
        ])
        
        credits_question = any(phrase in query_normalized for phrase in [
            'creditos adicionales', 'creditos extra', 'creditos de mas',
            'cuantos creditos adicionales', 'cuantos creditos extras',
            'cuantos creditos mas', 'creditos adicionales me otorga',
            'sin cursos pendientes', 'no tengo cursos pendientes',
            'ningun curso pendiente', 'sin ningún curso pendiente'
        ])
        
        contact_question = any(phrase in query_normalized for phrase in [
            'que correo', 'cual es el correo', 'correo electronico',
            'a que correo', 'donde contactar', 'como contactar',
            'talleres extracurriculares', 'talleres', 'inscribirme en talleres',
            'oficina de', 'upacdr'
        ])
        
        exception_enrollment_question = any(phrase in query_normalized for phrase in [
            'matricula por excepcion',
            'por excepcion',
            'excepcionalmente',
            'faltan dos asignaturas',
            'dos asignaturas para egresar',
            'una es prerrequisito',
            'llevarlas juntas',
            'llevar en paralelo',
            'llevar las dos'
        ])
        
        # ✅ NUEVO: Detectar preguntas sobre AUTORIDADES/RESPONSABLES
        authority_question = any(phrase in query_normalized for phrase in [
            'quien establece',
            'quien programa',
            'quien define',
            'quien aprueba',
            'quien determina',
            'que entidad',
            'que organo',
            'consejo universitario',
            'decano',
            'director',
            'vicerrectorado'
        ])
        
        # ✅ NUEVO: Detectar preguntas sobre EQUIVALENCIA DE ABANDONO
        equivalence_question = any(phrase in query_normalized for phrase in [
            'equivalente',
            'equivale',
            'es equivalente a',
            'abandono es equivalente',
            'condicion de abandono',
            'conteo de matriculas',
            'matriculas ejecutadas'
        ])
        
        keyword_scores = defaultdict(float)
        
        for i, doc in enumerate(documents):
            content_normalized = normalize(doc['content'])
            score = 0
            
            # ✅ LÓGICA ESPECIAL PARA PREGUNTAS SOBRE AUTORIDADES (DEBE IR PRIMERO)
            if authority_question:
                authority_keywords = [
                    'consejo universitario',
                    'establecera',
                    'establece',
                    'calendario academico',
                    'programa las fechas',
                    'programara',
                    'articulo 11',
                    'anualmente'
                ]
                
                matches = sum(1 for kw in authority_keywords if kw in content_normalized)
                
                has_council = 'consejo universitario' in content_normalized
                has_calendar = 'calendario academico' in content_normalized
                has_establish = any(word in content_normalized for word in ['establecera', 'establece', 'programa'])
                
                if has_council and has_calendar and has_establish:
                    score = 4000
                    logger.info(f"    🎯 MATCH PERFECTO de autoridad (Consejo + calendario) en {doc.get('id_chunk')}")
                elif matches >= 3:
                    score = 3500
                    logger.info(f"    ✅ Autoridad encontrada en {doc.get('id_chunk')}: {matches} keywords")
                elif matches >= 2:
                    score = 2500 + (matches * 200)
                    logger.info(f"    ✅ Parcial autoridad en {doc.get('id_chunk')}: {matches} keywords")
                elif matches == 1:
                    score = 1000
                    logger.info(f"    ⚠️ Débil coincidencia en {doc.get('id_chunk')}: {matches} keyword")
                else:
                    score = 20
                
                keyword_scores[i] = score
                continue
            
            # ✅ LÓGICA ESPECIAL PARA PREGUNTAS SOBRE EQUIVALENCIA DE ABANDONO
            if equivalence_question:
                equivalence_keywords = [
                    'abandono',
                    'equivalente',
                    'desaprobacion',
                    'conteo de matriculas',
                    'matriculas ejecutadas',
                    'calificacion final',
                    'disposicion final',
                    'primera'
                ]
                
                matches = sum(1 for kw in equivalence_keywords if kw in content_normalized)
                
                has_abandonment = 'abandono' in content_normalized
                has_equivalent = any(word in content_normalized for word in ['equivalente', 'equivale'])
                has_disapproval = 'desaprobacion' in content_normalized
                has_counting = any(phrase in content_normalized for phrase in ['conteo de matriculas', 'matriculas ejecutadas'])
                
                if has_abandonment and has_equivalent and (has_disapproval or has_counting):
                    score = 4000
                    logger.info(f"    🎯 MATCH PERFECTO de equivalencia de abandono en {doc.get('id_chunk')}")
                elif matches >= 3:
                    score = 3500
                    logger.info(f"    ✅ Equivalencia encontrada en {doc.get('id_chunk')}: {matches} keywords")
                elif matches >= 2:
                    score = 2500 + (matches * 200)
                    logger.info(f"    ✅ Parcial equivalencia en {doc.get('id_chunk')}: {matches} keywords")
                elif matches == 1:
                    score = 1000
                    logger.info(f"    ⚠️ Débil coincidencia en {doc.get('id_chunk')}: {matches} keyword")
                else:
                    score = 20
                
                keyword_scores[i] = score
                continue
            
            # ✅ LÓGICA ESPECIAL PARA MATRÍCULA POR EXCEPCIÓN
            if exception_enrollment_question:
                # Palabras clave críticas
                exception_keywords = [
                    'matricula por excepcion',
                    'excepcion',
                    'dos (2) asignaturas',
                    'dos asignaturas',
                    'para egresar',
                    'egresar',
                    'prerrequisito',
                    'en paralelo',
                    'llevar las dos',
                    'simultaneamente'
                ]
                
                # Buscar coincidencias
                matches = sum(1 for kw in exception_keywords if kw in content_normalized)
                
                # Bonus especial si es el documento MEX-003-EGR2 o similar
                has_two_courses = any(phrase in content_normalized for phrase in ['dos (2) asignaturas', 'dos asignaturas', 'falte solo dos'])
                has_prerequisite = 'prerrequisito' in content_normalized
                has_parallel = any(phrase in content_normalized for phrase in ['en paralelo', 'llevar las dos', 'llevar los dos'])
                has_exception = 'excepcion' in content_normalized
                
                if has_two_courses and has_prerequisite and has_parallel:
                    score = 4000  # Score ALTÍSIMO para match perfecto
                    logger.info(f"    🎯 MATCH PERFECTO de matrícula por excepción (2 cursos + prerreq) en {doc.get('id_chunk')}")
                elif matches >= 3:
                    score = 3500
                    logger.info(f"    ✅ Matrícula por excepción encontrada en {doc.get('id_chunk')}: {matches} keywords")
                elif matches >= 2:
                    score = 2500 + (matches * 200)
                    logger.info(f"    ✅ Parcial matrícula por excepción en {doc.get('id_chunk')}: {matches} keywords")
                elif matches == 1:
                    score = 1000
                    logger.info(f"    ⚠️ Débil coincidencia en {doc.get('id_chunk')}: {matches} keyword")
                else:
                    score = 20
                
                keyword_scores[i] = score
                continue
            
            # LÓGICA ESPECIAL PARA PREGUNTAS DE CONTACTO/TALLERES
            if contact_question:
                # Palabras clave críticas
                contact_keywords = [
                    'talleres extracurriculares',
                    'taller',
                    'correo',
                    'upacdr',
                    'oficina de promocion',
                    'arte, cultura, deporte',
                    'contactar',
                    'inscripcion',
                    '@unsa.edu.pe'
                ]
                
                # Buscar coincidencias
                matches = sum(1 for kw in contact_keywords if kw in content_normalized)
                
                # Bonus especial si es el documento de talleres extracurriculares
                has_talleres = 'talleres extracurriculares' in content_normalized
                has_email = '@unsa.edu.pe' in content_normalized or 'upacdr' in content_normalized
                has_office = 'oficina' in content_normalized
                
                if has_talleres and has_email:
                    score = 3500  # Score ALTÍSIMO para match perfecto
                    logger.info(f"    🎯 MATCH PERFECTO de talleres + correo en {doc.get('id_chunk')}")
                elif has_talleres or (has_email and has_office):
                    score = 3000
                    logger.info(f"    ✅ Talleres/contacto encontrado en {doc.get('id_chunk')}: {matches} keywords")
                elif matches >= 2:
                    score = 2500 + (matches * 200)
                    logger.info(f"    ✅ Contacto/talleres encontrado en {doc.get('id_chunk')}: {matches} keywords")
                elif matches == 1:
                    score = 1000
                    logger.info(f"    ⚠️ Parcialmente relevante en {doc.get('id_chunk')}: {matches} keyword")
                else:
                    score = 20
                
                keyword_scores[i] = score
                continue
            
            # LÓGICA ESPECIAL PARA PREGUNTAS SOBRE CRÉDITOS ADICIONALES
            if credits_question:
                credit_keywords = [
                    'seis (6) creditos adicionales',
                    'seis (06) creditos adicionales',
                    '6 creditos adicionales',
                    '06 creditos adicionales',
                    'creditos adicionales',
                    'sin cursos pendientes',
                    'ningun curso pendiente',
                    'no tengan ningun curso',
                    'sistema considera automaticamente'
                ]
                
                matches = sum(1 for kw in credit_keywords if kw in content_normalized)
                
                has_six = any(num in content_normalized for num in ['seis (6)', 'seis (06)', '6 creditos', '06 creditos'])
                has_additional = 'creditos adicionales' in content_normalized
                has_no_pending = any(phrase in content_normalized for phrase in ['sin cursos pendientes', 'ningun curso pendiente', 'no tengan ningun curso'])
                
                if has_six and has_additional and has_no_pending:
                    score = 3500
                    logger.info(f"    🎯 MATCH PERFECTO de créditos adicionales en {doc.get('id_chunk')}")
                elif has_six and has_additional:
                    score = 3000
                    logger.info(f"    ✅ Créditos adicionales (con número) en {doc.get('id_chunk')}: {matches} keywords")
                elif matches >= 2:
                    score = 2500 + (matches * 200)
                    logger.info(f"    ✅ Créditos adicionales encontrados en {doc.get('id_chunk')}: {matches} keywords")
                elif matches == 1:
                    score = 1000
                    logger.info(f"    ⚠️ Parcialmente relevante en {doc.get('id_chunk')}: {matches} keyword")
                else:
                    score = 20
                
                keyword_scores[i] = score
                continue
            
            # LÓGICA ESPECIAL PARA PREGUNTAS SOBRE CONSECUENCIAS
            if consequence_question:
                consequence_keywords = [
                    'perderan', 'pierden', 'perder', 'perdida',
                    'condicion de estudiante', 'condicion',
                    'postular nuevamente', 'volver a postular', 'postulacion',
                    'mas de tres', 'mas de 3 anos', 'tres anos',
                    'consecutivos o alternos', 'consecutivos', 'alternos'
                ]
                
                matches = sum(1 for kw in consequence_keywords if kw in content_normalized)
                
                if matches >= 2:
                    score = 3000 + (matches * 300)
                    logger.info(f"    🎯 Consecuencias encontradas en {doc.get('id_chunk')}: {matches} keywords")
                elif matches == 1:
                    score = 1500
                    logger.info(f"    ✅ Parcialmente relevante en {doc.get('id_chunk')}: {matches} keyword")
                else:
                    score = 20
                
                keyword_scores[i] = score
                continue
            
            # LÓGICA ESPECIAL PARA PREGUNTAS CONCEPTUALES
            if definition_question:
                definition_keywords = [
                    'es el acto', 'se considera', 'se define', 'definicion',
                    'acto formal', 'acto voluntario', 'acredita', 'condicion de estudiante',
                    'articulo 3', 'articulo 4', 'articulo'
                ]
                
                if 'acto formal' in query_normalized and 'acredita' in query_normalized:
                    if 'acto formal' in content_normalized and 'acredita' in content_normalized and 'condicion de estudiante' in content_normalized:
                        score = 3000
                        logger.info(f"    🎯 Definición EXACTA encontrada en {doc.get('id_chunk')}")
                    else:
                        score = 50
                else:
                    matches = sum(1 for kw in definition_keywords if kw in content_normalized)
                    if matches > 0:
                        score = 2000 + (matches * 200)
                        logger.info(f"    ✅ Definición encontrada en {doc.get('id_chunk')}: {matches} keywords")
                    else:
                        score = 20
                
                keyword_scores[i] = score
                continue
            
            # LÓGICA ESPECIAL PARA PREGUNTAS DE VALIDACIÓN
            if validation_question:
                validation_keywords = [
                    'validar', 'validacion', 'validar su matricula',
                    'obligatorio', 'obligado', 'debe',
                    'constancia', 'imprimir', 'finalizar', 'finalmente'
                ]
                
                matches = sum(1 for kw in validation_keywords if kw in content_normalized)
                
                if matches > 0:
                    score = 2000 + (matches * 200)
                    logger.info(f"    ✅ Validación encontrada en {doc.get('id_chunk')}: {matches} keywords")
                else:
                    score = 10
                
                keyword_scores[i] = score
                continue
            
            # LÓGICA ESPECIAL PARA PREGUNTAS DE COSTOS
            if cost_question:
                if payment_procedure_question:
                    payment_keywords = ['recibo', 'recibos', 'un solo', 'todas las asignaturas', 'monto total', 'pago']
                    keyword_count = sum(1 for kw in payment_keywords if kw in content_normalized)
                    
                    if keyword_count > 0:
                        score = 2000 + (keyword_count * 100)
                        logger.info(f"    ✅ Procedimiento de pago encontrado en {doc.get('id_chunk')}: {keyword_count} keywords")
                    else:
                        score = 5
                    
                    keyword_scores[i] = score
                    continue
                
                elif amount_question:
                    if 'tasa_soles' not in doc or not doc.get('tasa_soles'):
                        keyword_scores[i] = 0
                        continue
                    
                    score = 1000
                    
                    modalidades_en_query = []
                    if 'ordinario' in query_normalized:
                        modalidades_en_query.append('ordinario')
                    if 'profesional' in query_normalized or 'profesionales' in query_normalized:
                        modalidades_en_query.append('profesionales')
                    if 'ceprunsa' in query_normalized:
                        modalidades_en_query.append('ceprunsa')
                    if 'traslado' in query_normalized:
                        modalidades_en_query.append('traslado')
                    
                    procedencias_en_query = []
                    if 'otra escuela' in query_normalized or 'escuela unsa' in query_normalized:
                        procedencias_en_query.append('otra_escuela_unsa')
                    if 'universidad nacional' in query_normalized and 'otra' in query_normalized:
                        procedencias_en_query.append('universidad_nacional_otra')
                    if 'universidad particular' in query_normalized or 'universidad privada' in query_normalized:
                        procedencias_en_query.append('universidad_particular')
                    
                    doc_modalidad_value = doc.get('modalidad_pago_relacionada', '')
                    if doc_modalidad_value:
                        doc_modalidad_norm = normalize(doc_modalidad_value)
                        
                        modalidad_match = False
                        procedencia_match = False
                        
                        for modalidad in modalidades_en_query:
                            if modalidad in doc_modalidad_norm:
                                modalidad_match = True
                                score += 500
                                logger.info(f"    ✅ Modalidad '{modalidad}' encontrada en {doc.get('id_chunk')}")
                                break
                        
                        if 'universidad_particular' in procedencias_en_query:
                            if 'universidad particular' in doc_modalidad_norm:
                                procedencia_match = True
                                score += 800
                                logger.info(f"    ✅ Procedencia 'universidad particular' encontrada en {doc.get('id_chunk')}")
                        elif 'universidad_nacional_otra' in procedencias_en_query:
                            if 'universidad nacional' in doc_modalidad_norm and 'otra' in doc_modalidad_norm:
                                procedencia_match = True
                                score += 800
                                logger.info(f"    ✅ Procedencia 'universidad nacional (otra)' encontrada en {doc.get('id_chunk')}")
                        elif 'otra_escuela_unsa' in procedencias_en_query:
                            if 'otra escuela de la unsa' in doc_modalidad_norm or 'escuela de la unsa' in doc_modalidad_norm:
                                procedencia_match = True
                                score += 800
                                logger.info(f"    ✅ Procedencia 'otra escuela UNSA' encontrada en {doc.get('id_chunk')}")
                        
                        if modalidades_en_query and not modalidad_match:
                            score = 10
                            logger.info(f"    ❌ Modalidad NO coincide en {doc.get('id_chunk')}: '{doc_modalidad_value}'")
                        
                        if procedencias_en_query and not procedencia_match:
                            score = score * 0.05
                            logger.info(f"    ❌ Procedencia NO coincide en {doc.get('id_chunk')}: '{doc_modalidad_value}'")
                    
                    keyword_scores[i] = score
                    continue
                
                else:
                    if 'tasa_soles' in doc and doc.get('tasa_soles'):
                        score = 1000
                    elif any(kw in content_normalized for kw in ['recibo', 'pago', 'un solo', 'todas las asignaturas']):
                        score = 1500
                    else:
                        score = 0
                    
                    keyword_scores[i] = score
                    continue
            
            # Para preguntas NO especiales, usar lógica normal
            for word in expanded_words:
                if word in content_normalized:
                    count = content_normalized.count(word)
                    score += count * 2
            
            if query_normalized in content_normalized:
                score += 30
            
            # BONUS para preguntas sobre RESTRICCIONES
            if restriction_question:
                restriction_keywords = ['no se convalidan', 'restriccion', 'prohibido', 'no se permite', 'instituto', 'institutos', 'obligatorio']
                for kw in restriction_keywords:
                    if kw in content_normalized:
                        score += 60
                
                if 'sub_categoria' in doc:
                    sub_cat_value = doc.get('sub_categoria', '')
                    if sub_cat_value:
                        sub_cat = normalize(sub_cat_value)
                        if 'restriccion' in sub_cat or 'limitacion' in sub_cat:
                            score += 70
                
                negation_patterns = [
                    r'no se convalidan',
                    r'no se puede',
                    r'no se permite',
                    r'prohibido',
                    r'es obligatorio'
                ]
                
                for pattern in negation_patterns:
                    if re.search(pattern, content_normalized):
                        score += 50
            
            # BONUS para preguntas sobre criterios académicos
            if academic_question:
                academic_keywords = ['creditaje', 'creditos', 'similitud', '80%', 'contenido', 'igual', 'mayor']
                for kw in academic_keywords:
                    if kw in content_normalized:
                        score += 40
                
                if 'sub_categoria' in doc:
                    sub_cat_value = doc.get('sub_categoria', '')
                    if sub_cat_value and 'academico' in normalize(sub_cat_value):
                        score += 50
            
            # BONUS EXTRA para documentos con fechas
            if date_question:
                date_patterns = [
                    r'\d{1,2}\s+de\s+\w+',
                    r'del\s+\d{1,2}\s+al\s+\d{1,2}',
                    r'\d{1,2}\s*[-/]\s*\d{1,2}',
                ]
                
                for pattern in date_patterns:
                    if re.search(pattern, content_normalized):
                        score += 50
                        break
                
                if 'fecha_relevante' in doc and doc['fecha_relevante']:
                    score += 40
                
                if 'actividad_cronograma' in doc and doc['actividad_cronograma']:
                    score += 30
            
            # BONUS EXTRA para documentos con lugares
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
                        score += 10
            
            # Bonus por categoría relevante
            if 'categoria_principal' in doc:
                categoria_value = doc.get('categoria_principal', '')
                if categoria_value:
                    categoria = normalize(categoria_value)
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
        
        # Detectar tipo de pregunta
        query_lower = query.lower()
        is_date_query = any(w in query_lower for w in ['cuando', 'fecha', 'fechas', 'plazo', 'cronograma'])
        is_place_query = any(w in query_lower for w in ['donde', 'lugar', 'presentar', 'entregar'])
        is_restriction_query = (
            any(w in query_lower for w in ['se pueden', 'se puede', 'puedo', 'permiten', 'permite', 'instituto', 'restriccion', 'prohibido'])
            and 'matricula por excepcion' not in query_lower
            and 'excepcion' not in query_lower
        )
        is_cost_query = any(w in query_lower for w in ['cuanto', 'cuesta', 'costo', 'precio', 'pago', 'tasa', 'tarifa', 'valor', 'monto', 's/'])
        is_validation_query = any(w in query_lower for w in ['validar', 'validacion', 'finalizar', 'terminar', 'obligatorio', 'obligatoriamente', 'constancia', 'al finalizar'])
        is_definition_query = any(phrase in query_lower for phrase in ['que es', 'cual es', 'que significa', 'define', 'definicion', 'concepto', 'acto formal', 'acredita la condicion'])
        is_consequence_query = any(phrase in query_lower for phrase in ['que ocurre', 'que pasa', 'que sucede', 'dejan de matricularse', 'mas de tres', 'pierden'])
        is_credits_query = any(phrase in query_lower for phrase in ['creditos adicionales', 'creditos extra', 'cuantos creditos', 'sin cursos pendientes', 'ningun curso pendiente'])
        is_contact_query = any(phrase in query_lower for phrase in ['que correo', 'cual es el correo', 'correo electronico', 'talleres extracurriculares', 'talleres', 'contactar', 'inscribirme'])
        # ✅ NUEVO
        is_exception_enrollment_query = any(phrase in query_lower for phrase in ['matricula por excepcion', 'por excepcion', 'faltan dos asignaturas', 'llevarlas juntas', 'llevar en paralelo'])
        # ✅ NUEVO
        is_authority_query = any(phrase in query_lower for phrase in ['quien establece', 'quien programa', 'quien define', 'consejo universitario', 'que entidad', 'que organo'])
        # ✅ NUEVO
        is_equivalence_query = any(phrase in query_lower for phrase in ['equivalente', 'equivale', 'es equivalente a', 'abandono es equivalente', 'conteo de matriculas'])
        
        # Combinar puntuaciones
        combined_results = []
        for i, score in enumerate(scores[0]):
            if score > 0.2:
                doc_idx = indices[0][i]
                semantic_score = float(score)
                keyword_score = keyword_scores.get(doc_idx, 0)
                
                # Ajustar pesos dinámicamente
                if is_equivalence_query:  # ✅ NUEVO - Keywords dominan totalmente
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)  # 95% keywords
                elif is_authority_query:
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)
                elif is_exception_enrollment_query:
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)
                elif is_contact_query:
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)
                elif is_credits_query:
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)
                elif is_consequence_query:
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)
                elif is_definition_query:
                    combined_score = (semantic_score * 0.05) + (keyword_score * 0.95)
                elif is_validation_query:
                    combined_score = (semantic_score * 0.1) + (keyword_score * 0.9)
                elif is_cost_query:
                    combined_score = (semantic_score * 0.2) + (keyword_score * 0.8)
                elif is_restriction_query:
                    combined_score = (semantic_score * 0.3) + (keyword_score * 0.7)
                elif is_date_query or is_place_query:
                    combined_score = (semantic_score * 0.4) + (keyword_score * 0.6)
                else:
                    combined_score = (semantic_score * 0.7) + (keyword_score * 0.3)
                
                combined_results.append({
                    'documento': self.documents[doc_idx],
                    'score': combined_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score
                })
        
        # Ordenar por puntuación combinada
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Logging mejorado
        logger.info(f"📊 Query type - Fechas: {is_date_query}, Lugares: {is_place_query}, Restricciones: {is_restriction_query}, Costos: {is_cost_query}, Validación: {is_validation_query}, Definición: {is_definition_query}, Consecuencias: {is_consequence_query}, Créditos: {is_credits_query}, Contacto: {is_contact_query}, Matrícula Excepción: {is_exception_enrollment_query}, Autoridad: {is_authority_query}, Equivalencia: {is_equivalence_query}")
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
        """Construir prompt optimizado"""
        return f"""Eres un asistente especializado en normativas académicas de la Universidad Nacional de San Agustín (UNSA).

CONTEXTO (múltiples documentos relacionados):
{context}

PREGUNTA DEL ESTUDIANTE: {query}

INSTRUCCIONES CRÍTICAS:

1. **LEE CUIDADOSAMENTE** cada documento del contexto - están numerados (DOCUMENTO 1, DOCUMENTO 2, etc.)

2. **EXTRAE INFORMACIÓN ESPECÍFICA** según la pregunta:
   - Si preguntan "DÓNDE": Busca en "📍 Lugar" o en el contenido principal
   - Si preguntan "CUÁNDO/FECHAS": Busca en "📅 FECHAS" 
   - Si preguntan "CUÁNTO/COSTO": Busca en "💰 Costo"

3. **NO MEZCLES INFORMACIÓN** de diferentes documentos:
   - Un documento sobre "Presentación de expedientes" NO es lo mismo que "Pago"
   - Un documento sobre "Lugar de pago" NO es el lugar de presentación del expediente

4. **PRIORIZA** el documento más relevante,el DOCUMENTO 1 es el MÁS RELEVANTE para esta pregunta.
   - Si el DOCUMENTO 1 contiene la respuesta completa, ÚSALO y NO mezcles con otros documentos.
   - Solo usa otros documentos si el DOCUMENTO 1 no tiene información suficiente.

5. **FORMATO DE RESPUESTA**:
   - Responde de forma directa y estructurada
   - Si hay fechas, escríbelas como: "Del **17 de marzo** al **28 de marzo**"
   - Si hay lugares, especifica claramente: "en [lugar exacto]"
   - Si hay costos, menciónalos: "S/ [monto]"
   - Brinda la información sin mencionar los documentos de donde la extraiste.

6. **PROHIBIDO**:
   - Inventar información que no esté en el contexto
   - Mezclar información de documentos diferentes
   - Usar plantillas como "[día] de [mes]"

7. Si NO encuentras información específica en el contexto, di: "No encontré información sobre [tema específico]"

RESPUESTA:"""
    
    def _validate_dates_in_response(self, response, context):
        """Validar que las fechas mencionadas existan en el contexto"""
        # Extraer fechas de la respuesta
        date_patterns = [
            r'\d{1,2}\s+de\s+\w+',
            r'del\s+\d{1,2}\s+al\s+\d{1,2}',
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
            return False
        
        return True
    
    def _clean_response(self, response):
        """Limpiar respuesta de referencias a documentos internos"""
        # Eliminar referencias a "DOCUMENTO X"
        response = re.sub(r'Según el DOCUMENTO \d+[^,.:]*,?\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'En el DOCUMENTO \d+[^,.:]*,?\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'El DOCUMENTO \d+[^,.:]*\s+(indica|dice|menciona|establece)\s+que\s*', '', response, flags=re.IGNORECASE)
        
        # Eliminar códigos entre corchetes
        response = re.sub(r'\[CONV-[A-Z0-9-]+\]', '', response)
        response = re.sub(r'\[MAT-[A-Z0-9-]+\]', '', response)
        response = re.sub(r'\[RES-[A-Z0-9-]+\]', '', response)
        
        # Eliminar frases como "según el contexto proporcionado"
        response = re.sub(r'Según el contexto proporcionado,?\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'Basándome en la información proporcionada,?\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'De acuerdo (al|con el) contexto,?\s*', '', response, flags=re.IGNORECASE)
        
        # Limpiar espacios múltiples
        response = re.sub(r'\s+', ' ', response)
        
        return response.strip()
    
    def _generate_with_ollama(self, prompt, context=None, max_retries=2):
        """Generar respuesta con Ollama local"""
        
        if not self._ensure_ollama_running():
            raise Exception("Ollama no está disponible. Ejecuta: ollama serve")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 Generando con {self.ollama_model} (intento {attempt + 1}/{max_retries})...")
                start_time = time.time()
                
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0 if attempt > 0 else 0.1,
                            "num_predict": 1500,
                            "top_p": 0.8
                        }
                    },
                    timeout=180
                )
                
                elapsed = time.time() - start_time
                logger.info(f"⏱️ Tiempo: {elapsed:.2f}s")
                
                if response.status_code == 200:
                    answer = response.json()['response'].strip()
                    
                    # Limpiar respuesta
                    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                    answer = re.sub(r'<[^>]+>', '', answer)
                    answer = self._clean_response(answer)
                    
                    # Validar fechas si tenemos contexto
                    if context and self._validate_dates_in_response(answer, context):
                        return answer
                    elif not context:
                        return answer
                    else:
                        if attempt < max_retries - 1:
                            logger.warning("⚠️ Reintentando con instrucciones más estrictas...")
                            prompt = prompt.replace(
                                "INSTRUCCIONES CRÍTICAS:",
                                "⚠️ ADVERTENCIA: Tu respuesta anterior contenía fechas incorrectas. INSTRUCCIONES CRÍTICAS:"
                            )
                            continue
                        else:
                            logger.error("❌ Máximo de reintentos alcanzado.")
                            return answer
                else:
                    raise Exception(f"Ollama error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error en Ollama: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return answer
    
    def generate_response(self, query, context_docs):
        """Generar respuesta usando Ollama local"""
        try:
            # Construir contexto ENRIQUECIDO
            context_parts = []
            
            for idx, doc_wrapper in enumerate(context_docs, 1):
                doc = doc_wrapper['documento']
                
                doc_lines = []
                
                # Agregar ENCABEZADO del documento
                header = f"DOCUMENTO {idx}"
                if doc.get('id_chunk'):
                    header += f" [{doc['id_chunk']}]"
                if doc.get('categoria_principal'):
                    header += f" - {doc['categoria_principal']}"
                if doc.get('sub_categoria'):
                    header += f" > {doc['sub_categoria']}"
                
                doc_lines.append(header)
                doc_lines.append("-" * 50)
                
                # Agregar METADATA estructurada
                if doc.get('actividad_cronograma'):
                    doc_lines.append(f"📌 Actividad: {doc['actividad_cronograma']}")
                
                if doc.get('fecha_relevante'):
                    doc_lines.append(f"📅 FECHAS: {doc['fecha_relevante']}")
                
                if doc.get('lugar_pago'):
                    doc_lines.append(f"📍 Lugar de pago: {doc['lugar_pago']}")
                
                if doc.get('tasa_soles'):
                    doc_lines.append(f"💰 Costo: S/ {doc['tasa_soles']}")
                
                if any([doc.get('actividad_cronograma'), doc.get('fecha_relevante'), 
                       doc.get('lugar_pago'), doc.get('tasa_soles')]):
                    doc_lines.append("")
                
                # Agregar CONTENIDO
                doc_lines.append(f"📄 Información: {doc['content']}")
                
                context_parts.append("\n".join(doc_lines))
            
            # Unir todos los documentos
            context = "\n\n" + "="*70 + "\n\n".join([""] + context_parts) + "\n\n" + "="*70
            
            # DEBUG: Ver qué contexto se envía
            logger.info("=" * 80)
            logger.info("📄 CONTEXTO ENVIADO AL LLM:")
            logger.info(context[:1000] + "..." if len(context) > 1000 else context)
            logger.info("=" * 80)
            
            prompt = self._build_prompt(query, context)
            
            # Generar respuesta con Ollama
            return self._generate_with_ollama(prompt, context=context)
        
        except Exception as e:
            logger.error(f"Error en generación: {e}")
            return "Lo siento, no puedo procesar tu consulta en este momento. Por favor, intenta más tarde."
    
    def get_answer(self, question):
        """Método principal para obtener respuesta"""
        logger.info(f"🔍 Nueva consulta: {question}")
        
        # Buscar documentos relevantes
        relevant_docs = self.search_documents(question)
        
        if not relevant_docs:
            return "No encontré información relevante para tu consulta. Por favor, reformula tu pregunta."
        
        # Log documentos recuperados
        for i, doc in enumerate(relevant_docs, 1):
            doc_data = doc['documento']
            
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