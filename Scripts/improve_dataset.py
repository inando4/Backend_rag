import json
import os
import re
from pathlib import Path

def normalize_text(text):
    """Normalizar texto para mejor procesamiento"""
    import unicodedata
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text.lower()

def extract_keywords(content):
    """Extraer palabras clave del contenido"""
    content_lower = normalize_text(content)
    
    # Palabras clave importantes del dominio
    domain_keywords = {
        'matricula': ['matricula', 'matrícula', 'inscripción', 'registro'],
        'modificacion': ['modificacion', 'modificación', 'cambio', 'rectificación'],
        'convalidacion': ['convalidacion', 'convalidación', 'validación'],
        'excepcion': ['excepcion', 'excepción', 'especial'],
        'procedimiento': ['procedimiento', 'proceso', 'tramite', 'trámite'],
        'cronograma': ['cronograma', 'fecha', 'calendario', 'plazo'],
        'requisitos': ['requisitos', 'documentos', 'expediente'],
        'reserva': ['reserva', 'suspensión', 'pausa'],
        'reactualizacion': ['reactualizacion', 'reactualización', 'reactivación']
    }
    
    found_keywords = []
    for main_keyword, synonyms in domain_keywords.items():
        if any(synonym in content_lower for synonym in synonyms):
            found_keywords.append(main_keyword)
    
    # Agregar fechas y años encontrados
    date_patterns = [
        r'\b\d{1,2}\s+de\s+\w+',  # "28 de marzo"
        r'\b\w+\s+\d{4}',         # "marzo 2025"
        r'\b\d{4}\s*[-A-Z]*\b',   # "2025 A"
        r'\bdel\s+\d{1,2}\s+al\s+\d{1,2}'  # "del 17 al 28"
    ]
    
    for pattern in date_patterns:
        if re.search(pattern, content_lower):
            found_keywords.append('fechas')
            break
    
    return found_keywords

def group_related_documents(documents):
    """Agrupar documentos relacionados"""
    
    # Definir grupos de documentos que deben combinarse
    groups = {
        'convalidaciones': {
            'ids': ['conv_consideraciones_1_7', 'conv_requisitos_1_6', 'conv_observaciones_1_3', 
                   'conv_matricula_1_3', 'conv_silabo_1_2', 'conv_formato_1_10', 
                   'conv_tasas_modalidades', 'conv_cronograma_2025'],
            'title': 'Convalidaciones - Proceso Completo',
            'category': 'convalidaciones'
        },
        'excepcion_matricula': {
            'ids': ['exc_responsabilidad', 'exc_por_egresar', 'exc_curriculo_rigido', 
                   'exc_casos_no_contemplados', 'exc_procedimiento_1_6', 'exc_procedimiento_7_11'],
            'title': 'Matrícula por Excepción - Proceso Completo',
            'category': 'excepcion'
        },
        'evaluacion_jurados': {
            'ids': ['evaluacion_jurados_2025a_1', 'evaluacion_jurados_2025a_2'],
            'title': 'Evaluación por Jurados - Información Completa',
            'category': 'evaluacion'
        },
        'reactualizacion': {
            'ids': ['react_concepto_aplicacion', 'react_pago_derechos', 'react_procedimiento_envio',
                   'react_proceso_matricula', 'react_plazo_tramite'],
            'title': 'Reactualización de Matrícula - Proceso Completo',
            'category': 'reactualizacion'
        },
        'reserva': {
            'ids': ['res_restriccion_ingresantes', 'res_condiciones_generales', 'res_requisitos_expediente',
                   'res_tramite_fechas', 'res_proceso_retorno'],
            'title': 'Reserva de Matrícula - Proceso Completo',
            'category': 'reserva'
        },
        'levantamiento_reserva': {
            'ids': ['lev_reserva_concepto', 'lev_reserva_requisitos', 'lev_reserva_fechas_proceso',
                   'lev_reserva_post_proceso'],
            'title': 'Levantamiento de Reserva - Proceso Completo',
            'category': 'levantamiento_reserva'
        },
        'pautas_generales': {
            'ids': ['pautas_plan_funcionamiento', 'pautas_reprogramacion', 'pautas_informacion_publicidad',
                   'pautas_cupos_grupo', 'pautas_proceso_matricula', 'pautas_modificacion_matricula',
                   'pautas_talleres_extracurriculares', 'pautas_inicio_clases'],
            'title': 'Pautas Generales de Matrícula - Información Completa',
            'category': 'pautas'
        }
    }
    
    # Crear diccionario de documentos por ID
    docs_by_id = {doc['id']: doc for doc in documents}
    
    # Crear documentos combinados
    combined_docs = []
    used_ids = set()
    
    for group_name, group_info in groups.items():
        group_docs = []
        for doc_id in group_info['ids']:
            if doc_id in docs_by_id:
                group_docs.append(docs_by_id[doc_id])
                used_ids.add(doc_id)
        
        if group_docs:
            # Combinar contenido
            combined_content = []
            for doc in group_docs:
                section_title = doc['title']
                section_content = doc['content']
                combined_content.append(f"**{section_title}**\n{section_content}")
            
            # Crear documento combinado
            combined_doc = {
                'id': group_name,
                'title': group_info['title'],
                'content': '\n\n'.join(combined_content),
                'category': group_info['category'],
                'subcategory': 'completo',
                'keywords': extract_keywords('\n\n'.join(combined_content)),
                'year': '2025',
                'document_type': 'procedimiento_completo',
                'original_docs': [doc['id'] for doc in group_docs],
                'sections_count': len(group_docs)
            }
            combined_docs.append(combined_doc)
    
    # Agregar documentos individuales no agrupados
    for doc in documents:
        if doc['id'] not in used_ids:
            enhanced_doc = {
                **doc,
                'category': doc['id'].split('_')[0],
                'subcategory': 'individual',
                'keywords': extract_keywords(doc['content']),
                'year': '2025',
                'document_type': 'procedimiento_individual',
                'original_docs': [doc['id']],
                'sections_count': 1
            }
            combined_docs.append(enhanced_doc)
    
    return combined_docs

def improve_dataset(input_file, output_file):
    """Función principal para mejorar el dataset"""
    
    print("🚀 Iniciando mejora del dataset...")
    
    # Cargar datos originales
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"📊 Documentos originales: {len(original_data)}")
    
    # Agrupar documentos relacionados
    improved_data = group_related_documents(original_data)
    
    # Guardar dataset mejorado
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(improved_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset mejorado guardado en: {output_file}")
    print(f"📊 Documentos mejorados: {len(improved_data)}")
    
    # Mostrar estadísticas
    print("\n📈 Estadísticas de mejora:")
    categories = {}
    for doc in improved_data:
        category = doc['category']
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    for category, count in categories.items():
        print(f"  - {category}: {count} documentos")
    
    # Mostrar algunos ejemplos
    print("\n🔍 Ejemplos de documentos mejorados:")
    for i, doc in enumerate(improved_data[:3]):
        print(f"\n{i+1}. ID: {doc['id']}")
        print(f"   Título: {doc['title']}")
        print(f"   Categoría: {doc['category']}")
        print(f"   Keywords: {doc['keywords']}")
        print(f"   Documentos originales: {len(doc['original_docs'])}")
        print(f"   Contenido: {len(doc['content'])} caracteres")

def main():
    """Función principal"""
    # Definir rutas
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    input_file = data_dir / 'dataset_rag_matriculas.json'
    output_file = data_dir / 'dataset_rag_matriculas_mejorado.json'
    backup_file = data_dir / 'dataset_rag_matriculas_backup.json'
    
    # Verificar que existe el archivo original
    if not input_file.exists():
        print(f"❌ Error: No se encontró el archivo {input_file}")
        return
    
    # Crear backup
    import shutil
    shutil.copy2(input_file, backup_file)
    print(f"💾 Backup creado en: {backup_file}")
    
    # Mejorar dataset
    improve_dataset(input_file, output_file)
    
    print("\n🎉 ¡Mejora completada exitosamente!")
    print(f"📁 Archivo original: {input_file}")
    print(f"📁 Archivo mejorado: {output_file}")
    print(f"📁 Backup: {backup_file}")

if __name__ == "__main__":
    main()