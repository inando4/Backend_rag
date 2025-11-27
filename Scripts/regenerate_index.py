import json
import os
import faiss
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

def regenerate_faiss_index():
    """Regenerar índice FAISS después de modificar el dataset"""
    
    print("🚀 Regenerando índice FAISS...")
    
    # Rutas
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    json_path = data_dir / 'dataset_v2.json'
    index_path = data_dir / 'index.faiss'
    backup_index_path = data_dir / 'index_backup.faiss'
    
    # Verificar que existe el dataset
    if not json_path.exists():
        print(f"❌ Error: No se encontró {json_path}")
        return
    
    # Backup del índice actual
    if index_path.exists():
        import shutil
        shutil.copy2(index_path, backup_index_path)
        print(f"💾 Backup del índice creado: {backup_index_path}")
    
    # Cargar documentos
    print("📖 Cargando documentos...")
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print(f"📊 Total de documentos: {len(documents)}")
    
    # Verificar documentos de tasas
    print("\n💰 Verificando documentos de tasas...")
    tasas_docs = []
    for i, doc in enumerate(documents):
        if doc.get('tasa_soles'):
            tasas_docs.append({
                'index': i,
                'id': doc.get('id_chunk'),
                'tasa': doc.get('tasa_soles'),
                'modalidad': doc.get('modalidad_pago_relacionada', 'N/A')
            })
    
    if len(tasas_docs) == 0:
        print("❌ ERROR: No se encontraron documentos con tasas")
        return
    
    print(f"✅ Encontrados {len(tasas_docs)} documentos con tasas:")
    for doc in tasas_docs:
        icon = "⭐" if doc['id'] == 'CONV-018-PAGO-PROF-EXT' else "✓"
        print(f"   {icon} [{doc['index']}] {doc['id']}: S/ {doc['tasa']:.2f}")
        print(f"       {doc['modalidad']}")
    
    # Inicializar modelo de embeddings (MISMO que usa rag_service.py)
    print("\n🤖 Inicializando modelo de embeddings...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Crear embeddings
    print("🔄 Generando embeddings...")
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Crear índice FAISS
    print("\n📦 Creando índice FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
    
    # Normalizar embeddings para similitud coseno
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Guardar índice
    faiss.write_index(index, str(index_path))
    
    print(f"\n✅ Índice FAISS regenerado exitosamente")
    print(f"   - Documentos indexados: {index.ntotal}")
    print(f"   - Dimensión: {dimension}")
    print(f"   - Archivo: {index_path}")
    
    # Prueba de búsqueda para CONV-018
    print("\n🔍 Probando búsqueda de 'profesional universidad particular'...")
    query = "¿Cuánto cuesta convalidar un curso si soy de modalidad Profesional y el curso viene de una Universidad Particular?"
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, 15)
    
    print("\n📊 Top 15 resultados por similitud semántica:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        doc = documents[idx]
        marker = "🎯" if doc.get('id_chunk') == 'CONV-018-PAGO-PROF-EXT' else "  "
        tasa_info = f" - S/ {doc.get('tasa_soles'):.2f}" if doc.get('tasa_soles') else ""
        print(f"{marker} {rank}. [{idx}] Score: {score:.4f} - {doc.get('id_chunk', 'N/A')}{tasa_info}")
    
    # Verificar que CONV-018 está en los resultados
    conv_018_found = False
    conv_018_rank = None
    for rank, idx in enumerate(indices[0], 1):
        if documents[idx].get('id_chunk') == 'CONV-018-PAGO-PROF-EXT':
            conv_018_found = True
            conv_018_rank = rank
            break
    
    if conv_018_found:
        print(f"\n✅ CONV-018-PAGO-PROF-EXT encontrado en posición {conv_018_rank}")
    else:
        print("\n⚠️ CONV-018-PAGO-PROF-EXT NO está en el top 15 por similitud semántica")
        print("   Esto es normal - el sistema keyword search lo priorizará")

if __name__ == "__main__":
    regenerate_faiss_index()
    print("\n🎉 ¡Regeneración completada!")