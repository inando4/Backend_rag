import json
from pathlib import Path

def verify_dataset():
    """Verificar que el dataset tenga todos los documentos de tasas"""
    
    base_dir = Path(__file__).parent.parent
    json_path = base_dir / 'data' / 'dataset_v2.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    print("🔍 Verificando documentos de tasas...")
    print(f"📊 Total documentos: {len(documents)}\n")
    
    # Documentos de tasas esperados
    expected_tasas = {
        'CONV-014-PAGO-UNSA': 35.00,
        'CONV-015-PAGO-UNIV-NAC': 55.00,
        'CONV-016-PAGO-UNIV-PART': 105.00,
        'CONV-017-PAGO-PROF-UNSA': 55.00,
        'CONV-018-PAGO-PROF-EXT': 176.00,  # ← El que falta
    }
    
    found_tasas = {}
    for doc in documents:
        if doc.get('id_chunk') in expected_tasas:
            found_tasas[doc['id_chunk']] = doc.get('tasa_soles')
    
    print("✅ Documentos encontrados:")
    for id_chunk, tasa in expected_tasas.items():
        if id_chunk in found_tasas:
            status = "✓" if found_tasas[id_chunk] == tasa else f"⚠️ (esperado: {tasa}, encontrado: {found_tasas[id_chunk]})"
            print(f"   {status} {id_chunk}: S/ {found_tasas.get(id_chunk, 0):.2f}")
        else:
            print(f"   ❌ {id_chunk}: NO ENCONTRADO")
    
    # Verificar modalidades
    print("\n📝 Modalidades:")
    for doc in documents:
        if doc.get('id_chunk') in expected_tasas:
            modalidad = doc.get('modalidad_pago_relacionada', 'N/A')
            print(f"   {doc['id_chunk']}: {modalidad}")
    
    missing = set(expected_tasas.keys()) - set(found_tasas.keys())
    if missing:
        print(f"\n⚠️ DOCUMENTOS FALTANTES: {missing}")
        return False
    else:
        print("\n🎉 ¡Todos los documentos de tasas están presentes!")
        return True

if __name__ == "__main__":
    verify_dataset()