import json
import faiss
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import requests

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------
GROQ_API_KEY = ""
GROQ_MODEL = "llama3-8b-8192"  # puedes cambiar a mixtral-8x7b si deseas

# -----------------------------
# CARGAR Y EMBEDDEAR DOCUMENTOS
# -----------------------------

def cargar_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generar_embeddings(textos, modelo_emb):
    return modelo_emb.encode(textos, convert_to_numpy=True, show_progress_bar=True)

# -------------------------
# CREAR ÍNDICE FAISS NUEVO
# -------------------------

def crear_indice(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "index.faiss")
    print(f"Índice creado con {len(embeddings)} documentos.")
    return index

# -------------------------
# GENERAR RESPUESTA USANDO GROQ
# -------------------------

def responder_groq(prompt, max_tokens=256):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "Responde de manera clara, formal y solo usando el contexto proporcionado."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# -------------------------
# CONSULTAR AL SISTEMA
# -------------------------

def consultar(query, docs, embeddings, modelo_emb, top_k=3):
    vec_query = modelo_emb.encode([query], convert_to_numpy=True)
    index = faiss.read_index("index.faiss")
    _, indices = index.search(vec_query, top_k)

    contexto = "\n---\n".join([docs[i]["content"] for i in indices[0]])
    prompt = (
        "Contesta la siguiente pregunta con base únicamente en el contexto proporcionado.\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {query}\n\n"
        "Respuesta:"
    )
    respuesta = responder_groq(prompt)
    return respuesta

# -------------------------
# MAIN: INTERFAZ DE TERMINAL
# -------------------------

def main():
    print("Cargando dataset...")
    dataset = cargar_dataset("base_normativa_rag.json")
    textos = [doc["content"] for doc in dataset]

    modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    if not Path("index.faiss").exists():
        print("Generando embeddings...")
        embeddings = generar_embeddings(textos, modelo_embeddings)
        crear_indice(embeddings)
    else:
        print("Índice ya existe, cargando FAISS...")
        embeddings = generar_embeddings(textos, modelo_embeddings)

    print("\n🤖 Sistema RAG con Groq API activo. Escribe tu consulta (o 'salir'):\n")
    while True:
        query = input("Tú: ")
        if query.lower() in {"salir", "exit"}:
            break
        respuesta = consultar(query, dataset, embeddings, modelo_embeddings)
        print("\n📘 Respuesta:", respuesta)
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()