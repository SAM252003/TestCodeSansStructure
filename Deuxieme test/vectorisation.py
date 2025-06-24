# rag_full.py
import os, re, pathlib, json
import numpy as np
import faiss
import ollama                                       # FAISS + Ollama = mêmes libs que ton code
import pandas as pd

# =============== 0. PARAMÈTRES GLOBAUX ===============
LANGUAGE_MODEL  = "llama3.2:1b-instruct-fp16"
EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
INDEX_FILE      = "faiss_index.bin"
CHUNKS_FILE     = "chunks.json"
TOP_K_DEFAULT   = 3

# =========================================
# 1. CHARGER & PARSER LE LIVRE D’OR
# -----------------------------------------
with open("livredor_large_verbose.txt", encoding="utf-8") as file:
    lines = [l.strip() for l in file if l.strip()]

entries, current = [], {}
for line in lines:
    if line.startswith("Problème"):
        current = {"problem": line}
    elif line.startswith("Solution"):
        current["solution"] = line
    elif line.startswith("Mots clés"):
        current["keywords"] = line
        entries.append(current)
        current = {}

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 2. CONSTRUCTION FAISS
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def build_faiss(entries):
    first_chunk = "\n".join(entries[0].values())
    dim = len(ollama.embed(model=EMBEDDING_MODEL, input=first_chunk)["embeddings"][0])
    index = faiss.IndexFlatIP(dim)

    def ajout(txt):
        emb = np.array(
            ollama.embed(model=EMBEDDING_MODEL, input=txt)["embeddings"][0],
            dtype=np.float32,
        ).reshape(1, -1)
        faiss.normalize_L2(emb)
        index.add(emb)

    ajout(first_chunk)
    for e in entries[1:]:
        ajout("\n".join(e.values()))
    return index

if pathlib.Path(INDEX_FILE).exists():
    faiss_index = faiss.read_index(INDEX_FILE)
    all_chunks  = json.load(open(CHUNKS_FILE, encoding="utf-8"))
else:
    faiss_index = build_faiss(entries)
    all_chunks  = ["\n".join(e.values()) for e in entries]
    faiss.write_index(faiss_index, INDEX_FILE)
    json.dump(all_chunks, open(CHUNKS_FILE, "w", encoding="utf-8"))

# Même nom & même signature que ton code
def retrieve_solution_faiss(query: str, top_k: int = TOP_K_DEFAULT):
    # Générer l'embedding de la requête
    q_emb = np.array(
        ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0],
        dtype=np.float32,
    ).reshape(1, -1)

    faiss.normalize_L2(q_emb)
    dists, idxs = faiss_index.search(q_emb, top_k)

    results = []
    for idx, score in zip(idxs[0], dists[0]):
        if score < 0.75:
            continue  # Ignorer les correspondances trop faibles
        blk = entries[idx]
        results.append(
            {"problem": blk["problem"], "solution": blk["solution"], "score": float(score)}
        )

    return results


