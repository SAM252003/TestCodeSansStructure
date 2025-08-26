# main.py  – charge .env, expose constantes, et contient rag_response()
from pathlib import Path
import os
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TOP_K_DEFAULT   = int(os.getenv("TOP_K_DEFAULT", 4))
CHROMA_PATH     = os.getenv("CHROMA_PATH", "db")
# DEBUG prints

#importer RAG et retrieval
from retriever import retrieve_chunks
from RAG import rag_chain

# ── 3) fonction unique : rag_response(payload_json) ───────────────
def rag_response(payload: dict,
                 session_id: str = "id_default",
                 top_k: int = TOP_K_DEFAULT) -> str:

    #Cas où l'étape amont veut une précision
    if payload.get("ask_user"):
        return payload["ask_user"]

    #On Construit la requête a partir du JSON
    parts = [payload.get("problem"), payload.get("summary"), payload.get("application")]
    query = " ".join([p for p in parts if p])

    # On recupere les extraits pertinents
    retrieved = retrieve_chunks(query, top_k=top_k)
    context   = "\n\n".join(f"[{r['section']}] {r['text']}" for r in retrieved)

    # 3) Appel du chain LLM existant
    answer = rag_chain.invoke(
        {"query": query, "CONTEXT": context},
        config={"session_id": session_id}
    )

    return answer.content if hasattr(answer, "content") else str(answer)


# ── 4) boucle console (test manuel)
if __name__ == "__main__":
    import json
    print("Copie/colle un JSON en entrée (ou 'quit') :")
    while True:
        raw = input("\n payload > ")
        if raw.strip().lower() in {"quitter", "fin","quit","exit","merci","enrevoir","aurevoir","au revoir"}:
            break
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            print(" JSON invalide :", e)
            continue
        print("\n", rag_response(payload, session_id="console"))
