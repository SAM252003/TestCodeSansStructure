from vectorisation import retrieve_solution_faiss
from RAG  import *

def rag_response(user_query: str, session_id="id_default", top_k=3, min_score: float = 0.6):
    retrieved = retrieve_solution_faiss(user_query, top_k)
    retrieved = [r for r in retrieved if r["score"] >= min_score]

    context = "\n\n".join(
        f"[score={r['score']:.3f}] {r['problem']}\n{r['solution']}" for r in retrieved
    ) if retrieved else ""

    try:
        response = rag_chain.invoke(
            {"query": user_query, "CONTEXT": context},
            config={"session_id": session_id, "llm": llm, "k": 4},
        )
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        return f"[Erreur] RAG a échoué : {e}"

# =========================================
# 5. TEST RAPIDE (même style que ton main)
# -----------------------------------------
if __name__ == "__main__":
    while True:
        q = input("\n💬 Utilisateur > ")
        if q.lower() in {"quitter", "Merci"}:
            break
        answer = rag_response(q, session_id="id_123", top_k=5)
        if isinstance(answer, str):
            print("🤖 Agent :", answer)
        else:
            print("🤖 Agent :", answer.content)
