
# tools.py
from langchain_core.tools import tool
from vectorisation import retrieve_solution_faiss


@tool
def retrieve_it(query: str, top_k: int = 4, min_score: float = 0.75):
    """
    TOOL − Cherche des solutions IT pertinentes dans le livre d'or.
    Retourne une liste de dicts JSON-compatibles.
    """
    results = retrieve_solution_faiss(query, top_k=top_k)
    return [r for r in results if r["score"] >= min_score]

@tool
def explain(text: str):
    """
    Reformule de manière plus pédagogique un passage donné.
    """
    return text