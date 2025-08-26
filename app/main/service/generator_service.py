import traceback
from operator import itemgetter
from typing import Dict

from flask import current_app
from loguru import logger

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.retrievers.merger_retriever import MergerRetriever

from app.main.components.llm.llm_component import LLMComponent
from app.main.components.vector_store.vector_store_component import VectorStoreComponent
from app.main.util.utils import STATUS


class GeneratorService:
    def __init__(self) -> None:
        self.logger = logger
        self._template = """Utilisez le contexte et l'instruction suivant, pour répondre à la question à la fin.
Répondez en français et de manière claire et formelle.
Vos réponses ne doivent répondre qu'une seule fois à la question et ne pas comporter de texte après la réponse.
Si vous ne connaissez pas la réponse à une question, répondez simplement avec "Aucune réponse" au lieu d'inventer une réponse.

Instruction additionnelle: {instruction}

Contexte: {context}

Question: {question}

Réponse :"""
        self.custom_rag_prompt = PromptTemplate.from_template(self._template)
        self.logger.info("GeneratorService initialized")

    def answer_question(self, query_metadata: Dict):
        error = None
        status = STATUS.STARTED.value

        name = query_metadata.get("name", None)
        instruction = query_metadata.get("instruction", "")
        file_ids = query_metadata.get("file_ids", [])
        k = query_metadata.get("number_of_documents") or current_app.config.get("TOP_K_DEFAULT", 4)
        threshold = query_metadata.get("threshold") or current_app.config.get("SCORE_THRESHOLD", 0.0)

        try:
            status = STATUS.RUNNING.value

            self.logger.info(f"Query name: {name}")
            self.logger.info(f"Instruction: {instruction}")
            self.logger.info(f"File_ids: {file_ids}, top_k: {k}, threshold: {threshold}")

            llm = LLMComponent().get_llm()

            # Pas de fichier : pas de contexte
            if not file_ids:
                rag_chain = (
                    {
                        "instruction": itemgetter("instruction"),
                        "context": RunnableLambda(self.no_context),
                        "question": itemgetter("question"),
                    }
                    | self.custom_rag_prompt
                    | llm
                    | StrOutputParser()
                )
                response = rag_chain.invoke({"instruction": instruction, "question": name})

            # Un seul fichier
            elif len(file_ids) == 1:
                vector_store_component = VectorStoreComponent(LLMComponent())
                retriever = vector_store_component.get_retriever(file_ids[0], k, threshold)

                rag_chain = (
                    {
                        "instruction": itemgetter("instruction"),
                        "context": itemgetter("question") | retriever | self.format_docs,
                        "question": itemgetter("question"),
                    }
                    | self.custom_rag_prompt
                    | llm
                    | StrOutputParser()
                )
                response = rag_chain.invoke({"instruction": instruction, "question": name})

            # Plusieurs fichiers
            else:
                retrievers = []
                for file_id in file_ids:
                    self.logger.info(f"Getting retriever for: {file_id}")
                    vector_store_component = VectorStoreComponent(LLMComponent())
                    retriever = vector_store_component.get_retriever(file_id, k, threshold)
                    retrievers.append(retriever)

                merged = MergerRetriever(retrievers=retrievers)

                rag_chain = (
                    {
                        "instruction": itemgetter("instruction"),
                        "context": itemgetter("question") | merged | self.format_docs,
                        "question": itemgetter("question"),
                    }
                    | self.custom_rag_prompt
                    | llm
                    | StrOutputParser()
                )
                response = rag_chain.invoke({"instruction": instruction, "question": name})

            status = STATUS.DONE.value
            return {"generated_content": response, "sources": []}

        except Exception as e:
            status = STATUS.ERROR.value
            error = str(e)
            self.logger.error(f"Exception in GeneratorService.answer_question: {error}")
            traceback.print_exc()
            return {"status": status, "error": error}, 500

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def no_context(self, params):
        return ""
