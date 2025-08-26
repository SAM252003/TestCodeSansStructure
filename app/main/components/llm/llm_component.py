import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from flask import current_app as app

# LangChain OpenAI chat model
from langchain_community.chat_models import ChatOpenAI

# Charge .env si pas déjà fait en amont
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

class LLMComponent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMComponent, cls).__new__(cls)
            cls._instance.logger = logger
            cls._instance._initialize_llm()
        return cls._instance

    def _initialize_llm(self):
        model_name = os.getenv("LANGUAGE_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo"))
        temperature = float(os.getenv("LLM_TEMPERATURE", 0.2))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", 512))

        self.logger.info(f"Initializing OpenAI chat model {model_name} with temp={temperature}")

        # Utilise LangChain wrapper pour OpenAI
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_llm(self):
        return self.llm
