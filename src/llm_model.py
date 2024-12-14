import requests
from langchain_core.language_models.chat_models import BaseChatModel

from logger import get_logger

recipes_logger = get_logger("recipes")


class LLMModel:
    """Base class for LLM models."""

    def __init__(self, model: BaseChatModel):
        self.model = model

    def _handle_request_error(self, e: requests.exceptions.RequestException) -> None:
        """Handle request exceptions."""
        recipes_logger.info(f"API Request failed: {e}")

    def _handle_value_error(self, e: ValueError) -> None:
        """Handle value errors."""
        recipes_logger.info(f"Value error: {e}")

    async def runtask(self, param: str):
        raise NotImplementedError
