import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from llm_model import LLMModel
from logger import get_logger

recipes_logger = get_logger("recipes")


class LLMRAG(LLMModel):
    """Handles RAG tasks like recipe recommendations."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt_template: str,
        vectors: VectorStore,
    ):
        super().__init__(model)
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.retriever = vectors.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        self.retrieval_chain  = (
                RunnableParallel({"context": self.retriever , "input": RunnablePassthrough()})
                | self.prompt
                | self.model
                | StrOutputParser()
        )

    async def runtask(self, user_message: str) -> str:
        try:
            response = await self.retrieval_chain.ainvoke( user_message)
            #recipe_text = response.get("answer", "No recipe found.")
            recipes_logger.info(f"Recipe found:\n{response}")
            return response

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
        except ValueError as ve:
            self._handle_value_error(ve)
