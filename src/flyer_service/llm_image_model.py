import base64
from pathlib import Path

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from llm_model import LLMModel
from logger import get_logger

recipes_logger = get_logger("recipes")


# Define your desired data structure.
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import Optional

class ProductDetails(BaseModel):
    name_or_brand: Optional[str] = Field(None, description="The product name or brand logo.")
    price: Optional[str] = Field(None, description="The price of the product.")
    promotions: Optional[str] = Field(None, description="Any promotions, discounts, or special offers associated with the product.")
    category: str = Field(..., description="Product category, e.g., 'Chicken', 'Fish', 'Pork', 'Beef', 'Vegetable', 'Fruit', or 'Others'.")
    language: Optional[str] = Field(None, description="Language of product details (English/French).")

class ProductList(BaseModel):
    products: list[ProductDetails]

# Set up a parser + inject instructions into the prompt template.
output_parser = PydanticOutputParser(pydantic_object=ProductList)

class LLMImage(LLMModel):
    """Handles image analysis tasks like text extraction."""

    def __init__(self, model: BaseChatModel, prompt: str):
        super().__init__(model)
        self.prompt = prompt
        format_instructions = output_parser.get_format_instructions()
        self.prompt_template = self.prompt + "\n"
        self.prompt_template +=  f"""Provide your response in the following format:
                                {format_instructions}"""


        self.prompt_template  = """
        Act as an advanced OCR system and extract all visible text from the provided image. The image contains both English and French text, so ensure that both languages are captured accurately. If the text appears in distinct sections, paragraphs, or headings, maintain these divisions in your response. Return the text exactly as it appears in the image, preserving formatting like line breaks and special characters. Focus on capturing the text content without any commentary or interpretation. Thank you!
        """
        self.chain = self.model# | output_parser

    async def runtask(self, image_path: str) -> str:
        """Run Image task: Extract text from an image."""
        image_data = self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt_template},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"},},
            ],
        )

        try:
            response = await self.chain.ainvoke([message])
            return response.content
            #return response.model_dump_json(indent=4)

        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
        except ValueError as ve:
            self._handle_value_error(ve)

    def _encode_image(self, image_path: str) -> str:
        with Path(image_path).open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
