import asyncio
import json
from configparser import ConfigParser
from pathlib import Path

from confluent_kafka import Consumer, Producer

from common import TaskType
from config import get_config
from flyer_service.image_to_text import extract_text
from flyer_service.llm_image_model import LLMImage
from flyer_service.pdf_to_image import convert_pdf_to_images
from logger import get_logger
from model_factory import ModelFactory
from prompt_manager import PromptManager

recipes_logger = get_logger("recipes")

config_parser = ConfigParser(interpolation=None)
config_file = (Path(__file__).parent / "config.properties").open("r")
config_parser.read_file(config_file)
client_config = dict(config_parser["kafka_client"])

flyer_consumer = Consumer(client_config)
flyer_consumer.subscribe(["flyer"])

flyer_producer = Producer(client_config)
config = get_config()


def start_service() -> None:
    while True:
        msg = flyer_consumer.poll(0.1)
        if msg is None or msg.error():
            pass
        else:
            byte_key = msg.key()
            task = byte_key.decode("utf-8")
            recipes_logger.info(task)
            if task == "flyer_submitted":
                flyer_info = json.loads(msg.value())
                # TODO: Kafka
                # run this function in another process
                asyncio.run(extract_products(flyer_info))


async def extract_products(flyer_info: str) -> None:
    extract_images = True
    extract_products = False
    prompt_manager = PromptManager(config)
    model_factory = ModelFactory(config)
    if extract_images:
        await convert_pdf_to_images(
            pdf_path=config.data_path.pdf_path,
            output_folder=config.output_path.images_path,
        )
    if extract_products:
        await extract_text(
            config,
            model=LLMImage(
                model=model_factory.get_model(TaskType.EXTRACT_PRODUCT),
                prompt=prompt_manager.get_prompt(TaskType.EXTRACT_PRODUCT),
            ),
        )
    # TODO: Kafka
    # read the files add to topics  instead of using hard coded   ingredients_list
    ingredients_list = {"ingredients": ["chicken"]}
    flyer_producer.produce("flyer", key="flyer_processed", value=json.dumps(ingredients_list))
    flyer_producer.flush()
    recipes_logger.info(f"publish flyer_processed message to the flyer topic for {flyer_info} file")


if __name__ == "__main__":
    start_service()
    """
    start the graph
    Tool: Call pdf extractor to extract the images
    LLM: Call Multimodal LLM to extract the products from image
    end
    """
