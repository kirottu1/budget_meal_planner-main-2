import asyncio
import json
from configparser import ConfigParser
from pathlib import Path

from confluent_kafka import Consumer, Producer

from common import TaskType
from config import get_config
from logger import get_logger
from model_factory import ModelFactory
from prompt_manager import PromptManager
from recipe_service.llm_rag_model import LLMRAG
from recipe_service.recommend_recipes import recommend_recipes
from recipe_service.select_products import generate_random_products_selection
from recipe_service.vector_database import create_vector_database

recipes_logger = get_logger("recipes")

config_parser = ConfigParser(interpolation=None)

config_file = (Path(__file__).parent / "config.properties").open("r")
config_parser.read_file(config_file)
client_config = dict(config_parser["kafka_client"])

recipe_consumer = Consumer(client_config)
recipe_consumer.subscribe(["recipe"])

recipe_producer = Producer(client_config)
config = get_config()


def start_service() -> None:
    while True:
        msg = recipe_consumer.poll(0.1)
        if msg is None or msg.error():
            pass
        else:
            byte_key = msg.key()
            task = byte_key.decode("utf-8")
            recipes_logger.info(task)
            if task == "recipe_requested":
                ingredients_list = json.loads(msg.value())
                # TODO: Kafka
                # run this function in another process
                asyncio.run(recommend_recipe(ingredients_list))


async def recommend_recipe(ingredients_list: list[str]):
    prompt_manager = PromptManager(config)
    model_factory = ModelFactory(config)
    execute_recipe_recommendation = False
    if execute_recipe_recommendation:
        ingredients_list = generate_random_products_selection(config.output_path.products_path)
        await recommend_recipes(
            ingredients_list=ingredients_list,
            output_path=config.output_path.recipes_path,
            model=LLMRAG(
                model=model_factory.get_model(TaskType.RECOMMEND_RECIPES),
                prompt_template=prompt_manager.get_prompt(TaskType.RECOMMEND_RECIPES),
                vectors=create_vector_database(model_factory.get_model(TaskType.EMBEDDING)),
            ),
        )
    # TODO: Kafka
    # read the recipe files add to topics  instead of using hard coded   recipe
    recipe = {
        "recipe1": f"""Find a recipe that includes {ingredients_list} as ingredients.
                    **Recipe 1: Skillet-Braised Chicken**

                    * **name**: Skillet-Braised Chicken
                    * **preparation_time**: 5 minutes
                    * **directions**:
                      1. Season the chicken. Sauté it for 1 minute per side in a lightly oiled skillet over medium-high
                         heat until lightly browned.
                      2. Cover the skillet with a tight-fitting lid. Reduce the heat to low. Cook for 10 minutes.
                         Do not lift the lid.
                      3. Turn off the heat. Let the chicken rest for 10 minutes. Do not remove the lid.
                      4. Check if the chicken is cooked all the way through.
                    * **ingredients**:
                      - 1 chicken breast
                      - 1 Tablespoon oil
                      - Seasoning— such as salt, pepper, season salt, onion powder or garlic powder, as desired
                    * **calories**: Not provided in the context
                    * **total fat (PDV)**: Not provided in the context
                    * **sugar (PDV)**: Not provided in the context
                    * **sodium (PDV)**: Not provided in the context
                    * **protein (PDV)**: Not provided in the context
                    * **saturated fat (PDV)**: Not provided in the context
                    * **carbohydrates (PDV)**: Not provided in the context""",
    }
    recipe_producer.produce("recipe", key="recipe_ready", value=json.dumps(recipe))
    recipe_producer.flush()
    recipes_logger.info("publish recipe_ready message to the recipe topic")


if __name__ == "__main__":
    start_service()
    """
    start the graph
    Tool: call vector database
    LLM: Call LLm to ger recipe
    end
    """
