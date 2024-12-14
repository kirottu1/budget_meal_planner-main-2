import json
from queue import Queue
from threading import Thread

from confluent_kafka import Consumer, Producer

from logger import get_logger

recipes_logger = get_logger("recipes")


def process_flyer(path_to_flyer: str) -> None:
    producer_conf = {
        "bootstrap.servers": "localhost:19092",
        "partitioner": "random",
    }
    flyer_producer = Producer(producer_conf)
    flyer_producer.produce("flyer", key="flyer_submitted", value=json.dumps(path_to_flyer))
    flyer_producer.flush()
    recipes_logger.info(f"Published flyer_submitted message to the flyer topic for {path_to_flyer} file")


def check_flyer_processed(result_queue: Queue) -> None:
    recipes_logger.info("start check_flyer_processed")
    consumer_conf = {
        "bootstrap.servers": "localhost:19092",
        "group.id": "flyer_done",
        "auto.offset.reset": "earliest",
    }
    flyer_consumer = Consumer(consumer_conf)
    flyer_consumer.subscribe(["flyer"])
    while True:
        msg = flyer_consumer.poll(0.1)
        if msg is None or msg.error():
            pass
        else:
            byte_key = msg.key()
            task = byte_key.decode("utf-8")
            if task == "flyer_processed":
                ingredients = json.loads(msg.value())
                recipes_logger.info(f"\n Flyer processed with ingredients: {ingredients}")
                result_queue.put(("flyer_processed", ingredients))  # Place result in queue


def recommend_recipe(user_selected_ingredient: list[str]) -> None:
    producer_conf = {
        "bootstrap.servers": "localhost:19092",
        "partitioner": "random",
    }
    recipe_producer = Producer(producer_conf)
    recipe_producer.produce("recipe", key="recipe_requested", value=json.dumps(user_selected_ingredient))
    recipe_producer.flush()
    recipes_logger.info("Published recipe_requested message to the recipe topic")


def check_recipe_recommended(result_queue: Queue) -> None:
    recipes_logger.info("start check_recipe_recommended")
    consumer_conf = {
        "bootstrap.servers": "localhost:19092",
        "group.id": "recipe_done",
        "auto.offset.reset": "earliest",
    }
    recipe_consumer = Consumer(consumer_conf)
    recipe_consumer.subscribe(["recipe"])
    while True:
        msg = recipe_consumer.poll(0.1)
        if msg is None or msg.error():
            pass
        else:
            byte_key = msg.key()
            task = byte_key.decode("utf-8")
            if task == "recipe_ready":
                recipe = json.loads(msg.value())
                recipes_logger.info(f"\n Recommended recipe: {recipe}")
                result_queue.put(("recipe_ready", recipe))  # Place result in queue


def launch_consumer() -> Queue:
    result_queue = Queue()

    # Start consumer threads with access to the queue
    Thread(target=check_flyer_processed, args=(result_queue,), daemon=True).start()
    Thread(target=check_recipe_recommended, args=(result_queue,), daemon=True).start()
    return result_queue  # Return queue for accessing results


if __name__ == "__main__":
    try:
        # Initialize the result queue
        launch_consumer()

        while True:
            # Process flyer and recommend recipe as usual
            path_to_flyer = input("\nEnter flyer path: ")
            process_flyer(path_to_flyer)
            ingredient = input("Enter ingredient: ")
            user_selected_ingredient = [ingredient]
            recommend_recipe(user_selected_ingredient)
    except KeyboardInterrupt:
        pass
