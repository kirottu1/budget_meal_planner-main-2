import secrets
from pathlib import Path

from logger import get_logger
from utils import list_files_in_folder
import json

recipes_logger = get_logger("recipes")


def generate_random_products_selection(products_path: Path) -> list[str]:
    products = _parse_flyer_products(_load_flyer_contents(products_path))
    recipes_logger.info(products)
    selected_products = []
    for category, category_info in products.items():
        if len(category_info):
            random_index = secrets.randbelow(len(category_info))
            selected_products.append(category_info[random_index]["Category"])
            recipes_logger.info(f"Selected product from {category}: {category_info[random_index]}")
    return selected_products


def _load_flyer_contents(products_path: Path) -> list[dict]:
    flyer_contents = []
    for file_path in list_files_in_folder(products_path, "*.json"):
        with file_path.open("r") as file:
            flyer_contents.append(json.load(file))
    return flyer_contents

def _parse_flyer_products(flyer_contents: list[dict]) -> dict[str : list[dict]]:
    product_info = {
        "chicken": [],
        "beef": [],
        "pork": [],
        "fish": [],
        "vegetable": [],
        "fruit":[]
    }
    for page_product_info in flyer_contents:
        products = page_product_info["products"]
        for product in products:
            name = product['name_or_brand'].strip().lower()
            price = product['price']
            promo = product['promotions']
            category = product['category'].lower()
            if category in product_info:
                product_info[category].append(
                    {
                        "Name": name,
                        "Price": price,
                        "Promotions": promo,
                        "Category": category,
                    })

    return product_info