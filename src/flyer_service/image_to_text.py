import asyncio
from pathlib import Path

from tqdm.asyncio import tqdm

from config import Config
from llm_model import LLMModel
from logger import get_logger
from utils import get_name_from_path, list_files_in_folder
import cv2

recipes_logger = get_logger("recipes")

async def extract_text(config: Config, model: LLMModel):
    image_files_path = list_files_in_folder(config.output_path.images_path, "*.png")
    tasks = []
    for image_path in tqdm(image_files_path, desc="Extracting product from flyer pages"):
        recipes_logger.info(f"Extracting product from {get_name_from_path(image_path)}")
        out_put_path = f"{config.output_path.products_path}/{get_name_from_path(image_path)}"
        tasks.append(_process_image_for_extraction(image_path=image_path, out_put_path=out_put_path, model=model))
    await asyncio.gather(*tasks)


def _split_image_with_overlap_simple(image, window_size=(500, 500), overlap=250):
    height, width = image.shape[:2]
    window_w, window_h = window_size
    step_x = window_w - overlap
    step_y = window_h - overlap

    # Calculate remainders to see how much padding is needed
    remainder_x = width % step_x
    remainder_y = height % step_y

    # Calculate padding amounts
    pad_right = (step_x - remainder_x) if remainder_x > 0 else 0
    pad_bottom = (step_y - remainder_y) if remainder_y > 0 else 0

    # Apply padding
    padded_image = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    child_images = []
    # Loop through the image with the specified step size
    for y in range(0, height - window_h + 1, step_y):
        for x in range(0, width - window_w + 1, step_x):
            # Extract the window and add it to the list
            child_image = image[y:y + window_h, x:x + window_w]
            child_images.append(child_image)
    return child_images

def _is_mostly_white_or_black(image, threshold=0.95, color='white'):
    if color == 'white':
        # Set target pixel value for white (255) and tolerance
        target_value = 255
        tolerance = 10  # Adjust as needed for brightness variations
    elif color == 'black':
        # Set target pixel value for black (0) and tolerance
        target_value = 0
        tolerance = 10  # Adjust as needed for darkness variations
    else:
        raise ValueError("Color should be 'white' or 'black'")

    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Create a binary mask for pixels within the target value ± tolerance
    if color == 'white':
        mask = cv2.inRange(gray_image, target_value - tolerance, target_value)
    else:
        mask = cv2.inRange(gray_image, target_value, target_value + tolerance)

    # Calculate the percentage of target pixels
    target_pixel_ratio = np.sum(mask > 0) / mask.size

    return target_pixel_ratio >= threshold

async def _process_image_for_extraction(*, image_path: str, out_put_path: str, model: LLMModel) -> None:
    response_data = await model.runtask(image_path)
    recipes_logger.info(response_data)
    #TODO: produce a message to flyer topic : product_extraction_page_X_completed
    if response_data:
        with Path(f"{out_put_path}.txt").open("w") as file:
            file.write(response_data)