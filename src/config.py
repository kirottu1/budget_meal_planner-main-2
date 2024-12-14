from functools import cache
from pathlib import Path

import yaml
from pydantic import ValidationError
from pydantic.dataclasses import Field, dataclass

from common import TaskType
from logger import get_logger

recipes_logger = get_logger("recipes")


def validate_all_paths(paths: list[Path]) -> None:
    invalid_paths = list(filter(lambda path: not path.exists(), paths))

    if invalid_paths:
        invalid_paths_str = [str(path) for path in invalid_paths]
        raise ValueError(f"Paths do not exist: {', '.join(invalid_paths_str)}")


@dataclass(frozen=True)
class BaseModelConfig:
    model_name: str
    # TODO: add class for provider with all its information
    provider: str


@dataclass(frozen=True)
class ExtractProductConfig(BaseModelConfig):
    prompt_file: str
    temperature: float
    endpoint_name: str | None = Field(default=None)
    service_name: str | None = Field(default=None)

    @property
    def prompt_file_path(self) -> Path:
        return _get_project_root() / self.prompt_file

    def __post_init__(self):
        validate_all_paths([self.prompt_file_path])


@dataclass(frozen=True)
class EmbeddingModelConfig(BaseModelConfig):
    vector_index: str
    endpoint_name: str | None = Field(default=None)
    service_name: str | None = Field(default=None)

    @property
    def vector_index_path(self) -> Path:
        return _get_project_root() / self.vector_index


@dataclass(frozen=True)
class RecommendRecipesConfig(BaseModelConfig):
    prompt_file: str
    temperature: float
    endpoint_name: str | None = Field(default=None)
    service_name: str | None = Field(default=None)

    @property
    def prompt_file_path(self) -> Path:
        return _get_project_root() / self.prompt_file

    def __post_init__(self):
        validate_all_paths([self.prompt_file_path])


@dataclass(frozen=True)
class ModelConfig:
    extract_product: ExtractProductConfig
    embedding: EmbeddingModelConfig
    recommend_recipes: RecommendRecipesConfig


@dataclass(frozen=True)
class DataConfig:
    pdf: str
    recipe_books: str

    @property
    def pdf_path(self) -> Path:
        return _get_project_root() / self.pdf

    @property
    def recipe_books_path(self) -> Path:
        return _get_project_root() / self.recipe_books

    def __post_init__(self):
        validate_all_paths([self.pdf_path, self.recipe_books_path])


@dataclass(frozen=True)
class OutputConfig:
    images: str
    products: str
    recipes: str

    @property
    def images_path(self) -> Path:
        return _get_project_root() / self.images

    @property
    def products_path(self) -> Path:
        return _get_project_root() / self.products

    @property
    def recipes_path(self) -> Path:
        return _get_project_root() / self.recipes

    def __post_init__(self):
        for path in [self.images_path, self.products_path, self.recipes_path]:
            output_path = Path(path)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    model_configs: ModelConfig
    data_path: DataConfig
    output_path: OutputConfig

    def get_model_configs(self, task_type: TaskType) -> BaseModelConfig:
        if task_type == TaskType.EXTRACT_PRODUCT:
            return self.model_configs.extract_product
        if task_type == TaskType.EMBEDDING:
            return self.model_configs.embedding
        if task_type == TaskType.RECOMMEND_RECIPES:
            return self.model_configs.recommend_recipes
        return None

    def get_prompt_file_path(self, task_type: TaskType) -> Path | None:
        if task_type == TaskType.EXTRACT_PRODUCT:
            return self.model_configs.extract_product.prompt_file_path
        if task_type == TaskType.EMBEDDING:
            return None
        if task_type == TaskType.RECOMMEND_RECIPES:
            return self.model_configs.recommend_recipes.prompt_file_path
        return None


def _get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()


@cache
def get_config() -> Config:
    config_path = _get_project_root() / "configs/config.yaml"

    try:
        with config_path.open() as config_file:
            config_dict = yaml.safe_load(config_file)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from err
    except Exception as err:  # Catch other unexpected errors
        raise Exception(f"An unexpected error occurred while reading the config file: {err}") from err

    try:
        return Config(**config_dict)
    except (ValidationError, ValueError) as err:
        raise ValueError(f"Configuration validation failed: {err}") from err
