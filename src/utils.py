from pathlib import Path

from dotenv import load_dotenv


def list_files_in_folder(folder_path: Path, file_extension: str) -> list[Path]:
    return list(folder_path.glob(file_extension))


def get_name_from_path(file_path: Path) -> str:
    return file_path.stem


def load_environment() -> None:
    dotenv_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path)
