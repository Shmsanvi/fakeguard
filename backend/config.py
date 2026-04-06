from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    TEXT_MODEL_PATH: str = str(BASE_DIR / "models" / "roberta_fakenews")
    IMAGE_MODEL_PATH: str = str(BASE_DIR / "models" / "efficientnet_forgery")

    TEXT_WEIGHT: float = 0.55
    IMAGE_WEIGHT: float = 0.45

    FAKE_THRESHOLD: float = 0.60
    REAL_THRESHOLD: float = 0.40

    CONSISTENCY_THRESHOLD: float = 0.25

    ELA_QUALITY: int = 90
    ELA_SCALE: int = 15

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_IMAGE_SIZE_MB: int = 10
    REQUEST_TIMEOUT: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
