from pydantic import Field
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    model_path: str = Field(default="model.joblib", env="MODEL_PATH")
    app_env: str = Field(default="local", env="APP_ENV")

config = Config()
