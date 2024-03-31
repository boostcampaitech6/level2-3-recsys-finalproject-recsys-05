from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    region_name: str = "ap-northeast-2"
    dynamodb_aws_access_key_id: str = ""
    dynamodb_aws_secret_access_key: str = ""


config = Config(_env_file=".env", _env_file_encoding="utf-8")


def get_config():
    return config
