from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_drive_credential_file_path: str
    google_drive_base_folder_id: str


settings = Settings()
