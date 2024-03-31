from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import os


class GoogleDriveClient:
    def __init__(self, credentials_file_path: str, base_folder_id: str):
        self.credentials_file_path = credentials_file_path
        self.base_folder_id = base_folder_id
        self.service = None

    def authenticate(self):
        """Google Drive API 인증"""
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_file_path
        )
        self.service = build("drive", "v3", credentials=credentials)

    def upload_file(self, file_path, save_path) -> str:
        """파일 업로드"""
        if self.service is None:
            raise ValueError("Service not initialized")

        file_metadata = {
            "name": os.path.basename(file_path),
            "parents": [self.parse_save_path(save_path)],
        }
        media = MediaFileUpload(
            file_path, mimetype="application/octet-stream", resumable=True
        )
        file = (
            self.service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        return file.get("id")

    def parse_save_path(self, folder_path) -> str:
        """파일 경로 파싱"""
        folder_paths = folder_path.split("/")
        parent_id = self.base_folder_id

        for folder in folder_paths:
            parent_id = self.find_or_create(folder, parent_id)

        return parent_id

    def find_or_create(self, folder_name, parent_id=None):
        """폴더 찾기 또는 생성"""
        response = (
            self.service.files()
            .list(
                q=f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and '{parent_id}' in parents",
                spaces="drive",
                fields="files(id, name)",
            )
            .execute()
        )

        if len(response.get("files")) > 0:
            return response.get("files")[0].get("id")

        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        file = self.service.files().create(body=file_metadata, fields="id").execute()

        return file.get("id")


def new_google_drive_client(
    credentials_file_path: str,
    base_folder_id: str,
):
    """새로운 GoogleDriveClient 인스턴스 생성"""
    return GoogleDriveClient(credentials_file_path, base_folder_id)
