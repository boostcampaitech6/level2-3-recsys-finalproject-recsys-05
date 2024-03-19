import os
from google_auth_oauthlib import flow
from google.oauth2.credentials import Credentials

# 파일에 저장할 빅쿼리 인증 경로
credentials_path = "/home/ksj0061/level2-3-recsys-finalproject-recsys-05/pipline/keys/bigquery_credentials.json"

def authorize_bigquery():
    # 빅쿼리 인증 파일이 있을 경우 
    if os.path.exists(credentials_path):
        credentials = Credentials.from_authorized_user_file(credentials_path)
        
    # 빅쿼리 인증을 하지 않았을 경우
    else:
        f_path = "/home/ksj0061/level2-3-recsys-finalproject-recsys-05/pipline/keys/client_secrets.json"
        scopes = ["https://www.googleapis.com/auth/bigquery"]
        appflow = flow.InstalledAppFlow.from_client_secrets_file(f_path, scopes=scopes)
        credentials = appflow.run_local_server(port=8085)
        
        # 최조 인증을 마치면 json 파일에 빅쿼리 인증 저장
        with open(credentials_path, "w") as credentials_file:
            credentials_file.write(credentials.to_json())

    return credentials