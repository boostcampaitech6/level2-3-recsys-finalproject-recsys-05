from google_auth_oauthlib import flow

def authorize_bigquery():
    f_path = "/home/ksj0061/airflow-tutorial/client_secrets.json"
    scopes = ["https://www.googleapis.com/auth/bigquery"]
    appflow = flow.InstalledAppFlow.from_client_secrets_file(
        f_path, scopes=scopes
    )
    appflow.run_local_server(port= 8085)

    credentials = appflow.credentials
    return credentials