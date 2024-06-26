# Data

## User Summary Data Crwal Guide

패키지 설치

```bash
pip install aiohttp inquirer loguru tqdm orjson ijson
```

크롤링 실행

```bash
####### 스크립트 실행 #######
$ make start_user_crawler

#######  실행 결과   #######
[?] 디렉토리를 선택해주세요: .
[?] 변환할 파일을 선택해주세요: 
   [X] gold.json
   [ ] emerald.json
   [X] silver.json
   [ ] platinum.json
   [X] iron.json
   [ ] diamond.json
 > [X] bronze.json

[?] 저장할 디렉토리의 경로를 입력해주세요: ./summoner_info
[?] 마지막으로 크롤링한 소환사 ID를 입력해주세요: <소환사 아이디> # 공백일 경우 처음부터
```

로그 실시간으로 보는 방법
```bash
# 마지막 20줄 보기
watch -n 1 tail -n 20 user_summary.log
```

## File Upload Guide

패키지 설치

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2
pip install python-dotenv
```

`.env` 파일 생성

```bash
cp .env.example .env
```

`.env` 파일 수정

```bash
GOOGLE_DRIVE_CREDENTIAL_FILE_PATH=your_google_drive_credential_file_path
```

예시

```python
from data.utils.file_upload import new_google_drive_client
from data.config import settings


client = new_google_drive_client(
    settings.google_drive_credential_file_path,
    settings.google_drive_base_folder_id,
)

file_id = client.upload_file(
    file_path="silver.json",
    save_path="2024/02/19",
)
```

## Convert JSON to Avro Guide

패키지 설치

```bash
pip install fastavro orjson loguru inquirer
```

json 파일들을 avro 포맷으로 변환

```bash
####### 스크립트 실행 #######
python -m data.utils.avro_converter

#######  실행 결과   #######
[?] 변환할 파일의 경로를 입력해주세요: .
[?] 변환할 파일을 선택해주세요: 
   [ ] gold.json
   [X] emerald.json
   [ ] silver.json
   [X] platinum.json
   [ ] iron.json
   [X] diamond.json
 > [ ] bronze.json

[?] Schema를 선택해주세요: 
 > USER_LIST

[?] 저장할 파일 이름을 선택해주세요: emerald_and_above.avro

2024-02-16 23:36:23.760 | INFO     | __main__:__init__:18 - Initializing ARVOConverter
2024-02-16 23:36:23.762 | INFO     | __main__:__init__:19 - Schema: USER_LIST
2024-02-16 23:36:23.762 | INFO     | __main__:__init__:20 - Output file name: emerald_and_above.avro
2024-02-16 23:36:23.762 | INFO     | __main__:__init__:21 - Input files: ['emerald.json', 'platinum.json', 'diamond.json']
2024-02-16 23:36:23.762 | INFO     | __main__:start:40 - Parsing emerald.json
2024-02-16 23:36:23.921 | INFO     | __main__:start:40 - Parsing platinum.json
2024-02-16 23:36:24.193 | INFO     | __main__:start:40 - Parsing diamond.json
2024-02-16 23:36:24.237 | INFO     | __main__:start:46 - Writing to emerald_and_above.avro
2024-02-16 23:36:25.074 | INFO     | __main__:start:49 - Conversion complete
```