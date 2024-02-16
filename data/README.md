# Data

## Convert JSON to Avro Guide

패키지 설치

```bash
$ pip install fastavro orjson loguru inquirer
```

json 파일들을 avro 포맷으로 변환

```bash
####### 스크립트 실행 #######
$ python -m data.utils.avro_converter

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