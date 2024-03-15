FROM confluentinc/cp-kafka-connect:7.3.0

ENV CONNECT_PLUGIN_PATH="/usr/share/java,/usr/share/confluent-hub-components"

RUN confluent-hub install --no-prompt snowflakeinc/snowflake-kafka-connector:1.5.5 && \
    confluent-hub install --no-prompt confluentinc/kafka-connect-jdbc:10.2.2 && \
    confluent-hub install --no-prompt confluentinc/kafka-connect-json-schema-converter:7.3.0 && \
    
# docker 파일 생성
# 각각 따로 
# 실행시 docker compose로 실행
# 하나의 디렉토리안에 yaml 따로 생성
# container 간의 파이썬 버전 통일은 의미 없음 -> 파이썬을 실행해주는 독립된 컴퓨터

# 컨테이너 접속해서 라이브러리 설치하면 됨 -> 컨테이너를 접속할 일이 거의 없어야 함
# 최종적으로 올릴 컨테이너는 
# 컨테이너는 띄워 놓고 안에 들어가서 세팅을 하기보다는 docker compose up으로 실행했을 때 모든 세팅
# kafka는 kafka랑 consumer를 연결
# pip3 install aiohttp asyncio numpy pandas &&\
# pip3 install apache-airflow==2.6.3 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.6.3/constraints-3.8.txt"
# dockerfile -> iamge를 만들기 위한 설계도