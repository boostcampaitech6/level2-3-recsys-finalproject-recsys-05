
from pyspark.sql import SparkSession
import pandas as pd
from pandas_gbq import gbq
from google.oauth2 import service_account
from google.cloud import bigquery
from auth_bigquery import authorize_bigquery

class ETL():
    # spark session 생성
    def __init__(self):
        self.appName = "PySpark - Read JSON Lines"
        
        # 세션 생성시 master 지정 x
        self.spark = SparkSession.builder \
            .appName(self.appName) \
            .getOrCreate()
    
    # json 파일 불러오기
    def extract(self, data_path, num_partitions): # num_partitions: 데이터를 분산하여 읽을 파티션 개수
        df = self.spark.read.json(data_path).repartition(num_partitions)
        df.show(3)
        print("extract Done.")
        return df

    # 전처리 함수
    def transform(self, view_id, query, data_path, num_partitions):
        df = self.extract(data_path, num_partitions)
        df.createOrReplaceTempView(view_id)
        # self.spark.catalog.cacheTable(view_id)
        df = self.spark.sql(query)
        # self.spark.catalog.uncacheTable(view_id)
        df = df.toPandas().dropna()
        df.head(3)
        
        # df.rdd.getNumPartitions() # 사용안할 경우: 22.70798945426941, 사용할 경우: 22.165385007858276 하지만 메모리 문제 발생
        # 캐시 생성을 안할 경우에 연산 속도가 2초 빠름
        print("transform Done.")
        return df
        
    def load(self, view_id, query, data_path, num_partitions, dataset_id, table_id):
        credentials = authorize_bigquery()
        
        df = self.transform(view_id, query, data_path, num_partitions)
        
        key_file_path = "/home/ksj0061/airflow-tutorial/teemo-415918-414755ce7c80.json"
        
        # Create BigQuery client
        credential = service_account.Credentials.from_service_account_file(key_file_path)
        bigquery.Client(credentials = credential, project = credential.project_id)
        
        # 빅쿼리 테이블
        project_id = credential.project_id
        dataset_id = dataset_id
        table_id = table_id

        df = df[:100] # 나중에 제거하기

        # unlist
        df = df.explode(["champ_id", "play"]).reset_index(drop=True) # df 탐색후 첫 번째 열을 for문으로 돌려 타입 검사 -> 리스트일 경우 explode, obj일 경우 str

        # 데이터 타입 변경
        df["champ_id"] = df["champ_id"].astype(int)
        df["play"] = df["play"].astype(int)
        df["summoner_id"] = df["summoner_id"].astype(str)
        
        gbq.to_gbq(df, destination_table= f"{dataset_id}.{table_id}", credentials=credentials, project_id=project_id, if_exists="append")

        # PySpark 세션 종료
        self.spark.stop()
        print("load Done.")
        