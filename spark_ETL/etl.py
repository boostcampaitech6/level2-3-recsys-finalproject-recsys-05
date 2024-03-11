import pyspark
from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import to_json, struct
from pyspark.sql.types import StringType

class ETL():
    # spark session 생성
    def __init__(self):
        self.appName = "PySpark - Read JSON Lines"
        #spark_home = "./airflow-tutorial/.venv/bin/pyspark"
        #os.environ['SPARK_HOME'] = spark_home
        # 세션 생성시 master 지정 x
        self.spark = SparkSession.builder \
            .appName(self.appName) \
            .getOrCreate()
    
    # json 파일 불러오기
    def extract(self, data_path, num_partitions): # num_partitions: 데이터를 분산하여 읽을 파티션 개수
        df = self.spark.read.json(data_path).repartition(num_partitions)
        df.show(5)
        return df

    # 전처리 함수
    def transform(self, view_id, query, data_path, num_partitions):
        df = self.extract(data_path, num_partitions)
        df.createOrReplaceTempView(view_id)
        summoner_id = self.spark.sql(query).na.drop()
        summoner_id.show(5)
        
        # df.rdd.getNumPartitions() # 사용안할 경우: 22.70798945426941, 사용할 경우: 22.165385007858276 하지만 메모리 문제 발생
        # 캐시 생성을 안할 경우에 연산 속도가 2초 빠름
        summoner_id.write.mode('overwrite').option("compression", "none").json("/home/ksj0061/airflow-tutorial/test_P04")
        
        # PySpark 세션 종료
        self.spark.stop()
        
        return summoner_id
        