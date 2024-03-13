from etl import ETL
from config import CFG
import time
from auth_bigquery import authorize_bigquery
def main(cfg: CFG, credentials):
    etl = ETL()
    df = etl.extract(cfg.data_path, cfg.num_partitions)
    df = etl.transform(df, cfg.view_id, cfg.query)
    etl.load(df, cfg.dataset_id, cfg.table_id, cfg.view_id, credentials)
    
if __name__ == "__main__":
    credentials = authorize_bigquery()
    start = time.time()
    cfg = CFG('summoner.yaml')
    main(cfg = cfg, credentials = credentials)
    print("time :", time.time() - start)