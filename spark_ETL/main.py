from etl import ETL
from config import CFG
import time

def main(cfg: CFG):
    etl = ETL()
    etl.transform(cfg.view_id, cfg.query, cfg.data_path, cfg.num_partitions)
    
if __name__ == "__main__":
    start = time.time()
    cfg = CFG('summoner.yaml')
    main(cfg = cfg)
    print("time :", time.time() - start)