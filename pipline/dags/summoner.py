import aiohttp
import asyncio

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

import json
from pandas_gbq import gbq
from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from gbq_oauth.auth_bigquery import authorize_bigquery

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'start_date': datetime(2024, 3, 15),
    'retry_delay': timedelta(days=14),
}

async def get_summoner(session, credentials):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}
    tier = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "DIAMOND"]
    division = ["I", "II", "III", "IV"]
    page_num = 1
    tier_num = 0
    division_num = 0
    api_key = 'RGAPI-13554f38-e55d-4779-b611-9314ca067f71'
    cnt = 0
    # Service account JSON key file path
    key_file_path = "/home/ksj0061/level2-3-recsys-finalproject-recsys-05/pipline/keys/teemo-415918-414755ce7c80.json"

    # Create BigQuery client
    credential = service_account.Credentials.from_service_account_file(key_file_path)
    bigquery.Client(credentials = credential, project = credential.project_id)
    
    project_id = credential.project_id
    dataset_id = "summoner_dataset"
    table_id = "summoner_entries_table"

    while True:
        url = f"https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier[tier_num]}/{division[division_num]}?page={page_num}&api_key={api_key}"
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                try:
                    content = await response.json()
                    cnt += 1
                    if len(content) != 205:
                        page_num = 1
                        if tier_num == len(tier) and division_num == len(division):
                            break
                        if division_num == len(division):
                            division_num = 0
                            tier_num += 1
                        else:
                            division_num += 1
                    else:
                        tier_list = list(map(lambda player: player['tier'], content))
                        rank_list = list(map(lambda player: player['rank'], content))
                        summonerId_list = list(map(lambda player: player['summonerId'], content))
                        summonerName_list = list(map(lambda player: player['summonerName'], content))
                        leaguePoints_list = list(map(lambda player: player['leaguePoints'], content))
                        wins_list = list(map(lambda player: player['wins'], content))
                        losses_list = list(map(lambda player: player['losses'], content))
                        
                        data = {
                            'tier': tier_list,
                            'rank': rank_list,
                            'summonerId': summonerId_list,
                            'leaguePoints': leaguePoints_list,
                            'summonerName': summonerName_list,
                            'wins': wins_list,
                            "losses": losses_list
                            }
                        
                        df = pd.DataFrame(data)
                        gbq.to_gbq(df, destination_table= f"{dataset_id}.{table_id}", credentials=credentials, project_id=project_id, if_exists="append")
                        print(f"tier: {tier[tier_num]}, division: {division[division_num]}, page_num: {page_num}")
                        page_num += 1
                        
                except Exception as e:
                    print(f"An unexpected error occurred for {url}: {e}")
                    
            elif response.status == 404:
                print("Not Found")
                return 
            elif response.status != 200:
                print(f"Error: {response.status}, Retrying for {url}")
                await asyncio.sleep(5)
            else:
                response.raise_for_status()

async def main(credentials):
    async with aiohttp.ClientSession() as session:
        await get_summoner(session, credentials)

def run_task():
    credentials = authorize_bigquery()
    asyncio.run(main(credentials))

dag = DAG(
    dag_id='summoner_infos',
    description="get summoner info data",
    default_args=default_args,
    schedule="0 0 */14 * 4",  # 2주 목요일에 실행
    catchup=True,
)

get_summoner_info_task = PythonOperator(
    task_id="get_summoner_info_task",
    python_callable=run_task,
    dag=dag,
)

get_summoner_info_task