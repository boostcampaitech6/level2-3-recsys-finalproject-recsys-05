import aiohttp
import asyncio

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

import time
import random
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

async def fetch_puuid(session, headers, riot_api_key, summonerId):
    retry = 0
    max_retries = 10  # 최대 재시도 횟수 설정
    url = f"https://kr.api.riotgames.com/lol/summoner/v4/summoners/{summonerId}?api_key={riot_api_key}"
    while retry < max_retries:
        try:
            async with session.get(url, headers=headers) as response:
                retry_after = response.headers.get("Retry-After")
                if response.status == 200:
                    content = await response.json()
                    puuid = content["puuid"]
                    return puuid
                elif response.status == 429:
                    retry += 1
                    print(f"429 Error occurred in {summonerId}, retrying {retry_after} (retry {retry}/{max_retries})")
                    await asyncio.sleep(int(retry_after)+1)
                else:
                    print(f"status {response.status} Error {response.status} occurred in {summonerId}")
                    await asyncio.sleep(10)
                    break
        except Exception as e:
            print(f"Error occurred in {str(e), summonerId}")
            retry += 1 
            await asyncio.sleep(10)
            if retry == max_retries:
                print(f"Maximum retries ({max_retries}) reached for match {summonerId}. Skipping...")

async def get_puuid(session, headers, riot_api_key, summonerId_list):
    puuid_list = []
    tasks = []
    for summonerId in summonerId_list:
        task = asyncio.create_task(fetch_puuid(session, headers, riot_api_key, summonerId))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if result is not None and not isinstance(result, Exception):
            puuid_list.append(result)
        elif isinstance(result, Exception):
            print(f"error: {result}")
            await asyncio.sleep(random.uniform(4, 5))
    return puuid_list

async def get_summoner(session, credentials):
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}
    tier = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
    division = ["I", "II", "III", "IV"]
    page_num = 0
    tier_num = 0
    division_num = 0

    with open("../pipeline/keys/riot_api.json") as f:
        riot_key = json.load(f)
    riot_api_key = riot_key["key"]

    df = pd.DataFrame()
    
    key_file_path = "../pipeline/keys/teemo.json"
    credential = service_account.Credentials.from_service_account_file(key_file_path)
    bigquery.Client(credentials=credential, project=credential.project_id)

    project_id = credential.project_id
    dataset_id = "summoner_dataset"
    table_id = "summoner"
    retry = 0
    max_retries = 10  # 최대 재시도 횟수 설정
    url = f"https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier[tier_num]}/{division[division_num]}?page={page_num}&api_key={riot_api_key}"
    while retry < max_retries:
        try:
            async with session.get(url, headers=headers) as response:
                retry_after = response.headers.get("Retry-After")
                if response.status == 200:
                    content = await response.json()
                    print(f"tier: {tier[tier_num]}, division: {division[division_num]}, page_num: {page_num}, length: {len(content)}")
                    if len(content) == 0:
                        if tier[tier_num] == tier[-1] and division[division_num] == division[-1]:
                            gbq.to_gbq(df, destination_table=f"{dataset_id}.{table_id}", credentials=credentials, project_id=project_id, if_exists="append")
                            print(f"Done, tier: {tier[tier_num]}, division: {division[division_num]}, page_num: {page_num}, length: {len(content)}")
                            break
                        page_num = 1
                        if division[division_num] == division[-1]:
                            division_num = 0
                            tier_num += 1
                        else:
                            division_num += 1
                    else:
                        tier_list = []
                        rank_list = []
                        summonerId_list = []
                        summonerName_list = []
                        leaguePoints_list = []
                        wins_list = []
                        losses_list = []
                        
                        for player in content:
                            tier_list.append(player['tier'])
                            rank_list.append(player['rank'])
                            summonerId_list.append(player['summonerId'])
                            summonerName_list.append(player['summonerName'])
                            leaguePoints_list.append(player['leaguePoints'])
                            wins_list.append(player['wins'])
                            losses_list.append(player['losses'])
                        puuid_list = await get_puuid(session, headers, riot_api_key, summonerId_list)

                        data = {
                            'tier': tier_list,
                            'rank': rank_list,
                            'puuid': puuid_list,
                            'summonerId': summonerId_list,
                            'leaguePoints': leaguePoints_list,
                            'summonerName': summonerName_list,
                            'wins': wins_list,
                            "losses": losses_list
                        }
                        page_num += 1
                        df_new = pd.DataFrame(data)
                        df = pd.concat([df, df_new], ignore_index=True)
                        if len(df) >= 10000:
                            gbq.to_gbq(df, destination_table=f"{dataset_id}.{table_id}", credentials=credentials, project_id=project_id, if_exists="append")
                            print(f"To gbq, tier: {tier[tier_num]}, division: {division[division_num]}, page_num: {page_num-1}, length: {len(content)}")
                            df = pd.DataFrame()
                elif response.status == 429:
                    retry += 1
                    print(f"429 Error occurred in {url}, retrying {retry_after} (retry {retry}/{max_retries})")
                    await asyncio.sleep(int(retry_after)+1)
                elif response.status == 503:
                    print(f"503 Error occurred in {url}, retrying {retry_after} (retry {retry}/{max_retries})")
                    retry += 1
                    await asyncio.sleep(5)
                else:
                    print(f"Error {response.status} occurred in {url}")
                    await asyncio.sleep(5)
                    return None        
        except aiohttp.ClientError as e:
            print(f"Network error occurred: {e}, {url}")
            retry += 1
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Unexpected error occurred: {e}, {url}")
            retry += 1
            await asyncio.sleep(5)
            if retry == max_retries: 
                print(f"Maximum retries ({max_retries}) reached for match {url}. Skipping...")
    return None

async def main(credentials):
    async with aiohttp.ClientSession() as session:
        await get_summoner(session, credentials)

def run_task():
    start_time = time.time()
    credentials = authorize_bigquery()
    asyncio.run(main(credentials))
    end_time = time.time()
    total_time = end_time - start_time
    print(f"total time: {total_time} sec")

dag = DAG(
    dag_id='summoner_info',
    description="get summoner data info",
    default_args=default_args,
    schedule="0 0 */14 * 4",  # 2주 목요일에 실행
    catchup=True,
)

get_summoner_info_task = PythonOperator(
    task_id="get_summoner_info",
    python_callable=run_task,
    dag=dag,
)

get_summoner_info_task