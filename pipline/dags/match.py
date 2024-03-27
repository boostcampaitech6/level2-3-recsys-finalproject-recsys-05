import aiohttp
import asyncio

from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

import json
from pandas_gbq import gbq
from google.oauth2 import service_account
from google.cloud import bigquery
import logging
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from gbq_oauth.auth_bigquery import authorize_bigquery

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'start_date': datetime(2024, 3, 25),
    'retry_delay': timedelta(days=14),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(f"../pipline/logs/match/{datetime.now()}.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

async def get_match_info(session, headers, match_id, riot_api_key):
    retry = 0
    max_retries = 10  # 최대 재시도 횟수 설정
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={riot_api_key}"
    while retry < max_retries:
        try:
            async with session.get(url, headers=headers) as response:
                retry_after = response.headers.get("Retry-After")                    
                if response.status == 200:
                    logger.info(match_id)
                    content = await response.json()
                    player_data = {
                    "summonerId": [],
                    "riotIdGameName": [],
                    "riotIdTagline": [],
                    "summonerLevel": [],
                    "teamId": [],
                    "individualPosition": [],
                    "role": [],
                    "championId": [],
                    "champLevel": [],
                    "item0": [],
                    "item1": [],
                    "item2": [],
                    "item3": [],
                    "item4": [],
                    "item5": [],
                    "item6": [],
                    "summoner1Casts": [],
                    "summoner1Id": [],
                    "summoner2Casts": [],
                    "summoner2Id": [],
                    "kills": [],
                    "deaths": [],
                    "assists": [],
                    "goldEarned": [],
                    "visionScore": [],
                    "visionWardsBoughtInGame": [],
                    "wardsPlaced": [],
                    "wardsKilled": [],
                    "totalDamageDealtToChampions": [],
                    "totalDamageTaken": [],
                    "timeCCingOthers": [],
                    "totalTimeCCDealt": [],
                    "totalHeal": [],
                    "totalMinionsKilled": [],
                    "totalAllyJungleMinionsKilled": [],
                    "dragonKills": [],
                    "baronKills": [],
                    "objectivesStolen": [],
                    "turretKills": [],
                    "turretsLost": [],
                    "turretTakedowns": [],
                    "commandPings": [],
                    "dangerPings": [],
                    "holdPings": [],
                    "needVisionPings": [],
                    "onMyWayPings": [],
                    "longestTimeSpentLiving": [],
                    "win": [],
                    "spell1Casts": [],
                    "spell2Casts": [],
                    "spell3Casts": [],
                    "spell4Casts": []
                }

                    for player in content["info"]["participants"]:
                        player_data["summonerId"].append(player["summonerId"])
                        player_data["riotIdGameName"].append(player["riotIdGameName"])
                        player_data["riotIdTagline"].append(player["riotIdTagline"])
                        player_data["summonerLevel"].append(player["summonerLevel"])
                        player_data["teamId"].append(player["teamId"])
                        player_data["individualPosition"].append(player["individualPosition"])
                        player_data["role"].append(player["role"])
                        player_data["championId"].append(player["championId"])
                        player_data["champLevel"].append(player["champLevel"])
                        player_data["item0"].append(player["item0"])
                        player_data["item1"].append(player["item1"])
                        player_data["item2"].append(player["item2"])
                        player_data["item3"].append(player["item3"])
                        player_data["item4"].append(player["item4"])
                        player_data["item5"].append(player["item5"])
                        player_data["item6"].append(player["item6"])
                        player_data["summoner1Casts"].append(player["summoner1Casts"])
                        player_data["summoner1Id"].append(player["summoner1Id"])
                        player_data["summoner2Casts"].append(player["summoner2Casts"])
                        player_data["summoner2Id"].append(player["summoner2Id"])
                        player_data["kills"].append(player["kills"])
                        player_data["deaths"].append(player["deaths"])
                        player_data["assists"].append(player["assists"])
                        player_data["goldEarned"].append(player["goldEarned"])
                        player_data["visionScore"].append(player["visionScore"])
                        player_data["visionWardsBoughtInGame"].append(player["visionWardsBoughtInGame"])
                        player_data["wardsPlaced"].append(player["wardsPlaced"])
                        player_data["wardsKilled"].append(player["wardsKilled"])
                        player_data["totalDamageDealtToChampions"].append(player["totalDamageDealtToChampions"])
                        player_data["totalDamageTaken"].append(player["totalDamageTaken"])
                        player_data["timeCCingOthers"].append(player["timeCCingOthers"])
                        player_data["totalTimeCCDealt"].append(player["totalTimeCCDealt"])
                        player_data["totalHeal"].append(player["totalHeal"])
                        player_data["totalMinionsKilled"].append(player["totalMinionsKilled"])
                        player_data["totalAllyJungleMinionsKilled"].append(player["totalAllyJungleMinionsKilled"])
                        player_data["dragonKills"].append(player["dragonKills"])
                        player_data["baronKills"].append(player["baronKills"])
                        player_data["objectivesStolen"].append(player["objectivesStolen"])
                        player_data["turretKills"].append(player["turretKills"])
                        player_data["turretsLost"].append(player["turretsLost"])
                        player_data["turretTakedowns"].append(player["turretTakedowns"])
                        player_data["commandPings"].append(player["commandPings"])
                        player_data["dangerPings"].append(player["dangerPings"])
                        player_data["holdPings"].append(player["holdPings"])
                        player_data["needVisionPings"].append(player["needVisionPings"])
                        player_data["onMyWayPings"].append(player["onMyWayPings"])
                        player_data["longestTimeSpentLiving"].append(player["longestTimeSpentLiving"])
                        player_data["win"].append(player["win"])
                        player_data["spell1Casts"].append(player["spell1Casts"])
                        player_data["spell2Casts"].append(player["spell2Casts"])
                        player_data["spell3Casts"].append(player["spell3Casts"])
                        player_data["spell4Casts"].append(player["spell4Casts"])
                    
                    challenges_dict = {
                        'epicMonsterSteals': [],
                        'stealthWardsPlaced': [],
                        'abilityUses': []
                    }
                    for player in content["info"]["participants"]:
                        challenges_dict['epicMonsterSteals'].append(player["challenges"]["epicMonsterSteals"])
                        challenges_dict['stealthWardsPlaced'].append(player["challenges"]["stealthWardsPlaced"])
                        challenges_dict['abilityUses'].append(player["challenges"]["abilityUses"])
                        
                    perks_dict = {
                        "defense": [],
                        "flex": [],
                        "offense": []
                        }

                    for player in content["info"]["participants"]:
                        perks_dict["defense"].append(player["perks"]["statPerks"]["defense"])
                        perks_dict["flex"].append(player["perks"]["statPerks"]["flex"])
                        perks_dict["offense"].append(player["perks"]["statPerks"]["offense"])
                        
                    # player_data 딕셔너리를 데이터프레임으로 변환
                    player_df = pd.DataFrame(player_data)
                    del player_data
                    # challenges_dict 딕셔너리를 데이터프레임으로 변환
                    challenges_df = pd.DataFrame(challenges_dict)
                    del challenges_dict
                    # perks_dict 딕셔너리를 데이터프레임으로 변환
                    perks_df = pd.DataFrame(perks_dict)
                    del perks_dict
                    new_match_df = pd.concat([player_df, challenges_df, perks_df], axis=1)
                    
                    new_match_df['matchId'] = content["metadata"]["matchId"]
                    new_match_df['gameCreation'] = content["info"]["gameCreation"]
                    new_match_df['gameDuration'] = content["info"]["gameDuration"]
                    new_match_df['gameVersion'] = content["info"]["gameVersion"]
                    del content
                    return new_match_df
                elif response.status == 429:
                    retry += 1
                    logger.info(f"429 Error occurred in {match_id}, retrying {retry_after} (retry {retry}/{max_retries})")
                    await asyncio.sleep(int(retry_after)+1)
                elif response.status == 503:
                    logger.info(f"503 Error occurred in {match_id}, retrying {retry_after} (retry {retry}/{max_retries})")
                    retry += 1
                    await asyncio.sleep(5)
                else:
                    logger.info(f"Error {response.status} occurred in {match_id}")
                    await asyncio.sleep(5)
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Network error occurred: {e}, {match_id}")
            retry += 1
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}, {match_id}")
            retry += 1
            await asyncio.sleep(5)
            if retry == max_retries: 
                logger.info(f"Maximum retries ({max_retries}) reached for match {match_id}. Skipping...")
    return None
            
async def get_match_ids(session, headers, riot_api_key, puuid, start_date, end_date, match_count):
    retry = 0
    max_retries = 10  # 최대 재시도 횟수 설정
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={start_date}&endTime={end_date}&type=ranked&start=0&count={match_count}&api_key={riot_api_key}"
    while retry < max_retries:
        try:
            async with session.get(url, headers=headers) as response:            
                retry_after = response.headers.get("Retry-After")
                if response.status == 200:
                    logger.info(f"puuid, {puuid}")
                    content = await response.json()
                    if len(content) != 0:
                        return_value = content
                        del content
                        return return_value
                elif response.status == 429:
                    retry += 1
                    logger.info(f"429 Error occurred in {puuid}, retrying {retry_after} (retry {retry}/{max_retries})")
                    await asyncio.sleep(int(retry_after)+1)
                else:
                    logger.info(f"status {response.status} Error {response.status} occurred in {puuid}")
                    await asyncio.sleep(10)
                    break
        except Exception as e:
            logger.error(f"Error occurred in {str(e), puuid}")
            retry += 1 
            await asyncio.sleep(10)
            if retry == max_retries:
                logger.info(f"Maximum retries ({max_retries}) reached for match {puuid}. Skipping...")

async def get_match_id(session, headers, credential, credentials, riot_api_key, puuid_list):
    match_df = pd.DataFrame()
    now = datetime.now()
    two_weeks_before = now - timedelta(weeks=2)
    start_date = int(datetime.timestamp(two_weeks_before))
    end_date = int(datetime.timestamp(now))
    match_count = 100
    project_id = credential.project_id
    dataset_id = "match_dataset"
    table_id = "match_test_test"

    match_id_list = [get_match_ids(session, headers, riot_api_key, puuid, start_date, end_date, match_count) for puuid in puuid_list]
    
    match_id_chunks = await asyncio.gather(*match_id_list)
    match_id_list = [match_id for chunk in match_id_chunks if chunk is not None for match_id in chunk]
    
    
    unique_match_id_list = list(set(match_id_list))
    logger.info(f"match count: {len(unique_match_id_list)}")

    match_id_list = [get_match_info(session, headers, match_id, riot_api_key) for match_id in unique_match_id_list]
    logger.info("Finished get_match_info")
    
    new_match_df_list = await asyncio.gather(*match_id_list)
    
    for df in new_match_df_list:
        if df is not None and not df.empty:            
            match_df = pd.concat([match_df, df], ignore_index=True)
            if len(match_df) >= 50000:
                logger.info(f"match_df length: {len(match_df)}")
                for column in match_df.columns:
                    if match_df[column].dtype == 'object':
                        match_df[column] = match_df[column].astype(str)     
                    if match_df[column].dtype == 'float64':
                        match_df[column] = match_df[column].astype(int)
                gbq.to_gbq(match_df, destination_table=f"{dataset_id}.{table_id}", credentials=credentials, project_id=project_id, if_exists="append")
                match_df = pd.DataFrame()
                logger.info("Data appended to BigQuery table.")
    logger.info("Finished concat DataFrame")
    
    
    if not match_df.empty:
        logger.info(f"match_df length: {len(match_df)}")
        for column in match_df.columns:
            if match_df[column].dtype == 'object':
                match_df[column] = match_df[column].astype(str)     
            if match_df[column].dtype == 'float64':
                match_df[column] = match_df[column].astype(int)
        print(match_df.info())
        gbq.to_gbq(match_df, destination_table=f"{dataset_id}.{table_id}", credentials=credentials, project_id=project_id, if_exists="append")
        logger.info("Remaining data appended to BigQuery table.")
        logger.info("Done")

def get_puuid(client):
    query = '''
        SELECT DISTINCT puuid FROM 
        `teemo-415918.summoner_dataset.summoner` WHERE tier = "DIAMOND" AND rank = "III" LIMIT 5
        '''

    job = client.query(query)
    job_df = job.to_dataframe()
    puuid_list = job_df["puuid"].to_list()
    return puuid_list

async def main(credentials):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6000)) as session:
        key_file_path = "../pipline/keys/teemo-415918-414755ce7c80.json"
        credential = service_account.Credentials.from_service_account_file(key_file_path)
        client = bigquery.Client(credentials=credentials, project=credential.project_id)
        with open("../pipline/keys/riot_api.json") as f:
            riot_key = json.load(f)
        riot_api_key = riot_key["key"]
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"}
        puuid_list = get_puuid(client)
        logger.info(f"puuid count: {len(puuid_list)}")
        await get_match_id(session, headers, credential, credentials, riot_api_key, puuid_list)
   
def run_task():
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Started at: {start_time}")
    credentials = authorize_bigquery()
    asyncio.run(main(credentials))
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time
    logger.info(f"전체 실행시간: {total_time}")
    logger.info(f"Completed at: {end_time}")

dag = DAG(
    dag_id='match_info',
    description="get match data info",
    default_args=default_args,
    schedule="0 0 */14 * 4",  # 2주 목요일에 실행
    catchup=True,
)

get_match_info_task = PythonOperator(
    task_id="get_match_info",
    python_callable=run_task,
    dag=dag,
)

get_match_info_task