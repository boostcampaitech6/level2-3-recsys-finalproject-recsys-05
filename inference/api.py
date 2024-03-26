from fastapi import APIRouter
from fastapi import Depends
from models.inference import inference
from schemas import PredictionRequest, PredictionResponse

from pymongo import MongoClient
import asyncio
import datetime

router = APIRouter()

# MongoDB

client = MongoClient('mongodb://localhost:27017/')
db = client['lol_db']
collection = db['most_champion']

async def get_most3_champions(
        summoner_id: str,
        ):
    result = collection.find({"summoner_id": summoner_id})
    # print(f'result : {result}')
    most3_champions = []

    for user in result:
        most3_champions = user['champion_id']

    return most3_champions


# BigQuery

# from google.cloud import bigquery

# credentials = service_account.Credentials.from_service_account_file(
#     "your_service_account_file.json"
# )

# client = bigquery.Client(credentials=credentials, project=credentials.project_id)
# async def get_most3_champions(
#         summoner_id: str,
#         ):
#     query = f"""
#     SELECT champion_id
#     FROM lol_db.most_champion
#     WHERE summoner_id = {summoner_id}
#     """
#     result = await client.query(query)
#     most3_champions = []
#     for row in result:
#         most3_champions.append(row['champion_id'])

#     return most3_champions

@router.get("/duo-recommendation/{summoner_id}")
async def duo_recommendation(
    request: PredictionRequest,
    summoner_id: str
) -> PredictionResponse:
    # model predict
    anchor_summonor_id = request.anchor_summonor_id
    candidate_summonor_ids = request.candidate_summonor_ids

    candidate2idx = {candidate_summonor_id: idx for idx, candidate_summonor_id in enumerate(candidate_summonor_ids)}
    idx2candidate = {idx: candidate_summonor_id for idx, candidate_summonor_id in enumerate(candidate_summonor_ids)}
    # indexing candidatie_summonor_ids
    anchor_most3_champions = await get_most3_champions(anchor_summonor_id)

    # 비동기로 처리
    tasks = [get_most3_champions(candidate_summonor_id) for candidate_summonor_id in candidate_summonor_ids]
    result = await asyncio.gather(*tasks)

    candidate_most3_champions = {candidate2idx[candidate_summonor_id]: champion for candidate_summonor_id, champion in zip(candidate_summonor_ids, result)}

    # prediction : dict of {idx: score}
    inference_result = inference(anchor_most3_champions, candidate_most3_champions)

    # index to candidate_summonor_id with score
    final_prediction = {idx2candidate[idx]: score for idx, score in inference_result.items()}

    # return prediction
    return PredictionResponse(summonor_ids_score=final_prediction, created_at=str(datetime.datetime.now()))
