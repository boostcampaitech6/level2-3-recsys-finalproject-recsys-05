from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Depends
from models.inference import inference
from schemas import PredictionRequest, PredictionResponse
import httpx
import asyncio
from config import get_config, Config
import datetime

router = APIRouter()

# 캐싱 고려하기.. post 활용 생각해보기

async def get_most3_champions(
        summoner_id: str,
        config: Config = Depends(get_config)
        ):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{config.riot_api_url}{summoner_id}", headers={"X-Riot-Token": config.riot_api_key})
        # Todo: error handling 추가하기
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Summoner not found")
        # Todo: top 3 champions 가져오는 거 구현하기
        return response.json()

@router.get("/duo-recommendation/{summoner_id}")
async def duo_recommendation(
    request: PredictionRequest,
    config: Config = Depends(get_config),
) -> PredictionResponse:
    # model load
    if config.model_path is None:
        raise ValueError("Model path is not defined")

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

    # example return
    # prediction = {
    #     "summonor_id_1": 0.5,
    #     "summonor_id_2": 0.3,
    #     "summonor_id_3": 0.2,
    #     "summonor_id_4": 0.1,
    # }

    # return prediction
    return PredictionResponse(summonor_ids_score=final_prediction, created_at=str(datetime.datetime.now()))
