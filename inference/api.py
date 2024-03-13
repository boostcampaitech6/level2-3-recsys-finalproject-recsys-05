from fastapi import APIRouter
from schemas import PredictionRequest, PredictionResponse
from load_model import get_model
import datetime

router = APIRouter()

@router.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    # model load
    model = get_model()

    # # model predict
    anchor_summonor_id = request.anchor_summonor_id
    candidate_summonor_ids = request.candidate_summonor_ids
    prediction = model.predict(anchor_summonor_id, candidate_summonor_ids)

    # example return
    prediction = ['1', '2', '3', '4', '5']

    # return prediction
    return PredictionResponse(summonor_ids=prediction, created_at=str(datetime.datetime.now()))
