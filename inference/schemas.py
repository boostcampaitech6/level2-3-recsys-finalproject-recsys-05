from pydantic import Field, BaseModel
from typing import List
import datetime

class PredictionRequest(BaseModel):
    anchor_summonor_id: str = Field(..., title="Anchor Summonor ID", description="Summonor ID to compare others to")
    candidate_summonor_ids: List[str] = Field(..., title="Candidate Summonor IDs", description="List of summonor IDs to compare to the anchor")

class PredictionResponse(BaseModel):
    summonor_ids: List[str] = Field(..., title="Summonor IDs", description="List of summonor IDs")
    created_at: str = Field(default_factory=datetime.datetime.now, title="Created At", description="Timestamp of when the prediction was made")
