from pydantic import Field, BaseModel
from typing import List
import datetime

class PredictionResponse(BaseModel):
    summonor_ids: List[str] = Field(..., title="Summonor IDs", description="List of summonor IDs")
    created_at: str = Field(default_factory=datetime.datetime.now, title="Created At", description="Timestamp of when the prediction was made")
