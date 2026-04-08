from pydantic import BaseModel
from typing import Optional, Any


class StepRequest(BaseModel):
    action: str
    parameters: Optional[dict] = {}


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = 42


class ResetResponse(BaseModel):
    observation: str
    info: dict


class HealthResponse(BaseModel):
    status: str
    version: str
