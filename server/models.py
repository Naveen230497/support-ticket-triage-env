from pydantic import BaseModel
from typing import Optional, Any, Union


class StepRequest(BaseModel):
    action: str
    parameters: Optional[dict] = {}


class StepResponse(BaseModel):
    observation: Union[dict, str]
    reward: float
    done: bool
    info: dict


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = 42


class ResetResponse(BaseModel):
    observation: Union[dict, str]
    info: dict


class HealthResponse(BaseModel):
    status: str
    version: str
