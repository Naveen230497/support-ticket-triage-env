"""FastAPI application for the Support Ticket Triage Environment."""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, Field
from typing import Optional, Dict, Any
from dataclasses import asdict
import time
import re
from collections import defaultdict
from .environment import SupportTicketEnvironment
from .tasks import list_tasks
from .graders import grade

VALID_TASK_IDS = {"task_easy", "task_medium", "task_hard"}
VALID_ACTION_TYPES = {
    "set_category", "set_priority", "assign_team",
    "add_tag", "set_resolution_time", "merge_duplicate",
    "escalate", "mark_resolved",
}


class RateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.limit = requests_per_minute
        self.log: dict = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        self.log[client_id] = [t for t in self.log[client_id] if now - t < 60]
        if len(self.log[client_id]) >= self.limit:
            return False
        self.log[client_id].append(now)
        return True


rate_limiter = RateLimiter()


def sanitize_string(value: str, max_length: int = 500) -> str:
    if not value:
        return ""
    cleaned = re.sub(r'[<>"\\;{}]', "", str(value))
    return cleaned[:max_length]


env = SupportTicketEnvironment()
env.reset("task_easy")

app = FastAPI(
    title="Support Ticket Triage Environment",
    description=(
        "Real-world support ticket triage and resolution environment for AI agents. "
        "Agents inspect support tickets and fix triage issues such as wrong category, "
        "missing priority, unassigned team, and unresolved duplicates."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 100 requests per minute."},
            headers={"Retry-After": "60"},
        )
    return await call_next(request)


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="task_easy", pattern=r"^task_(easy|medium|hard)$")

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, v):
        if v not in VALID_TASK_IDS:
            raise ValueError(f"Invalid task_id. Allowed: {sorted(VALID_TASK_IDS)}")
        return v


class StepRequest(BaseModel):
    action_type: str = Field(...)
    value: Optional[str] = Field(None, max_length=500)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v):
        if v not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action_type '{v}'. Allowed: {sorted(VALID_ACTION_TYPES)}")
        return v

    @field_validator("value")
    @classmethod
    def sanitize_value(cls, v):
        return sanitize_string(v) if v else None


class GraderRequest(BaseModel):
    task_id: str = Field(..., pattern=r"^task_(easy|medium|hard)$")


def obs_to_dict(obs):
    try:
        return asdict(obs)
    except Exception:
        return obs.__dict__ if hasattr(obs, "__dict__") else str(obs)


def state_to_dict(s):
    try:
        return asdict(s)
    except Exception:
        return s.__dict__ if hasattr(s, "__dict__") else str(s)


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Support Ticket Triage Environment API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "tasks": "GET /tasks",
            "grader": "POST /grader",
            "baseline": "POST /baseline",
        },
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "env": "support-ticket-triage-env",
        "version": "1.0.0",
    }


@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest = None):
    """Reset the environment to the start of a new episode."""
    task_id = (req.task_id if req else None) or "task_easy"
    try:
        obs = env.reset(task_id=task_id)
        return {"observation": obs_to_dict(obs)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reset failed: {e}")


@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    """Submit one action and advance the episode by one step."""
    try:
        class Action:
            pass
        action = Action()
        action.action_type = req.action_type
        action.value = req.value
        action.confidence = req.confidence
        obs = env.step(action)
        return {
            "observation": obs_to_dict(obs),
            "reward": obs.reward,
            "done": obs.done,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Step failed: {e}")


@app.get("/state", tags=["Environment"])
def state():
    """Return the current episode state."""
    return {"state": state_to_dict(env.state)}


@app.get("/tasks", tags=["Environment"])
def tasks():
    """List all available tasks and the action schema."""
    return {"tasks": list_tasks()}


@app.post("/grader", tags=["Grading"])
def grader(req: GraderRequest):
    """Grade the current episode state for a given task."""
    try:
        snapshot = env.get_episode_snapshot()
        score = grade(req.task_id, snapshot)
        return {
            "task_id": req.task_id,
            "score": score,
            "snapshot": snapshot,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Grading failed: {e}")


@app.post("/baseline", tags=["Grading"])
def baseline():
    """Run a deterministic baseline agent across all tasks and return scores."""
    baseline_plans = {
        "task_easy": [
            {"action_type": "set_category", "value": "account_access"},
            {"action_type": "set_priority", "value": "high"},
            {"action_type": "mark_resolved"},
        ],
        "task_medium": [
            {"action_type": "set_category", "value": "billing"},
            {"action_type": "set_priority", "value": "high"},
            {"action_type": "assign_team", "value": "billing_team"},
            {"action_type": "add_tag", "value": "refund"},
            {"action_type": "set_resolution_time", "value": "4"},
            {"action_type": "mark_resolved"},
        ],
        "task_hard": [
            {"action_type": "set_category", "value": "technical"},
            {"action_type": "set_priority", "value": "critical"},
            {"action_type": "assign_team", "value": "tech_support"},
            {"action_type": "merge_duplicate"},
            {"action_type": "escalate"},
            {"action_type": "set_resolution_time", "value": "1"},
            {"action_type": "mark_resolved"},
        ],
    }
    results = {}
    for task_id, plan in baseline_plans.items():
        try:
            env.reset(task_id=task_id)
            for entry in plan:
                class Action:
                    pass
                act = Action()
                act.action_type = entry["action_type"]
                act.value = entry.get("value")
                act.confidence = 1.0
                env.step(act)
            snapshot = env.get_episode_snapshot()
            score = grade(task_id, snapshot)
            results[task_id] = {"score": score, "status": "success"}
        except Exception as e:
            results[task_id] = {"score": 0.0, "status": "error", "error": str(e)}
    env.reset("task_easy")
    return {"baseline_scores": results}


def main():
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
