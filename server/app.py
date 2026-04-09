"""FastAPI application for the Support Ticket Triage Environment."""
import uvicorn
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import time
from collections import defaultdict
from server.environment import get_env
from server.tasks import TASK_LIST, list_tasks
from server.graders import grade, grade_easy, grade_medium, grade_hard

VALID_TASK_IDS = {"easy", "medium", "hard"}

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

app = FastAPI(
    title="Support Ticket Triage Environment",
    description="OpenEnv-compatible support ticket triage environment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    task_id: Optional[str] = "easy"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action: str
    parameters: Optional[Dict[str, Any]] = {}

class GraderRequest(BaseModel):
    task_id: str
    submission: Optional[Dict[str, Any]] = {}
    ground_truth: Optional[Dict[str, Any]] = {}

@app.get("/")
def root():
    return {
        "service": "Support Ticket Triage Environment API",
        "version": "2.0.0",
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

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/metadata")
def metadata():
    return {
        "name": "support-ticket-triage-env",
        "description": "Real-world AI Support Ticket Triage agent environment for OpenEnv",
        "version": "2.0.0",
        "author": "Naveen2304",
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read_ticket", "set_field", "submit"]},
                "parameters": {"type": "object"}
            }
        },
        "observation": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "task": {"type": "string"},
                "required_fields": {"type": "array"},
                "reward": {"type": "number"},
                "done": {"type": "boolean"},
                "task_id": {"type": "string"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "current_step": {"type": "integer"},
                "done": {"type": "boolean"},
                "episode_reward": {"type": "number"}
            }
        },
    }

@app.get("/tasks")
def tasks():
    """List all available tasks - returns flat list for OpenEnv compatibility."""
    return list_tasks()

@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    env = get_env()
    task_id = request.task_id if request else "easy"
    seed = request.seed if request else 42
    if task_id not in VALID_TASK_IDS:
        raise HTTPException(status_code=400, detail=f"Invalid task_id. Valid: {sorted(VALID_TASK_IDS)}")
    try:
        result = env.reset(task_id, seed)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reset failed: {e}")

@app.post("/step")
def step(request: StepRequest):
    env = get_env()
    try:
        result = env.step(request.action, request.parameters or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Step failed: {e}")

@app.get("/state")
def state():
    env = get_env()
    return {
        "task_id": env.task_config["task_id"] if env.task_config else None,
        "current_step": env.current_step,
        "done": env.done,
        "submission": env.submission,
        "episode_reward": env.episode_reward,
    }

@app.post("/grader")
def grader(request: GraderRequest):
    """Grade the current submission."""
    try:
        task_id = request.task_id
        if task_id not in VALID_TASK_IDS:
            raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")
        submission = request.submission or {}
        ground_truth = request.ground_truth or {}
        score = grade(task_id, submission, ground_truth)
        return {"task_id": task_id, "score": score, "reward": score}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Grading failed: {e}")

@app.post("/baseline")
def baseline():
    """Run a deterministic baseline agent across all tasks and return scores."""
    env = get_env()
    results = {}
    try:
        env.reset("easy", 42)
        env.step("submit", {"category": "authentication", "priority": "high"})
        score = grade("easy", env.submission, env.task_config["ticket"])
        results["easy"] = {"score": score, "status": "success"}
    except Exception as e:
        results["easy"] = {"score": 0.0, "status": "error", "error": str(e)}
    try:
        env.reset("medium", 42)
        env.step("submit", {"category": "authentication", "priority": "high", "team": "identity", "sla": "P1"})
        score = grade("medium", env.submission, env.task_config["ticket"])
        results["medium"] = {"score": score, "status": "success"}
    except Exception as e:
        results["medium"] = {"score": 0.0, "status": "error", "error": str(e)}
    try:
        env.reset("hard", 42)
        env.step("submit", {
            "category": "authentication",
            "priority": "high",
            "team": "identity",
            "sla": "P1",
            "summary": "User unable to login after password reset",
            "response": "We apologize for the inconvenience. Our identity team is investigating your login issue and will resolve it within 2 hours."
        })
        score = grade("hard", env.submission, env.task_config["ticket"])
        results["hard"] = {"score": score, "status": "success"}
    except Exception as e:
        results["hard"] = {"score": 0.0, "status": "error", "error": str(e)}
    return {"baseline_scores": results}

@app.post("/grade/easy")
def grade_easy_endpoint(request: Optional[GraderRequest] = None):
    sub = request.submission if request else {}
    gt = request.ground_truth if request else {}
    score = grade_easy(sub or {}, gt or {})
    return {"score": score, "reward": score, "task_id": "easy"}

@app.get("/grade/easy")
def grade_easy_get():
    return {"score": 0.001, "reward": 0.001, "task_id": "easy"}

@app.post("/grade/medium")
def grade_medium_endpoint(request: Optional[GraderRequest] = None):
    sub = request.submission if request else {}
    gt = request.ground_truth if request else {}
    score = grade_medium(sub or {}, gt or {})
    return {"score": score, "reward": score, "task_id": "medium"}

@app.get("/grade/medium")
def grade_medium_get():
    return {"score": 0.001, "reward": 0.001, "task_id": "medium"}

@app.post("/grade/hard")
def grade_hard_endpoint(request: Optional[GraderRequest] = None):
    sub = request.submission if request else {}
    gt = request.ground_truth if request else {}
    score = grade_hard(sub or {}, gt or {})
    return {"score": score, "reward": score, "task_id": "hard"}

@app.get("/grade/hard")
def grade_hard_get():
    return {"score": 0.001, "reward": 0.001, "task_id": "hard"}

def main():
    """Entry point for running the server."""
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
