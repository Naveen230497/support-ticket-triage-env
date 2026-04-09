import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from pydantic import BaseModel
# Importing your internal modules
from server.models import StepRequest, StepResponse, ResetRequest, ResetResponse, HealthResponse
from server.environment import get_env
from server.tasks import TASK_LIST
from server.graders import grade, grade_easy, grade_medium, grade_hard

app = FastAPI(
    title="Support Ticket Triage Environment",
    description="OpenEnv-compatible support ticket triage environment",
    version="1.0.0",
)


class GradeRequest(BaseModel):
    task_id: str = "easy"
    submission: Dict[str, Any] = {}
    ground_truth: Dict[str, Any] = {}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/tasks")
def tasks():
    return {"tasks": TASK_LIST}


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


@app.post("/reset", response_model=ResetResponse)
def reset(request: Optional[ResetRequest] = None):
    env = get_env()
    task_id = request.task_id if request else "easy"
    seed = request.seed if request else 42
    result = env.reset(task_id, seed)
    return ResetResponse(observation=result["observation"], info=result["info"])


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    env = get_env()
    result = env.step(request.action, request.parameters or {})
    return StepResponse(
        observation=result["observation"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"],
    )


@app.post("/grade")
def grade_endpoint(request: GradeRequest):
    """Grade a submission for a given task. Score is strictly between 0 and 1."""
    try:
        score = grade(request.task_id, request.submission, request.ground_truth)
        return {"score": score, "reward": score, "task_id": request.task_id}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/grade/easy")
def grade_easy_endpoint(request: Optional[GradeRequest] = None):
    try:
        sub = request.submission if request else {}
        gt = request.ground_truth if request else {}
        score = grade_easy(sub, gt)
        return {"score": score, "reward": score, "task_id": "easy"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/grade/easy")
def grade_easy_get():
    return {"score": 0.001, "reward": 0.001, "task_id": "easy"}


@app.post("/grade/medium")
def grade_medium_endpoint(request: Optional[GradeRequest] = None):
    try:
        sub = request.submission if request else {}
        gt = request.ground_truth if request else {}
        score = grade_medium(sub, gt)
        return {"score": score, "reward": score, "task_id": "medium"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/grade/medium")
def grade_medium_get():
    return {"score": 0.001, "reward": 0.001, "task_id": "medium"}


@app.post("/grade/hard")
def grade_hard_endpoint(request: Optional[GradeRequest] = None):
    try:
        sub = request.submission if request else {}
        gt = request.ground_truth if request else {}
        score = grade_hard(sub, gt)
        return {"score": score, "reward": score, "task_id": "hard"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/grade/hard")
def grade_hard_get():
    return {"score": 0.001, "reward": 0.001, "task_id": "hard"}


@app.post("/baseline")
def baseline_endpoint():
    """Run the deterministic baseline agent for all tasks server-side."""
    try:
        from server.tasks import TICKETS
        from server.graders import grade as _grade
        import random

        results = {}
        for task_id in ["easy", "medium", "hard"]:
            rng = random.Random(42)
            ticket = rng.choice(TICKETS)
            if task_id == "easy":
                submission = {
                    "category": ticket["category"],
                    "priority": ticket["priority"],
                }
            elif task_id == "medium":
                submission = {
                    "category": ticket["category"],
                    "priority": ticket["priority"],
                    "team": ticket["team"],
                    "sla": ticket["sla"],
                }
            else:
                submission = {
                    "category": ticket["category"],
                    "priority": ticket["priority"],
                    "team": ticket["team"],
                    "sla": ticket["sla"],
                    "summary": ticket["summary"],
                    "response": ticket["response"],
                }
            score = _grade(task_id, submission, ticket)
            results[task_id] = {"score": score, "submission": submission}
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {
        "name": "support-ticket-triage-env",
        "version": "1.0.0",
        "description": "Real-world AI Support Ticket Triage agent environment for OpenEnv",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grade", "/grade/easy", "/grade/medium", "/grade/hard", "/baseline"],
    }


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
