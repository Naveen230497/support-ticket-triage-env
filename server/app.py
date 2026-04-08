import uvicorn
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from typing import Optional

# Importing your internal modules
from server.models import StepRequest, StepResponse, ResetRequest, ResetResponse, HealthResponse
from server.environment import get_env
from server.tasks import TASK_LIST, get_task_config
from server.graders import grade

app = FastAPI(
    title="Support Ticket Triage Environment",
    description="OpenEnv-compatible support ticket triage environment",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/tasks")
def tasks():
    return {"tasks": TASK_LIST}


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
def grade_endpoint(request: dict):
    """Grade a submission for a given task.
    Accepts: {"task_id": str, "submission": dict, "ground_truth": dict (optional)}
    If ground_truth is not provided, uses internal task ground_truth.
    """
    try:
        task_id = request.get("task_id", "easy")
        submission = request.get("submission", {})
        ground_truth = request.get("ground_truth", None)

        # If no ground_truth provided, use the task's ticket as ground_truth
        if not ground_truth:
            task_config = get_task_config(task_id, seed=42)
            ground_truth = task_config["ticket"]

        score = grade(task_id, submission, ground_truth)
        return {"score": score, "task_id": task_id}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/")
def root():
    return {
        "name": "support-ticket-triage-env",
        "version": "1.0.0",
        "description": "Real-world AI Support Ticket Triage agent environment for OpenEnv",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/grade"],
    }


def main():
    """
    The entry point called by the 'server' script in pyproject.toml.
    We use the app object directly to ensure the validator can call it.
    """
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
