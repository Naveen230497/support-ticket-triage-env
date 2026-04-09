"""
inference.py - LLM-based agent for the Support Ticket Triage Environment.

Required by hackathon rules:
- Uses the OpenAI client
- Reads configuration from environment variables
- Must be named inference.py and placed at the repo root
- Emits structured [START], [STEP], [END] log lines in plain text
"""
import os
import sys
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "no-key")

BENCHMARK = "support-ticket-triage-env"
SUCCESS_THRESHOLD = 0.5

TASKS = [
    {"id": "easy", "max_steps": 5},
    {"id": "medium", "max_steps": 8},
    {"id": "hard", "max_steps": 12},
]

SYSTEM_PROMPT = """You are a customer support triage expert.
You receive a JSON observation describing a support ticket.
Return a single JSON action object with these fields:
  action: one of [read_ticket, set_field, submit]
  parameters: object with relevant fields

For 'set_field': parameters = {"field": "<fieldname>", "value": "<value>"}
For 'submit': parameters = {"category": "...", "priority": "...", ...}

Available fields: category, priority, team, sla, summary, response
Category values: authentication, billing, bug, how-to, integration
Priority values: low, medium, high, critical
Team values: identity, finance, mobile, support, integrations
SLA values: P1, P2, P3

Return ONLY valid JSON. No explanation, no markdown, no code fences."""

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error=None):
    error_str = "null" if error is None else str(error).replace(" ", "_")[:50]
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def call_llm(observation: dict) -> dict:
    obs_text = json.dumps(observation, indent=2)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current observation:\n{obs_text}\n\nWhat action do you take?"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"action": "submit", "parameters": {"category": "bug", "priority": "medium"}}

def format_action_str(action: dict) -> str:
    act = action.get("action", "unknown")
    params = action.get("parameters", {})
    if act == "set_field":
        return f"set_field({params.get('field', '')},{params.get('value', '')})"
    if act == "submit":
        return f"submit({','.join(f'{k}={v}' for k,v in params.items())})"
    return f"{act}()"

def run_task(task_id: str, max_steps: int) -> float:
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": 42}, timeout=30)
        r.raise_for_status()
        result = r.json()
        obs = result.get("observation", result)
        info = result.get("info", {})
        done = obs.get("done", False) if isinstance(obs, dict) else False

        for step in range(1, max_steps + 1):
            if done:
                break

            error = None
            action = {"action": "submit", "parameters": {}}
            try:
                action = call_llm(obs if isinstance(obs, dict) else {})
            except Exception as e:
                error = str(e)

            action_str = format_action_str(action)
            act_name = action.get("action", "submit")
            act_params = action.get("parameters", {})

            try:
                r = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": act_name, "parameters": act_params},
                    timeout=30
                )
                r.raise_for_status()
                result = r.json()
                obs = result.get("observation", result)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            reward = min(max(reward, 0.0), 1.0)
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Get final score from /grader endpoint
        try:
            r = requests.post(
                f"{ENV_URL}/grader",
                json={"task_id": task_id},
                timeout=30
            )
            r.raise_for_status()
            score = float(r.json().get("score", 0.0))
            score = min(max(score, 0.0), 1.0)
        except Exception:
            score = (sum(rewards) / len(rewards)) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
        score = 0.0
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

def main():
    results = {}
    for task in TASKS:
        try:
            score = run_task(task["id"], task["max_steps"])
            results[task["id"]] = score
        except Exception as e:
            print(f"[ERROR] {task['id']} failed: {e}", flush=True)
            results[task["id"]] = 0.0

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"[RESULT] average_score={avg:.4f}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
