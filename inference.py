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
    {"id": "task_easy", "max_steps": 8},
    {"id": "task_medium", "max_steps": 20},
    {"id": "task_hard", "max_steps": 30},
]

SYSTEM_PROMPT = """You are a customer support ticket triage specialist agent.
You receive a JSON observation describing a support ticket with triage issues.
Return a single JSON action object with these fields:
  action_type: one of [set_category, set_priority, assign_team, add_tag, set_resolution_time, merge_duplicate, escalate, mark_resolved]
  value: string (required for set_category, set_priority, assign_team, add_tag, set_resolution_time)
  confidence: float between 0.0 and 1.0

Valid values:
- Categories: billing, technical, account_access, product_feedback, shipping
- Priorities: low, medium, high, critical
- Teams: billing_team, tech_support, account_team, product_team, logistics
- Tags: refund, outage, payment, security, escalation
- Resolution time: number of hours as string (e.g. "4" for 4 hours)

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
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current observation:\n{obs_text}\n\nWhat action do you take?"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return {"action_type": "mark_resolved", "confidence": 0.5}


def format_action_str(action: dict) -> str:
    atype = action.get("action_type", "unknown")
    if atype in ("set_category", "set_priority", "assign_team", "add_tag", "set_resolution_time"):
        return f"{atype}({action.get('value', '')})"
    return f"{atype}()"


def run_task(task_id: str, max_steps: int) -> float:
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        result = r.json()
        obs = result.get("observation", result)
        done = obs.get("done", False) if isinstance(obs, dict) else False

        for step in range(1, max_steps + 1):
            if done:
                break

            error = None
            action = {"action_type": "mark_resolved", "confidence": 0.5}
            try:
                action = call_llm(obs if isinstance(obs, dict) else {})
            except Exception as e:
                error = str(e)

            action_str = format_action_str(action)
            try:
                r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
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

        try:
            r = requests.post(f"{ENV_URL}/grader", json={"task_id": task_id}, timeout=30)
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
