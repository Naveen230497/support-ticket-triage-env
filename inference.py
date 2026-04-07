"""
Inference Script for Support Ticket Triage Environment
=======================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"
TASK_NAME = os.getenv("TICKET_TASK", "task_easy")
BENCHMARK = os.getenv("TICKET_BENCHMARK", "support_ticket_triage")
MAX_STEPS = 15
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5

VALID_CATEGORIES = ["billing", "technical", "account_access", "product_feedback", "shipping"]
VALID_PRIORITIES = ["low", "medium", "high", "critical"]
VALID_TEAMS = ["billing_team", "tech_support", "account_team", "product_team", "logistics"]
VALID_ACTIONS = ["set_category", "set_priority", "assign_team", "add_tag",
                  "set_resolution_time", "merge_duplicate", "escalate", "mark_resolved"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer support triage agent. You receive a support ticket and must fix all triage issues.
    Valid action_types: set_category, set_priority, assign_team, add_tag, set_resolution_time, merge_duplicate, escalate, mark_resolved
    Valid categories: billing, technical, account_access, product_feedback, shipping
    Valid priorities: low, medium, high, critical
    Valid teams: billing_team, tech_support, account_team, product_team, logistics

    Respond with EXACTLY one JSON object per turn:
    {"action_type": "<type>", "value": "<value or null>"}

    Rules:
    - set_category: value must be one of the valid categories
    - set_priority: value must be one of the valid priorities
    - assign_team: value must be one of the valid teams
    - add_tag: value is a short tag string (e.g. "refund", "crash")
    - set_resolution_time: value is number of hours as a string (e.g. "4")
    - merge_duplicate: value is null
    - escalate: value is null
    - mark_resolved: value is null, call this only when all issues are fixed
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def env_reset(task_id: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, value: Optional[str] = None) -> dict:
    payload = {"action_type": action_type, "value": value, "confidence": 1.0}
    resp = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_grader(task_id: str) -> dict:
    resp = requests.post(f"{ENV_BASE_URL}/grader", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_user_prompt(obs: dict, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    ticket = {
        "ticket_id": obs.get("ticket_id"),
        "title": obs.get("title"),
        "description": obs.get("description"),
        "current_category": obs.get("current_category") or "(empty)",
        "current_priority": obs.get("current_priority") or "(empty)",
        "assigned_team": obs.get("assigned_team") or "(empty)",
        "tags": obs.get("tags", []),
        "resolution_time_hours": obs.get("resolution_time_hours", 0),
        "issues_remaining": obs.get("issues_remaining"),
        "feedback": obs.get("feedback"),
        "has_duplicate": obs.get("duplicate_ticket") is not None,
    }
    return textwrap.dedent(
        f"""
        Step: {step}
        Ticket: {ticket}
        History:\n{history_block}
        What is your next action? Respond with exactly one JSON object.
        """
    ).strip()


def get_model_action(client: OpenAI, obs: dict, step: int, history: List[str]):
    import json
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            action_type = parsed.get("action_type", "mark_resolved")
            value = parsed.get("value")
            if isinstance(value, (int, float)):
                value = str(value)
            return action_type, value
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
    return "mark_resolved", None


def run_task(client: OpenAI, task_id: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        reset_result = env_reset(task_id)
        obs = reset_result.get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action_type, value = get_model_action(client, obs, step, history)
            action_str = f"{action_type}('{value}')" if value else f"{action_type}()"

            step_result = env_step(action_type, value)
            obs = step_result.get("observation", {})
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            error = obs.get("feedback") if reward < 0 else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f} | {obs.get('feedback', '')}")

            if done:
                break

        grade_result = env_grader(task_id)
        score = grade_result.get("score", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    run_task(client, TASK_NAME)


if __name__ == "__main__":
    main()
