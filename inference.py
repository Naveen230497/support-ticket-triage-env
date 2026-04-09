import os
import json
import sys
import httpx
from openai import OpenAI

# --- GLOBAL CONFIGURATION ---
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
OPENAI_API_KEY = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_ID = os.environ.get("TASK_ID", "easy")
SEED = int(os.environ.get("SEED", "42"))

# Initialize OpenAI client via env variables (required by hackathon)
client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)


def call_env(endpoint: str, payload: dict) -> dict:
    try:
        with httpx.Client(timeout=60) as http:
            response = http.post(f"{ENV_URL}{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"[DEBUG] Env call error: {e}", file=sys.stderr)
        return {"observation": {}, "reward": 0.001, "done": True, "info": {"error": str(e)}}


def call_grade(task_id: str, submission: dict, ground_truth: dict) -> float:
    """Call the /grade endpoint."""
    try:
        with httpx.Client(timeout=60) as http:
            payload = {
                "task_id": task_id,
                "submission": submission,
                "ground_truth": ground_truth,
            }
            response = http.post(f"{ENV_URL}/grade", json=payload)
            response.raise_for_status()
            data = response.json()
            score = float(data.get("score", 0.001))
            score = max(0.001, min(0.999, score))
            return score
    except Exception as e:
        print(f"[DEBUG] Grade call error: {e}", file=sys.stderr)
        return 0.001


def agent_step(observation: dict, task_id: str, step: int, info: dict) -> dict:
    """Use LLM to decide next action based on observation."""
    required_fields = info.get("required_fields", ["category", "priority"])

    if task_id == "easy":
        fields_hint = "category (authentication/billing/bug/how-to/integration) and priority (low/medium/high/critical)"
    elif task_id == "medium":
        fields_hint = "category, priority, team (identity/finance/mobile/support/integrations), and sla (P1/P2/P3)"
    else:
        fields_hint = "category, priority, team, sla, summary (short summary), and response (draft reply to customer)"

    # Convert dict observation to string for LLM
    obs_str = json.dumps(observation, indent=2) if isinstance(observation, dict) else str(observation)

    system_prompt = (
        "You are a customer support triage expert. Analyze the support ticket and return a JSON action.\n"
        "Actions available:\n"
        "  - {\"action\": \"set_field\", \"parameters\": {\"field\": \"\", \"value\": \"\"}} - set a field\n"
        "  - {\"action\": \"submit\", \"parameters\": {}} - submit final answer\n"
        f"Required fields for this task: {fields_hint}\n"
        "Return ONLY valid JSON, nothing else."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Step {step}. Current observation:\n{obs_str}"},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(completion.choices[0].message.content)
        if "action" not in result:
            result = {"action": "read_ticket", "parameters": {}}
        return result
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", file=sys.stderr)
        return {"action": "read_ticket", "parameters": {}}


def main():
    # 1. REQUIRED START FORMAT (exact format required by validator)
    print(f"[START] task={TASK_ID} env=support-ticket-triage-env model={MODEL}")
    sys.stdout.flush()

    rewards = []
    step_count = 0
    score = 0.001
    submission = {}
    ground_truth = {}

    try:
        # 2. Reset environment
        reset_result = call_env("/reset", {"task_id": TASK_ID, "seed": SEED})
        observation = reset_result.get("observation", {})
        info = reset_result.get("info", {})
        ground_truth = info.get("ground_truth", {})
        max_steps = info.get("max_steps", 10)

        # 3. Agent Loop
        for step in range(1, max_steps + 1):
            step_count = step
            action_dict = agent_step(observation, TASK_ID, step, info)
            action = action_dict.get("action", "read_ticket")
            parameters = action_dict.get("parameters", {})

            # Execute step
            payload = {"action": action, "parameters": parameters}
            step_result = call_env("/step", payload)
            observation = step_result.get("observation", {})
            reward = float(step_result.get("reward", 0.001))
            done = step_result.get("done", False)
            info_data = step_result.get("info", {})

            error_msg = "null"
            if isinstance(info_data, dict):
                err = info_data.get("error", None)
                if err:
                    error_msg = str(err)
                if info_data.get("submission"):
                    submission = info_data["submission"]
                if info_data.get("ground_truth"):
                    ground_truth = info_data["ground_truth"]

            # Clamp reward strictly between 0 and 1
            reward = max(0.001, min(0.999, reward))
            rewards.append(reward)

            # 4. REQUIRED STEP FORMAT
            print(f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={error_msg}")
            sys.stdout.flush()

            if done:
                break

        # 5. Get final score from grader
        if submission and ground_truth:
            score = call_grade(TASK_ID, submission, ground_truth)
        elif rewards:
            score = max(0.001, min(0.999, rewards[-1]))
        else:
            score = 0.001

    except Exception as e:
        print(f"[DEBUG] Main loop error: {e}", file=sys.stderr)
    finally:
        score = max(0.001, min(0.999, float(score)))
        success = score > 0.5
        rewards_str = ",".join([f"{r:.4f}" for r in rewards]) if rewards else "0.0010"

        # 6. REQUIRED END FORMAT
        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.4f} rewards={rewards_str}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
