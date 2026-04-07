"""Deterministic baseline agent for Support Ticket Triage Environment.

Runs a rule-based agent on all 3 tasks and reports reproducible scores.
Usage:
    # Start server first
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    # Then run baseline
    python baseline.py

Expected output:
    task_easy:   score=1.0
    task_medium: score=1.0
    task_hard:   score=1.0
"""
import requests
import sys

BASE_URL = "http://localhost:7860"

BASELINE_PLANS = {
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


def run_task(task_id: str, plan: list) -> float:
    # Reset
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    print(f"  Reset: task={task_id}")

    # Execute plan
    for step_num, action in enumerate(plan, 1):
        payload = {"action_type": action["action_type"],
                   "value": action.get("value"),
                   "confidence": 1.0}
        r = requests.post(f"{BASE_URL}/step", json=payload)
        r.raise_for_status()
        data = r.json()
        obs = data.get("observation", {})
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        print(f"  Step {step_num}: {action['action_type']}({action.get('value', '')}) -> reward={reward:.2f} done={done} | {obs.get('feedback', '')}")
        if done:
            break

    # Grade
    r = requests.post(f"{BASE_URL}/grader", json={"task_id": task_id})
    r.raise_for_status()
    grade_data = r.json()
    score = grade_data.get("score", 0.0)
    return score


def main():
    print("\n=== Support Ticket Triage Environment — Baseline Agent ===\n")
    results = {}
    all_passed = True

    for task_id, plan in BASELINE_PLANS.items():
        print(f"[Task: {task_id}]")
        try:
            score = run_task(task_id, plan)
            results[task_id] = score
            status = "PASS" if score >= 0.99 else "PARTIAL"
            print(f"  => Score: {score:.3f} [{status}]\n")
            if score < 0.99:
                all_passed = False
        except Exception as e:
            print(f"  => ERROR: {e}\n")
            results[task_id] = 0.0
            all_passed = False

    print("=== Summary ===")
    for task_id, score in results.items():
        print(f"  {task_id}: {score:.3f}")
    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"  Average: {avg:.3f}")
    print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
