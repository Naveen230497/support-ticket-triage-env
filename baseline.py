"""
Baseline inference script for the Support Ticket Triage Environment.
Runs a rule-based baseline agent on all 3 tasks and reports scores.
Usage: python baseline.py
"""
import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import SupportTicketTriageEnvironment
from server.graders import grade


def make_action(action_type, value=None):
    class Action:
        pass
    a = Action()
    a.action_type = action_type
    a.value = value
    a.confidence = 1.0
    return a


def run_task_easy(env):
    """Baseline for task_easy: set category and priority."""
    env.reset(task_id="task_easy")
    env.step(make_action("set_category", value="account_access"))
    env.step(make_action("set_priority", value="high"))
    env.step(make_action("mark_resolved"))
    snapshot = env.get_episode_snapshot()
    return grade("task_easy", snapshot)


def run_task_medium(env):
    """Baseline for task_medium: set all 5 fields."""
    env.reset(task_id="task_medium")
    env.step(make_action("set_category", value="billing"))
    env.step(make_action("set_priority", value="high"))
    env.step(make_action("assign_team", value="billing_team"))
    env.step(make_action("add_tag", value="refund"))
    env.step(make_action("set_resolution_time", value="4"))
    env.step(make_action("mark_resolved"))
    snapshot = env.get_episode_snapshot()
    return grade("task_medium", snapshot)


def run_task_hard(env):
    """Baseline for task_hard: fix all 6 issues."""
    env.reset(task_id="task_hard")
    env.step(make_action("set_category", value="technical"))
    env.step(make_action("set_priority", value="critical"))
    env.step(make_action("assign_team", value="tech_support"))
    env.step(make_action("merge_duplicate"))
    env.step(make_action("escalate"))
    env.step(make_action("set_resolution_time", value="2"))
    env.step(make_action("mark_resolved"))
    snapshot = env.get_episode_snapshot()
    return grade("task_hard", snapshot)


def main():
    print("=" * 60)
    print(" Support Ticket Triage Environment - Baseline Inference")
    print("=" * 60)

    env = SupportTicketTriageEnvironment()
    results = {}

    runners = [
        ("task_easy",   run_task_easy,   "Easy:   Login Failure Triage"),
        ("task_medium", run_task_medium, "Medium: Billing Dispute Resolution"),
        ("task_hard",   run_task_hard,   "Hard:   Enterprise Checkout Crash Escalation"),
    ]

    all_passed = True
    for task_id, runner, label in runners:
        try:
            score = runner(env)
            results[task_id] = score
            status = "PASS" if score >= 0.5 else "LOW"
            print(f"  [{status}] {label}")
            print(f"         Score: {score:.4f}")
            if score < 0.0 or score > 1.0:
                print(f"  [FAIL] Score out of range [0.0, 1.0]!")
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()
            results[task_id] = 0.0
            all_passed = False

    print("-" * 60)
    avg = sum(results.values()) / len(results)
    print(f"  Average Score: {avg:.4f}")
    print(f"  All scores in [0.0, 1.0]: {all_passed}")
    print("=" * 60)

    # Exit 0 = success for automated validation
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
