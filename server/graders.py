"""Deterministic graders for each task in the Support Ticket Triage Environment."""
from typing import Dict, Any

VALID_CATEGORIES = {"billing", "technical", "account_access", "product_feedback", "shipping"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_TEAMS = {"billing_team", "tech_support", "account_team", "product_team", "logistics"}


def grade(task_id: str, snapshot: Dict[str, Any]) -> float:
    """Grade the current episode snapshot for a given task. Returns score in [0.0, 1.0]."""
    if task_id == "task_easy":
        return _grade_easy(snapshot)
    elif task_id == "task_medium":
        return _grade_medium(snapshot)
    elif task_id == "task_hard":
        return _grade_hard(snapshot)
    raise ValueError(f"Unknown task_id: {task_id}")


def _grade_easy(snapshot: Dict[str, Any]) -> float:
    """task_easy: 0.5 per correct field (category + priority). Max = 1.0."""
    score = 0.0
    if snapshot.get("current_category") == "account_access":
        score += 0.5
    if snapshot.get("current_priority") == "high":
        score += 0.5
    return round(score, 3)


def _grade_medium(snapshot: Dict[str, Any]) -> float:
    """task_medium: 0.20 per correct field (5 fields). Max = 1.0."""
    score = 0.0
    if snapshot.get("current_category") == "billing":
        score += 0.20
    if snapshot.get("current_priority") == "high":
        score += 0.20
    if snapshot.get("assigned_team") == "billing_team":
        score += 0.20
    tags = snapshot.get("tags", [])
    if "refund" in [t.lower() for t in tags]:
        score += 0.20
    rt = snapshot.get("resolution_time_hours", 0.0)
    if 0 < rt <= 8.0:
        score += 0.20
    return round(score, 3)


def _grade_hard(snapshot: Dict[str, Any]) -> float:
    """task_hard: 1/6 per resolved issue (6 issues). Max = 1.0."""
    score = 0.0
    per_issue = round(1.0 / 6, 6)
    if snapshot.get("current_category") == "technical":
        score += per_issue
    if snapshot.get("current_priority") == "critical":
        score += per_issue
    if snapshot.get("assigned_team") == "tech_support":
        score += per_issue
    if snapshot.get("duplicate_merged", False):
        score += per_issue
    if snapshot.get("is_escalated", False):
        score += per_issue
    rt = snapshot.get("resolution_time_hours", 0.0)
    if 0 < rt <= 2.0:
        score += per_issue
    return round(min(score, 1.0), 3)
