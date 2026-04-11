from typing import Dict, Any

SCORE_MIN = 0.05
SCORE_MAX = 0.95


def clamp(score: float) -> float:
    return max(SCORE_MIN, min(SCORE_MAX, round(float(score), 4)))


def grade_easy(episode: Dict[str, Any]) -> float:
    """Score based on whether category and priority are correctly set.
    0.5 per correct field (max 1.0).
    """
    ticket = episode.get("current_ticket", {})
    score = 0.0

    if ticket.get("current_category", "").strip().lower() == "account_access":
        score += 0.5
    if ticket.get("current_priority", "").strip().lower() == "high":
        score += 0.5

    return clamp(score)


def grade_medium(episode: Dict[str, Any]) -> float:
    """Score based on 5 fields. 0.20 per correct field (max 1.0)."""
    ticket = episode.get("current_ticket", {})
    score = 0.0

    if ticket.get("current_category", "").strip().lower() == "billing":
        score += 0.20
    if ticket.get("current_priority", "").strip().lower() == "high":
        score += 0.20
    if ticket.get("assigned_team", "").strip().lower() == "billing_team":
        score += 0.20
    if "refund" in [t.lower() for t in ticket.get("tags", [])]:
        score += 0.20
    res_time = ticket.get("resolution_time_hours", 0.0)
    if 0 < res_time <= 8.0:
        score += 0.20

    return clamp(score)


def grade_hard(episode: Dict[str, Any]) -> float:
    """Score based on 6 issues. 1/6 per resolved issue (max 1.0)."""
    ticket = episode.get("current_ticket", {})
    merged = episode.get("duplicate_merged", False)
    escalated = ticket.get("escalated", False)

    weight = 1.0 / 6.0
    score = 0.0

    if ticket.get("current_category", "").strip().lower() == "technical":
        score += weight
    if ticket.get("current_priority", "").strip().lower() == "critical":
        score += weight
    if ticket.get("assigned_team", "").strip().lower() == "tech_support":
        score += weight
    if merged:
        score += weight
    if escalated:
        score += weight
    res_time = ticket.get("resolution_time_hours", 0.0)
    if 0 < res_time <= 4.0:
        score += weight

    return clamp(score)


GRADERS: Dict[str, Any] = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}


def grade(task_id: str, episode: Dict[str, Any]) -> float:
    """Dispatch grading to the correct task grader."""
    grader_fn = GRADERS.get(task_id)
    if grader_fn is None:
        raise ValueError(f"No grader registered for task_id: {task_id!r}")
    return grader_fn(episode)
