from difflib import SequenceMatcher

# Scores must be strictly between 0.0 and 1.0 (exclusive)
_MIN_SCORE = 0.001
_MAX_SCORE = 0.999


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) - never exactly 0.0 or 1.0."""
    return max(_MIN_SCORE, min(_MAX_SCORE, round(float(score), 4)))


def _similarity(a: str, b: str) -> float:
    """String similarity ratio between 0.0 and 1.0."""
    if not a and not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def grade_easy(submission: dict, ground_truth: dict) -> float:
    """Grade category + priority. 0.5 each. Returns value strictly in (0, 1)."""
    score = 0.0
    gt_category = ground_truth.get("category", "").lower()
    gt_priority = ground_truth.get("priority", "").lower()
    if gt_category and submission.get("category", "").lower() == gt_category:
        score += 0.499
    if gt_priority and submission.get("priority", "").lower() == gt_priority:
        score += 0.499
    return _clamp(score)


def grade_medium(submission: dict, ground_truth: dict) -> float:
    """Grade category + priority + team + sla. 0.25 each. Returns value strictly in (0, 1)."""
    score = 0.0
    fields = [("category", 0.249), ("priority", 0.249), ("team", 0.249), ("sla", 0.249)]
    for field, weight in fields:
        gt_val = ground_truth.get(field, "").lower()
        if gt_val and submission.get(field, "").lower() == gt_val:
            score += weight
    return _clamp(score)


def grade_hard(submission: dict, ground_truth: dict) -> float:
    """Grade all fields. Exact match fields 0.15 each; summary + response use similarity."""
    score = 0.0
    exact_fields = [("category", 0.149), ("priority", 0.149), ("team", 0.149), ("sla", 0.149)]
    for field, weight in exact_fields:
        gt_val = ground_truth.get(field, "").lower()
        if gt_val and submission.get(field, "").lower() == gt_val:
            score += weight
    # Summary similarity (max ~0.2)
    gt_summary = ground_truth.get("summary", "")
    summary_sim = _similarity(submission.get("summary", ""), gt_summary)
    score += summary_sim * 0.199
    # Response similarity (max ~0.2)
    gt_response = ground_truth.get("response", "")
    response_sim = _similarity(submission.get("response", ""), gt_response)
    score += response_sim * 0.199
    return _clamp(score)


def grade(task_id: str, submission: dict, ground_truth: dict) -> float:
    """Grade a submission for a given task. Always returns strictly (0, 1)."""
    if task_id == "easy":
        return grade_easy(submission, ground_truth)
    elif task_id == "medium":
        return grade_medium(submission, ground_truth)
    elif task_id == "hard":
        return grade_hard(submission, ground_truth)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
