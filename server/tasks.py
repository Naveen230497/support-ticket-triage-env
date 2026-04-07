"""Task definitions for the Support Ticket Triage Environment."""
import copy
from typing import Dict, Any, List

TASKS: Dict[str, Dict[str, Any]] = {

    "task_easy": {
        "id": "task_easy",
        "name": "Login Failure Triage",
        "difficulty": "easy",
        "description": "A user cannot log in. The ticket is missing category and priority. Set them correctly.",
        "max_steps": 8,
        "initial_ticket": {
            "id": "TKT-1001",
            "title": "Cannot login to my account",
            "description": "I have been trying to login since morning but keep getting an error: Invalid credentials. Please help.",
            "category": "",
            "priority": "",
            "assigned_team": "",
            "tags": [],
            "resolution_time_hours": 0.0,
            "is_escalated": False,
        },
        "required_fixes": [
            {"type": "set_category", "expected": "account_access"},
            {"type": "set_priority", "expected": "high"},
        ],
        "total_issues": 2,
        "sla_hours": 4.0,
        "duplicate_ticket": None,
    },

    "task_medium": {
        "id": "task_medium",
        "name": "Billing Dispute Resolution",
        "difficulty": "medium",
        "description": "A customer was charged incorrectly. The ticket is missing category, priority, team, a tag, and resolution time.",
        "max_steps": 20,
        "initial_ticket": {
            "id": "TKT-2042",
            "title": "Charged twice for subscription",
            "description": "I was billed twice this month for my Pro subscription. Please refund the duplicate charge immediately. My order ID is ORD-8823.",
            "category": "",
            "priority": "",
            "assigned_team": "",
            "tags": [],
            "resolution_time_hours": 0.0,
            "is_escalated": False,
        },
        "required_fixes": [
            {"type": "set_category", "expected": "billing"},
            {"type": "set_priority", "expected": "high"},
            {"type": "assign_team", "expected": "billing_team"},
            {"type": "add_tag", "expected": "refund"},
            {"type": "set_resolution_time", "sla_max": 8.0},
        ],
        "total_issues": 5,
        "sla_hours": 8.0,
        "duplicate_ticket": None,
    },

    "task_hard": {
        "id": "task_hard",
        "name": "Enterprise Checkout Crash Escalation",
        "difficulty": "hard",
        "description": "An enterprise customer reports a critical checkout crash. The ticket has WRONG pre-filled values, a duplicate, and needs escalation.",
        "max_steps": 30,
        "initial_ticket": {
            "id": "TKT-3099",
            "title": "Checkout crashes on payment step - blocking 500 users",
            "description": "Our entire company cannot complete purchases. The payment page throws a 500 error. This is blocking revenue. Urgent fix needed.",
            "category": "product_feedback",
            "priority": "low",
            "assigned_team": "",
            "tags": [],
            "resolution_time_hours": 0.0,
            "is_escalated": False,
        },
        "required_fixes": [
            {"type": "set_category", "expected": "technical"},
            {"type": "set_priority", "expected": "critical"},
            {"type": "assign_team", "expected": "tech_support"},
            {"type": "merge_duplicate"},
            {"type": "escalate"},
            {"type": "set_resolution_time", "sla_max": 2.0},
        ],
        "total_issues": 6,
        "sla_hours": 2.0,
        "duplicate_ticket": {
            "id": "TKT-3100",
            "title": "500 error on checkout - same issue",
            "description": "Duplicate of TKT-3099. Same 500 error on checkout page.",
            "category": "technical",
            "priority": "critical",
        },
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")
    return copy.deepcopy(TASKS[task_id])


def list_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "total_issues": t["total_issues"],
            "action_types": [
                "set_category", "set_priority", "assign_team",
                "add_tag", "set_resolution_time", "merge_duplicate",
                "escalate", "mark_resolved"
            ],
            "valid_categories": ["billing", "technical", "account_access", "product_feedback", "shipping"],
            "valid_priorities": ["low", "medium", "high", "critical"],
            "valid_teams": ["billing_team", "tech_support", "account_team", "product_team", "logistics"],
        }
        for t in TASKS.values()
    ]
