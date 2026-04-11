import copy
from typing import Dict, Any, List

TASKS: Dict[str, Dict[str, Any]] = {

    "task_easy": {
        "id": "task_easy",
        "name": "Login Failure Triage",
        "difficulty": "easy",
        "description": "A user cannot log in and gets 'Invalid credentials' error. Category and priority are missing. Fix them.",
        "max_steps": 8,
        "initial_ticket": {
            "id": "TKT-1001",
            "title": "Cannot login to my account",
            "description": "I have been trying to login for the past 2 hours but keep getting 'Invalid credentials' error even though I just reset my password. Locked out of dashboard completely.",
            "current_category": "",
            "current_priority": "",
            "assigned_team": "",
            "tags": [],
            "resolution_time_hours": 0.0,
            "escalated": False,
        },
        "required_fixes": [
            {"type": "set_category", "expected": "account_access"},
            {"type": "set_priority", "expected": "high"},
        ],
        "total_issues": 2,
    },

    "task_medium": {
        "id": "task_medium",
        "name": "Billing Dispute Resolution",
        "difficulty": "medium",
        "description": "Customer was charged twice for subscription and needs a refund. Five fields are missing: category, priority, team, tag, and resolution time.",
        "max_steps": 20,
        "initial_ticket": {
            "id": "TKT-2002",
            "title": "Charged twice for monthly subscription",
            "description": "I was charged $99.99 twice on my credit card for the same monthly subscription. I need a refund for the duplicate charge. Transaction IDs: TXN-55812 and TXN-55813. This happened on 2024-03-15.",
            "current_category": "",
            "current_priority": "",
            "assigned_team": "",
            "tags": [],
            "resolution_time_hours": 0.0,
            "escalated": False,
        },
        "required_fixes": [
            {"type": "set_category",        "expected": "billing"},
            {"type": "set_priority",        "expected": "high"},
            {"type": "assign_team",         "expected": "billing_team"},
            {"type": "add_tag",             "expected": "refund"},
            {"type": "set_resolution_time", "expected_max": 8.0},
        ],
        "total_issues": 5,
    },

    "task_hard": {
        "id": "task_hard",
        "name": "Enterprise Checkout Crash Escalation",
        "difficulty": "hard",
        "description": (
            "Enterprise checkout crashes for 500+ users. The ticket has WRONG pre-filled values: "
            "category is 'product_feedback' (should be 'technical'), priority is 'low' (should be 'critical'). "
            "No team assigned, there is an unmerged duplicate ticket, no escalation, and no SLA time set. "
            "Fix all 6 issues."
        ),
        "max_steps": 30,
        "initial_ticket": {
            "id": "TKT-3003",
            "title": "Checkout page crashes for enterprise customers",
            "description": (
                "URGENT: Our enterprise checkout page throws a 500 error for all users since 03:00 UTC. "
                "Over 500 customers affected. Revenue impact estimated at $50,000/hour. "
                "Stack trace shows NullPointerException in PaymentGateway.processOrder(). "
                "Multiple enterprise clients (Acme Corp, GlobalTech, MegaRetail) have escalated to their account managers."
            ),
            "current_category": "product_feedback",
            "current_priority": "low",
            "assigned_team": "",
            "tags": [],
            "resolution_time_hours": 0.0,
            "escalated": False,
        },
        "duplicate_ticket": {
            "id": "TKT-3004",
            "title": "Payment processing broken on checkout",
            "description": "Checkout is failing with error 500. Same issue as TKT-3003.",
            "current_category": "technical",
            "current_priority": "critical",
            "assigned_team": "tech_support",
            "tags": ["outage", "payment"],
            "resolution_time_hours": 2.0,
            "escalated": True,
        },
        "required_fixes": [
            {"type": "set_category",        "expected": "technical"},
            {"type": "set_priority",        "expected": "critical"},
            {"type": "assign_team",         "expected": "tech_support"},
            {"type": "merge_duplicate"},
            {"type": "escalate"},
            {"type": "set_resolution_time", "expected_max": 4.0},
        ],
        "total_issues": 6,
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}")
    return copy.deepcopy(TASKS[task_id])


def list_tasks() -> List[Dict[str, Any]]:
    action_schema = {
        "action_type": {
            "type": "string",
            "enum": [
                "set_category", "set_priority", "assign_team",
                "add_tag", "set_resolution_time", "merge_duplicate",
                "escalate", "mark_resolved",
            ],
        },
        "value": {"type": "string", "required": False},
        "confidence": {"type": "number", "min": 0.0, "max": 1.0},
    }
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "action_schema": action_schema,
        }
        for t in TASKS.values()
    ]
