import random
from typing import Optional, List, Dict, Any

TICKETS = [
    {
        "id": "T001",
        "subject": "Cannot login to my account",
        "body": "I have been trying to login for the past 2 hours but keep getting invalid password error even though I reset it.",
        "category": "authentication",
        "priority": "high",
        "team": "identity",
        "sla": "P1",
        "summary": "User unable to login after password reset",
        "response": "We apologize for the inconvenience. Our identity team is investigating your login issue and will resolve it within 2 hours."
    },
    {
        "id": "T002",
        "subject": "Billing charge incorrect",
        "body": "I was charged $150 instead of $50 for my monthly subscription. Please refund the extra amount.",
        "category": "billing",
        "priority": "high",
        "team": "finance",
        "sla": "P1",
        "summary": "User overcharged $100 for monthly subscription",
        "response": "We sincerely apologize for the billing error. Our finance team will process a refund of $100 within 3-5 business days."
    },
    {
        "id": "T003",
        "subject": "App crashes on startup",
        "body": "The mobile app crashes immediately after opening. This started after the last update.",
        "category": "bug",
        "priority": "medium",
        "team": "mobile",
        "sla": "P2",
        "summary": "App crashes on startup after recent update",
        "response": "Thank you for reporting this. Our mobile team is aware of the crash issue in the latest update and is working on a fix."
    },
    {
        "id": "T004",
        "subject": "How do I export my data?",
        "body": "I need to export all my data to CSV format. I cannot find the option in settings.",
        "category": "how-to",
        "priority": "low",
        "team": "support",
        "sla": "P3",
        "summary": "User needs guidance on data export to CSV",
        "response": "You can export your data by going to Settings > Data Management > Export. Select CSV format and click Download."
    },
    {
        "id": "T005",
        "subject": "Integration with Slack not working",
        "body": "The Slack integration stopped sending notifications 2 days ago. Our webhook is configured correctly.",
        "category": "integration",
        "priority": "medium",
        "team": "integrations",
        "sla": "P2",
        "summary": "Slack webhook integration not sending notifications",
        "response": "We have identified an issue with Slack webhook notifications. Our integrations team is working on a fix expected within 24 hours."
    },
]


def get_task_config(task_id: str, seed: int = 42) -> dict:
    rng = random.Random(seed)
    ticket = rng.choice(TICKETS)
    if task_id == "easy":
        return {
            "task_id": task_id,
            "ticket": ticket,
            "required_fields": ["category", "priority"],
            "description": "Classify this support ticket by category and priority.",
            "max_steps": 5,
        }
    elif task_id == "medium":
        return {
            "task_id": task_id,
            "ticket": ticket,
            "required_fields": ["category", "priority", "team", "sla"],
            "description": "Classify this ticket, assign to the correct team and SLA tier.",
            "max_steps": 8,
        }
    elif task_id == "hard":
        return {
            "task_id": task_id,
            "ticket": ticket,
            "required_fields": ["category", "priority", "team", "sla", "summary", "response"],
            "description": "Fully triage: classify, route, summarize and draft an initial response.",
            "max_steps": 12,
        }
    else:
        raise ValueError(f"Unknown task_id: {task_id}")


ACTION_SCHEMA = {
    "action": {
        "type": "string",
        "enum": ["read_ticket", "set_field", "submit"]
    },
    "parameters": {
        "type": "object",
        "required": False
    },
}


def list_tasks() -> List[Dict[str, Any]]:
    """Return list of all tasks with action schema (reference-compatible format)."""
    return [
        {
            "id": "easy",
            "name": "Basic Ticket Classification",
            "difficulty": "easy",
            "max_steps": 5,
            "description": "Classify support ticket by category and priority",
            "action_schema": ACTION_SCHEMA,
        },
        {
            "id": "medium",
            "name": "Ticket Routing with SLA",
            "difficulty": "medium",
            "max_steps": 8,
            "description": "Route ticket to correct team and assign SLA tier",
            "action_schema": ACTION_SCHEMA,
        },
        {
            "id": "hard",
            "name": "Full Triage with Resolution",
            "difficulty": "hard",
            "max_steps": 12,
            "description": "Classify, route, summarize, and draft initial response",
            "action_schema": ACTION_SCHEMA,
        },
    ]


# TASK_LIST for backwards compatibility
TASK_LIST = list_tasks()
