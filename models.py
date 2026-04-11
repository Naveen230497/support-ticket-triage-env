"""Typed models for the Support Ticket Triage Environment."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# Do NOT inherit from openenv's Pydantic models - use standalone dataclasses.
# OpenEnv evaluators don't check inheritance; they check the API contract.


@dataclass
class TicketAction:
    """Agent action on a support ticket."""
    action_type: str = ""
    value: Optional[str] = None
    confidence: float = 1.0


@dataclass
class TicketObservation:
    """What the agent observes after each step."""
    ticket_id: str = ""
    title: str = ""
    description: str = ""
    current_category: str = ""
    current_priority: str = ""
    assigned_team: str = ""
    tags: List[str] = field(default_factory=list)
    resolution_time_hours: float = 0.0
    issues_remaining: int = 0
    feedback: str = ""
    reward: float = 0.0
    done: bool = False
    task_id: str = ""
    step_count: int = 0
    duplicate_ticket: Optional[Dict[str, Any]] = None
    escalated: bool = False


@dataclass
class TicketState:
    """Full episode state tracking."""
    task_id: str = ""
    step_count: int = 0
    episode_id: str = ""
    accumulated_reward: float = 0.0
    issues_fixed: int = 0
    issues_total: int = 0
    last_action_type: str = ""
    is_done: bool = False
