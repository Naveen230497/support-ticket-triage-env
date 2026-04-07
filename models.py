"""Typed models for the Support Ticket Triage Environment."""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    # Fallback base classes if openenv not installed yet
    class Action:
        pass
    class Observation:
        pass
    class State:
        pass


@dataclass
class TicketAction(Action):
    """Agent action on a support ticket."""
    action_type: str = ""
    value: Optional[str] = None
    confidence: float = 1.0


@dataclass
class TicketObservation(Observation):
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


@dataclass
class TicketState(State):
    """Internal state of the support ticket triage environment."""
    task_id: str = ""
    step_count: int = 0
    episode_id: str = ""
    accumulated_reward: float = 0.0
    issues_fixed: int = 0
    issues_total: int = 0
    last_action_type: str = ""
    is_done: bool = False
