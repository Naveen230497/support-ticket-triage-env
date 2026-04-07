"""Core environment logic for Support Ticket Triage."""
import uuid
from typing import Optional, Dict, Any, List

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass

from .tasks import get_task
from .graders import grade

try:
    from ..models import TicketAction, TicketObservation, TicketState
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import TicketAction, TicketObservation, TicketState

VALID_CATEGORIES = {"billing", "technical", "account_access", "product_feedback", "shipping"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_TEAMS = {"billing_team", "tech_support", "account_team", "product_team", "logistics"}


class SupportTicketEnvironment(Environment):
    """Real-world support ticket triage and resolution environment."""

    def __init__(self):
        super().__init__()
        self._state = TicketState()
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_ticket: Optional[Dict[str, Any]] = None
        self._remaining_fixes: List[Dict] = []
        self._duplicate_merged: bool = False
        self._is_escalated: bool = False
        self._task_id: str = "task_easy"

    def reset(self, task_id: str = None) -> TicketObservation:
        if task_id is None:
            task_id = self._task_id
        task = get_task(task_id)
        self._current_task = task
        self._task_id = task_id
        self._current_ticket = dict(task["initial_ticket"])
        self._current_ticket["tags"] = list(task["initial_ticket"].get("tags", []))
        self._remaining_fixes = list(task["required_fixes"])
        self._duplicate_merged = False
        self._is_escalated = False
        self._state = TicketState(
            task_id=task_id,
            step_count=0,
            episode_id=str(uuid.uuid4()),
            accumulated_reward=0.0,
            issues_fixed=0,
            issues_total=task["total_issues"],
            last_action_type="",
            is_done=False,
        )
        return TicketObservation(
            ticket_id=self._current_ticket["id"],
            title=self._current_ticket["title"],
            description=self._current_ticket["description"],
            current_category=self._current_ticket["category"],
            current_priority=self._current_ticket["priority"],
            assigned_team=self._current_ticket["assigned_team"],
            tags=list(self._current_ticket["tags"]),
            resolution_time_hours=self._current_ticket["resolution_time_hours"],
            issues_remaining=len(self._remaining_fixes),
            feedback=f"Episode started. Task: {task['name']}. Fix {task['total_issues']} issue(s).",
            reward=0.0,
            done=False,
            task_id=task_id,
            step_count=0,
            duplicate_ticket=task.get("duplicate_ticket"),
        )

    def step(self, action) -> TicketObservation:
        if self._state.is_done:
            return self._terminal_obs("Episode already done. Call reset() to start a new episode.")
        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        reward, feedback = self._apply_action(action)
        self._state.accumulated_reward += reward
        max_steps = self._current_task["max_steps"]
        episode_complete = len(self._remaining_fixes) == 0 or self._state.step_count >= max_steps
        if episode_complete:
            self._state.is_done = True
            if len(self._remaining_fixes) == 0:
                reward += 0.2
                self._state.accumulated_reward += 0.2
                feedback += " All issues resolved. Bonus +0.20."
        return TicketObservation(
            ticket_id=self._current_ticket["id"],
            title=self._current_ticket["title"],
            description=self._current_ticket["description"],
            current_category=self._current_ticket["category"],
            current_priority=self._current_ticket["priority"],
            assigned_team=self._current_ticket["assigned_team"],
            tags=list(self._current_ticket["tags"]),
            resolution_time_hours=self._current_ticket["resolution_time_hours"],
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=reward,
            done=self._state.is_done,
            task_id=self._task_id,
            step_count=self._state.step_count,
            duplicate_ticket=self._current_task.get("duplicate_ticket"),
        )

    @property
    def state(self) -> TicketState:
        return self._state

    def get_episode_snapshot(self) -> Dict[str, Any]:
        return {
            "current_category": self._current_ticket.get("category", ""),
            "current_priority": self._current_ticket.get("priority", ""),
            "assigned_team": self._current_ticket.get("assigned_team", ""),
            "tags": list(self._current_ticket.get("tags", [])),
            "resolution_time_hours": self._current_ticket.get("resolution_time_hours", 0.0),
            "duplicate_merged": self._duplicate_merged,
            "is_escalated": self._is_escalated,
        }

    def _apply_action(self, action):
        action_type = action.action_type
        value = (action.value or "").strip().lower()

        if action_type == "set_category":
            if value not in VALID_CATEGORIES:
                return -0.05, f"Invalid category '{value}'. Valid: {sorted(VALID_CATEGORIES)}"
            self._current_ticket["category"] = value
            return self._check_fix("set_category", value)

        elif action_type == "set_priority":
            if value not in VALID_PRIORITIES:
                return -0.05, f"Invalid priority '{value}'. Valid: {sorted(VALID_PRIORITIES)}"
            self._current_ticket["priority"] = value
            return self._check_fix("set_priority", value)

        elif action_type == "assign_team":
            if value not in VALID_TEAMS:
                return -0.05, f"Invalid team '{value}'. Valid: {sorted(VALID_TEAMS)}"
            self._current_ticket["assigned_team"] = value
            return self._check_fix("assign_team", value)

        elif action_type == "add_tag":
            if not value:
                return -0.05, "add_tag requires a value."
            self._current_ticket["tags"].append(value)
            return self._check_fix("add_tag", value)

        elif action_type == "set_resolution_time":
            try:
                hours = float(value)
            except (ValueError, TypeError):
                return -0.05, f"set_resolution_time requires a numeric value, got '{value}'."
            sla = self._current_task.get("sla_hours", 24.0)
            self._current_ticket["resolution_time_hours"] = hours
            if hours <= 0:
                return -0.10, f"Resolution time must be > 0. Got {hours}."
            if hours > sla:
                return -0.10, f"Resolution time {hours}h exceeds SLA of {sla}h. -0.10"
            return self._check_fix("set_resolution_time", value)

        elif action_type == "merge_duplicate":
            dup = self._current_task.get("duplicate_ticket")
            if not dup:
                return -0.05, "No duplicate ticket to merge."
            self._duplicate_merged = True
            new_remaining = [f for f in self._remaining_fixes if f.get("type") != "merge_duplicate"]
            if len(new_remaining) < len(self._remaining_fixes):
                self._remaining_fixes = new_remaining
                self._state.issues_fixed += 1
                return 0.30, "Duplicate ticket merged successfully. +0.30"
            return 0.05, "Merged duplicate (no pending fix for this)."

        elif action_type == "escalate":
            self._is_escalated = True
            new_remaining = [f for f in self._remaining_fixes if f.get("type") != "escalate"]
            if len(new_remaining) < len(self._remaining_fixes):
                self._remaining_fixes = new_remaining
                self._state.issues_fixed += 1
                return 0.30, "Ticket escalated to senior support. +0.30"
            return 0.05, "Ticket escalated (no pending escalation fix)."

        elif action_type == "mark_resolved":
            if len(self._remaining_fixes) == 0:
                self._state.is_done = True
                return 0.10, "All issues resolved. Marking ticket as resolved."
            return -0.10, f"Cannot mark resolved: {len(self._remaining_fixes)} issue(s) remain."

        return -0.10, f"Unknown action_type: '{action_type}'."

    def _check_fix(self, fix_type: str, value: str):
        new_remaining = []
        fixed = False
        for fix in self._remaining_fixes:
            if fix.get("type") != fix_type:
                new_remaining.append(fix)
                continue
            expected = fix.get("expected", "")
            if expected and value.lower() == expected.lower():
                fixed = True
            elif fix_type == "set_resolution_time":
                try:
                    hours = float(value)
                    sla_max = fix.get("sla_max", 24.0)
                    if 0 < hours <= sla_max:
                        fixed = True
                    else:
                        new_remaining.append(fix)
                except ValueError:
                    new_remaining.append(fix)
            else:
                new_remaining.append(fix)
        if fixed:
            self._remaining_fixes = new_remaining
            self._state.issues_fixed += 1
            return 0.30, f"Action '{fix_type}'='{value}' resolves an issue. +0.30"
        return 0.05, f"Action '{fix_type}'='{value}' applied (no matching required fix). +0.05"

    def _terminal_obs(self, feedback: str) -> TicketObservation:
        return TicketObservation(
            ticket_id=self._current_ticket.get("id", ""),
            title=self._current_ticket.get("title", ""),
            description=self._current_ticket.get("description", ""),
            current_category=self._current_ticket.get("category", ""),
            current_priority=self._current_ticket.get("priority", ""),
            assigned_team=self._current_ticket.get("assigned_team", ""),
            tags=list(self._current_ticket.get("tags", [])),
            resolution_time_hours=self._current_ticket.get("resolution_time_hours", 0.0),
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=0.0,
            done=True,
            task_id=self._task_id,
            step_count=self._state.step_count,
        )
