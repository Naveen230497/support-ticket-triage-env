import uuid
import sys
import os
from typing import Optional, Dict, Any

from .tasks import get_task
from .graders import grade

# Import models with fallback for both package and standalone imports
try:
    from models import TicketAction, TicketObservation, TicketState
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import TicketAction, TicketObservation, TicketState


class SupportTicketTriageEnvironment:
    """Support ticket triage environment for OpenEnv."""

    def __init__(self):
        self._state = TicketState()
        self._current_task: Optional[Dict[str, Any]] = None
        self._current_ticket: Optional[Dict[str, Any]] = None
        self._duplicate_ticket: Optional[Dict[str, Any]] = None
        self._remaining_fixes: list = []
        self._duplicate_merged: bool = False
        self._task_id: str = "task_easy"

    def reset(self, task_id: str = None) -> TicketObservation:
        if task_id is None:
            task_id = self._task_id

        task = get_task(task_id)
        self._current_task = task
        self._task_id = task_id
        self._current_ticket = dict(task["initial_ticket"])
        self._current_ticket["tags"] = list(task["initial_ticket"].get("tags", []))
        self._duplicate_ticket = task.get("duplicate_ticket")
        self._remaining_fixes = list(task["required_fixes"])
        self._duplicate_merged = False
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
            current_category=self._current_ticket.get("current_category", ""),
            current_priority=self._current_ticket.get("current_priority", ""),
            assigned_team=self._current_ticket.get("assigned_team", ""),
            tags=list(self._current_ticket.get("tags", [])),
            resolution_time_hours=self._current_ticket.get("resolution_time_hours", 0.0),
            issues_remaining=len(self._remaining_fixes),
            feedback=f"Episode started. Task: {task['name']}. Fix {task['total_issues']} issue(s).",
            reward=0.0,
            done=False,
            task_id=task_id,
            step_count=0,
            duplicate_ticket=self._duplicate_ticket,
            escalated=self._current_ticket.get("escalated", False),
        )

    def step(self, action: TicketAction) -> TicketObservation:
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
            current_category=self._current_ticket.get("current_category", ""),
            current_priority=self._current_ticket.get("current_priority", ""),
            assigned_team=self._current_ticket.get("assigned_team", ""),
            tags=list(self._current_ticket.get("tags", [])),
            resolution_time_hours=self._current_ticket.get("resolution_time_hours", 0.0),
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=reward,
            done=self._state.is_done,
            task_id=self._task_id,
            step_count=self._state.step_count,
            duplicate_ticket=self._duplicate_ticket,
            escalated=self._current_ticket.get("escalated", False),
        )

    @property
    def state(self) -> TicketState:
        return self._state

    def get_episode_snapshot(self) -> Dict[str, Any]:
        return {
            "current_ticket": dict(self._current_ticket),
            "duplicate_merged": self._duplicate_merged,
        }

    def _apply_action(self, action: TicketAction):
        ticket = self._current_ticket

        if action.action_type == "set_category":
            if not action.value:
                return -0.05, "set_category requires a value."
            ticket["current_category"] = action.value
            return self._check_fix("set_category", action.value)

        elif action.action_type == "set_priority":
            if not action.value:
                return -0.05, "set_priority requires a value."
            ticket["current_priority"] = action.value
            return self._check_fix("set_priority", action.value)

        elif action.action_type == "assign_team":
            if not action.value:
                return -0.05, "assign_team requires a value."
            ticket["assigned_team"] = action.value
            return self._check_fix("assign_team", action.value)

        elif action.action_type == "add_tag":
            if not action.value:
                return -0.05, "add_tag requires a value."
            if action.value not in ticket["tags"]:
                ticket["tags"].append(action.value)
            return self._check_fix("add_tag", action.value)

        elif action.action_type == "set_resolution_time":
            if not action.value:
                return -0.05, "set_resolution_time requires a value (hours)."
            try:
                hours = float(action.value)
            except (ValueError, TypeError):
                return -0.05, f"Invalid resolution time: '{action.value}'. Must be a number."
            ticket["resolution_time_hours"] = hours
            return self._check_fix_resolution_time(hours)

        elif action.action_type == "merge_duplicate":
            if self._duplicate_ticket:
                # Inherit useful fields from duplicate
                dup = self._duplicate_ticket
                for tag in dup.get("tags", []):
                    if tag not in ticket["tags"]:
                        ticket["tags"].append(tag)
                self._duplicate_merged = True
                self._remaining_fixes = [
                    f for f in self._remaining_fixes
                    if f.get("type") != "merge_duplicate"
                ]
                self._state.issues_fixed += 1
                return 0.2, "Merged duplicate ticket. Tags inherited."
            return -0.05, "No duplicate ticket to merge."

        elif action.action_type == "escalate":
            ticket["escalated"] = True
            fixed = False
            new_remaining = []
            for fix in self._remaining_fixes:
                if fix["type"] == "escalate":
                    fixed = True
                else:
                    new_remaining.append(fix)
            if fixed:
                self._remaining_fixes = new_remaining
                self._state.issues_fixed += 1
                return 0.3, "Ticket escalated. +0.30"
            return 0.05, "Ticket escalated (no matching required fix)."

        elif action.action_type == "mark_resolved":
            if len(self._remaining_fixes) == 0:
                self._state.is_done = True
                return 0.1, "All issues resolved. Marking complete."
            return -0.1, f"Cannot mark resolved: {len(self._remaining_fixes)} issue(s) remain."

        return -0.1, f"Unknown action_type: '{action.action_type}'."

    def _check_fix(self, fix_type: str, value: str):
        fixed = False
        new_remaining = []
        for fix in self._remaining_fixes:
            if fix["type"] != fix_type:
                new_remaining.append(fix)
                continue
            if fix.get("expected") and value.strip().lower() == fix["expected"].lower():
                fixed = True
            else:
                new_remaining.append(fix)
        if fixed:
            self._remaining_fixes = new_remaining
            self._state.issues_fixed += 1
            return 0.3, f"'{fix_type}'='{value}' resolves an issue. +0.30"
        return 0.05, f"'{fix_type}'='{value}' set (no matching required fix)."

    def _check_fix_resolution_time(self, hours: float):
        fixed = False
        new_remaining = []
        for fix in self._remaining_fixes:
            if fix["type"] != "set_resolution_time":
                new_remaining.append(fix)
                continue
            max_hours = fix.get("expected_max", 24.0)
            if 0 < hours <= max_hours:
                fixed = True
            else:
                new_remaining.append(fix)
        if fixed:
            self._remaining_fixes = new_remaining
            self._state.issues_fixed += 1
            return 0.3, f"Resolution time {hours}h accepted. +0.30"
        return -0.1, f"Resolution time {hours}h not in expected range."

    def _terminal_obs(self, feedback: str) -> TicketObservation:
        return TicketObservation(
            ticket_id=self._current_ticket.get("id", ""),
            title=self._current_ticket.get("title", ""),
            description=self._current_ticket.get("description", ""),
            current_category=self._current_ticket.get("current_category", ""),
            current_priority=self._current_ticket.get("current_priority", ""),
            assigned_team=self._current_ticket.get("assigned_team", ""),
            tags=list(self._current_ticket.get("tags", [])),
            resolution_time_hours=self._current_ticket.get("resolution_time_hours", 0.0),
            issues_remaining=len(self._remaining_fixes),
            feedback=feedback,
            reward=0.0,
            done=True,
            task_id=self._task_id,
            step_count=self._state.step_count,
            escalated=self._current_ticket.get("escalated", False),
        )
