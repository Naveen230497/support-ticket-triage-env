from typing import Optional
from server.tasks import get_task_config
from server.graders import grade


class TicketTriageEnvironment:
    def __init__(self):
        self.task_config = None
        self.current_step = 0
        self.done = False
        self.submission = {}
        self.episode_reward = 0.0

    def reset(self, task_id: str, seed: int = 42) -> dict:
        self.task_config = get_task_config(task_id, seed)
        self.current_step = 0
        self.done = False
        self.submission = {}
        self.episode_reward = 0.0

        ticket = self.task_config["ticket"]
        observation = (
            f"SUPPORT TICKET\n"
            f"ID: {ticket['id']}\n"
            f"Subject: {ticket['subject']}\n"
            f"Body: {ticket['body']}\n"
            f"\nTask: {self.task_config['description']}\n"
            f"Required fields: {', '.join(self.task_config['required_fields'])}\n"
            f"Max steps: {self.task_config['max_steps']}"
        )
        return {
            "observation": observation,
            "info": {
                "task_id": task_id,
                "ticket_id": ticket["id"],
                "required_fields": self.task_config["required_fields"],
                "max_steps": self.task_config["max_steps"],
            },
        }

    def step(self, action: str, parameters: dict) -> dict:
        if self.done:
            return {
                "observation": "Episode is already done. Call /reset to start a new episode.",
                "reward": 0.0,
                "done": True,
                "info": {"error": "episode_done"},
            }

        self.current_step += 1
        reward = 0.0
        done = False
        observation = ""

        if action == "submit":
            for key, val in parameters.items():
                self.submission[key] = val
            reward = grade(
                self.task_config["task_id"],
                self.submission,
                self.task_config["ticket"],
            )
            self.episode_reward = reward
            done = True
            observation = f"Submission received. Score: {reward:.4f}"

        elif action == "set_field":
            field = parameters.get("field", "")
            value = parameters.get("value", "")
            self.submission[field] = value
            reward = grade(
                self.task_config["task_id"],
                self.submission,
                self.task_config["ticket"],
            )
            observation = f"Field '{field}' set to '{value}'. Current partial score: {reward:.4f}"

        elif action == "read_ticket":
            ticket = self.task_config["ticket"]
            observation = (
                f"Ticket ID: {ticket['id']}\n"
                f"Subject: {ticket['subject']}\n"
                f"Body: {ticket['body']}"
            )
        else:
            observation = f"Unknown action '{action}'. Available: read_ticket, set_field, submit"

        if self.current_step >= self.task_config["max_steps"] and not done:
            done = True
            observation += " [Max steps reached]"

        self.done = done
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": {
                "step": self.current_step,
                "submission": self.submission,
                "episode_reward": self.episode_reward,
            },
        }


_env = TicketTriageEnvironment()


def get_env() -> TicketTriageEnvironment:
    return _env
