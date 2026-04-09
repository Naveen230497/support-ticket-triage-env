from server.tasks import get_task_config
from server.graders import grade, _clamp

# Score that represents "no progress" - strictly between 0 and 1
_NO_PROGRESS_SCORE = 0.001


class TicketTriageEnvironment:
    def __init__(self):
        self.task_config = None
        self.current_step = 0
        self.done = False
        self.submission = {}
        self.episode_reward = _NO_PROGRESS_SCORE

    def reset(self, task_id: str, seed: int = 42) -> dict:
        self.task_config = get_task_config(task_id, seed)
        self.current_step = 0
        self.done = False
        self.submission = {}
        self.episode_reward = _NO_PROGRESS_SCORE

        ticket = self.task_config["ticket"]
        observation = {
            "ticket_id": ticket["id"],
            "subject": ticket["subject"],
            "body": ticket["body"],
            "task": self.task_config["description"],
            "required_fields": self.task_config["required_fields"],
            "max_steps": self.task_config["max_steps"],
        }
        return {
            "observation": observation,
            "info": {
                "task_id": task_id,
                "ticket_id": ticket["id"],
                "required_fields": self.task_config["required_fields"],
                "max_steps": self.task_config["max_steps"],
                "ground_truth": ticket,
            },
        }

    def step(self, action: str, parameters: dict) -> dict:
        if self.done:
            return {
                "observation": {"status": "Episode is already done. Call /reset to start a new episode."},
                "reward": _NO_PROGRESS_SCORE,
                "done": True,
                "info": {"error": "episode_done"},
            }

        self.current_step += 1
        reward = _NO_PROGRESS_SCORE
        done = False
        observation = {}

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
            observation = {"status": "submitted", "score": round(reward, 4)}

        elif action == "set_field":
            field = parameters.get("field", "")
            value = parameters.get("value", "")
            self.submission[field] = value
            reward = grade(
                self.task_config["task_id"],
                self.submission,
                self.task_config["ticket"],
            )
            observation = {"status": "field_set", "field": field, "value": value, "partial_score": round(reward, 4)}

        elif action == "read_ticket":
            ticket = self.task_config["ticket"]
            observation = {
                "ticket_id": ticket["id"],
                "subject": ticket["subject"],
                "body": ticket["body"],
            }

        else:
            observation = {"error": f"Unknown action '{action}'. Available: read_ticket, set_field, submit"}

        if self.current_step >= self.task_config["max_steps"] and not done:
            done = True
            observation["max_steps_reached"] = True
            reward = self.episode_reward if self.episode_reward > _NO_PROGRESS_SCORE else _NO_PROGRESS_SCORE

        # Ensure reward is ALWAYS strictly between 0 and 1
        reward = _clamp(reward)
        self.done = done

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": {
                "step": self.current_step,
                "submission": self.submission,
                "ground_truth": self.task_config["ticket"],
                "episode_reward": self.episode_reward,
            },
        }


_env = TicketTriageEnvironment()


def get_env() -> TicketTriageEnvironment:
    return _env
