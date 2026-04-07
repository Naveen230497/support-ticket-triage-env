# Support Ticket Triage Environment

> **Meta x PyTorch x Scaler OpenEnv Hackathon — Round 1 Submission**  
> Author: Guthikonda Naveen | guthikondanaveen9@gmail.com

## Overview

A real-world [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent acts as a **customer support triage specialist**. The agent reads incoming support tickets and fixes triage issues — wrong category, missing priority, unassigned team, unresolved duplicates, missing SLA times, and required escalations — simulating the actual workflow in enterprise SaaS support operations.

This environment is directly relevant to any team running a customer support desk, and fills a genuine gap in the OpenEnv ecosystem (no support triage environment exists).

## Environment Description

| Property | Value |
|----------|-------|
| Framework | OpenEnv (openenv-core) |
| Language | Python 3.11 |
| Server | FastAPI + Uvicorn |
| Port | 7860 |
| Tasks | 3 (easy, medium, hard) |
| Score Range | 0.0 – 1.0 per task |
| Reward Type | Partial (non-sparse, step-level) |

## Tasks

### Task 1 — Easy: Login Failure Triage
- **Ticket:** User cannot log in, gets "Invalid credentials" error
- **Issues:** `category` and `priority` are missing (2 issues)
- **Agent must:** Set category=`account_access`, priority=`high`
- **Max steps:** 8
- **Grader:** 0.5 per correct field (max 1.0)

### Task 2 — Medium: Billing Dispute Resolution
- **Ticket:** Customer charged twice for subscription, needs refund
- **Issues:** 5 fields missing — category, priority, team, tag, resolution time
- **Agent must:** Set category=`billing`, priority=`high`, team=`billing_team`, tag=`refund`, resolution_time<=8h
- **Max steps:** 20
- **Grader:** 0.20 per correct field (max 1.0)

### Task 3 — Hard: Enterprise Checkout Crash Escalation
- **Ticket:** Enterprise checkout crashes for 500 users — **WRONG** pre-filled values
- **Issues:** 6 issues — wrong category (`product_feedback`→`technical`), wrong priority (`low`→`critical`), no team, unmerged duplicate, no escalation, no SLA time
- **Agent must:** Correct all 6 issues including `merge_duplicate` and `escalate`
- **Max steps:** 30
- **Grader:** 1/6 per resolved issue (max 1.0)

## Action Space

| Field | Type | Values |
|-------|------|--------|
| `action_type` | string | `set_category`, `set_priority`, `assign_team`, `add_tag`, `set_resolution_time`, `merge_duplicate`, `escalate`, `mark_resolved` |
| `value` | string (optional) | The value to apply (category name, priority level, team name, tag, hours) |
| `confidence` | float | 0.0 – 1.0 |

**Valid values:**
- Categories: `billing`, `technical`, `account_access`, `product_feedback`, `shipping`
- Priorities: `low`, `medium`, `high`, `critical`
- Teams: `billing_team`, `tech_support`, `account_team`, `product_team`, `logistics`

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | string | Ticket ID |
| `title` | string | Ticket title |
| `description` | string | Full ticket description |
| `current_category` | string | Current category (may be wrong or empty) |
| `current_priority` | string | Current priority (may be wrong or empty) |
| `assigned_team` | string | Assigned team (may be empty) |
| `tags` | array | Current tags |
| `resolution_time_hours` | float | Planned resolution time |
| `issues_remaining` | int | Number of open issues |
| `feedback` | string | Human-readable result of last action |
| `reward` | float | Reward for last step |
| `done` | bool | Whether episode is complete |
| `duplicate_ticket` | object | Duplicate ticket info (hard task only) |

## Reward Function

| Action | Reward |
|--------|--------|
| Correct required fix | +0.30 |
| Merge duplicate correctly | +0.30 |
| Escalate correctly | +0.30 |
| Valid action (no matching fix) | +0.05 |
| Resolution time exceeds SLA | -0.10 |
| Mark resolved with issues remaining | -0.10 |
| Invalid value submitted | -0.05 |
| All issues resolved (bonus) | +0.20 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns 200 if service is up |
| `/reset` | POST | Start a new episode. Body: `{"task_id": "task_easy"}` |
| `/step` | POST | Take an action. Body: action object |
| `/state` | GET | Get current episode state |
| `/tasks` | GET | List all tasks and action schema |
| `/grader` | POST | Grade current episode. Body: `{"task_id": "..."}` |
| `/baseline` | POST | Run full baseline agent on all 3 tasks |

## Quick Start

### Option 1: Run locally with Python

```bash
# 1. Clone the repo
git clone https://github.com/Naveen230497/support-ticket-triage-env.git
cd support-ticket-triage-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 4. Run the baseline
python baseline.py
```

### Option 2: Run with Docker

```bash
# Build
docker build -t support-ticket-triage-env .

# Run
docker run -p 7860:7860 support-ticket-triage-env

# Test health
curl http://localhost:7860/health
```

### Test the environment manually

```bash
# Reset with easy task
curl -X POST http://localhost:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_easy"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"action_type": "set_category", "value": "account_access"}'

# Grade the episode
curl -X POST http://localhost:7860/grader \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "task_easy"}'
```

## Baseline Scores

The rule-based baseline agent achieves:

| Task | Score |
|------|-------|
| task_easy | 1.0 |
| task_medium | 1.0 |
| task_hard | 1.0 |
| **Average** | **1.0** |

Run `python baseline.py` to reproduce.

## Project Structure

```
support-ticket-triage-env/
├── openenv.yaml        # OpenEnv spec (metadata, tasks, endpoints, action/obs schema)
├── Dockerfile          # Container definition (python:3.11-slim-bookworm, port 7860)
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Build system + openenv-core entry points
├── inference.py        # LLM agent inference script (OpenAI client, structured stdout)
├── baseline.py         # Reproducible deterministic baseline script
├── models.py           # TicketAction, TicketObservation, TicketState dataclasses
└── server/
    ├── __init__.py     # Package init
    ├── app.py          # FastAPI server (all 7 endpoints + rate limiter)
    ├── environment.py  # Core SupportTicketEnvironment class
    ├── tasks.py        # 3 real-world task definitions
    └── graders.py      # Deterministic partial-credit graders
```

## Why This Environment Stands Out

1. **Real-world domain** — Support ticket triage is a core operational task at every SaaS company
2. **Meaningful difficulty progression** — Easy (2 blank fields) → Medium (5 blank fields) → Hard (wrong pre-filled + duplicate + escalation)
3. **Non-sparse rewards** — Every step provides signal (+0.30 for correct fix, -0.10 for penalty)
4. **Hard task genuinely challenges models** — Agent must detect and **correct** wrong existing values, not just fill blanks
5. **Novel domain** — No support triage env exists in OpenEnv ecosystem

## License

MIT — open for evaluation by the hackathon judges.

## Submission

- **GitHub**: https://github.com/Naveen230497/support-ticket-triage-env
- **HF Space**: https://huggingface.co/spaces/Naveen230497/support-ticket-triage-env
