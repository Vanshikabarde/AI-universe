# Banking Customer Service — OpenEnv Environment

> **Meta × PyTorch × Scaler School of Technology — OpenEnv AI Hackathon 2026**
> Theme: Customer Service Agents

---

## Environment description

A real-world RL environment where an LLM agent acts as a **banking customer service representative**.

The agent receives a customer's natural language query and must:
1. Identify which banking tools to call (balance check, card block, dispute, etc.)
2. Call tools across multiple turns to gather information
3. Produce a final response that fully resolves the customer's problem

This simulates a task humans actually perform — not a game or toy. Every scenario is a realistic banking support interaction covering the most common customer pain points: balance enquiries, fraud response, card emergencies, and loan management.

---

## Project structure

```
banking_cs_env/
├── models.py        # Typed Pydantic models — Observation, Action, Reward
├── bank_api.py      # Mock banking backend — 5 deterministic tool functions
├── tasks.py         # 9 tasks with graders (easy / medium / hard)
├── environment.py   # Core RL environment — step() / reset() / state()
├── server.py        # FastAPI HTTP server
├── inference.py     # Baseline inference script (OpenAI client)
├── openenv.yaml     # OpenEnv spec metadata
├── Dockerfile       # Container for HF Spaces deployment
├── requirements.txt
└── README.md
```

---

## Action space

Two action types:

**1. Tool call**
```json
{
  "action_type": "tool_call",
  "tool": "check_balance",
  "params": {"customer_id": "C001", "account_id": "ACC001"}
}
```

**2. Final response**
```json
{
  "action_type": "final_response",
  "response": "Your savings account balance is ₹45,230.75."
}
```

---

## Observation space

```json
{
  "task_id":         "T005",
  "difficulty":      "medium",
  "customer":        {"id": "C001", "name": "Arjun Mehta", "email": "..."},
  "query":           "I see a suspicious transaction on account ACC001...",
  "tools_available": [...],
  "instructions":    "You are a banking customer service agent...",
  "history":         [{"tool": "get_transactions", "result": {...}}],
  "step_count":      2,
  "done":            false,
  "feedback":        null
}
```

---

## Available tools

| Tool | Description |
|------|-------------|
| `check_balance` | Get account balance — one or all accounts |
| `get_transactions` | Get recent transaction history |
| `block_card` | Block a debit or credit card immediately |
| `check_loan_status` | Get outstanding amount, EMI, next due date |
| `raise_dispute` | Raise a dispute for a suspicious transaction |

---

## Task descriptions

| ID | Difficulty | Scenario | Required Tools |
|----|------------|----------|----------------|
| T001 | Easy | Savings account balance enquiry | `check_balance` |
| T002 | Easy | Recent transaction history | `get_transactions` |
| T003 | Easy | Home loan status + next EMI | `check_loan_status` |
| T004 | Medium | Emergency card block | `block_card` |
| T005 | Medium | Suspicious transaction → dispute | `get_transactions`, `raise_dispute` |
| T006 | Medium | Balance across accounts + loan EMI | `check_balance`, `check_loan_status` |
| T007 | Hard | Lost wallet — card block + balance + loan | `block_card`, `check_balance`, `check_loan_status` |
| T008 | Hard | Full fraud investigation | `get_transactions`, `raise_dispute`, `check_balance` |
| T009 | Hard | Card fraud + account check + loan | `block_card`, `get_transactions`, `check_loan_status` |

---

## Reward function

| Component | Weight | Criteria |
|-----------|--------|----------|
| Required tools used | 40% | Agent called all tools needed for the task |
| Task resolved | 35% | Final response contains correct key information |
| No forbidden tools | 10% | No irrelevant tools called |
| Response quality | 15% | Response ≥60 chars, uses helpful/polite language |

**Partial credit:** intermediate tool calls give a small +0.04 reward each when they match required tools, so reward flows across the full trajectory — not just at episode end.

---

## Setup and usage

### Local (recommended to test first)

```bash
# Install
pip install -r requirements.txt

# Terminal 1 — start server
python server.py
# → http://localhost:8000/docs

# Terminal 2 — run baseline
python inference.py
```

### Docker

```bash
docker build -t banking-cs-env .
docker run -p 8000:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  banking-cs-env
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |
| `PORT` | Server port | `7860` |

---

## API reference

### `POST /reset`
```json
{"difficulty": "easy", "task_id": "T001", "seed": 42}
```

### `POST /step`
```json
{"action_type": "tool_call", "tool": "check_balance", "params": {"customer_id": "C001"}}
```

### `GET /state`
Returns current episode state.

### `GET /health`
Returns `{"status": "ok"}`.

---

## Baseline scores

Run with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Expected Score |
|------|------------|----------------|
| T001 | Easy | ~0.85 |
| T005 | Medium | ~0.70 |
| T007 | Hard | ~0.55 |
| **Average** | | **~0.70** |

---

## Deploy to HuggingFace Spaces

1. Create a new Space (Docker SDK) at huggingface.co/spaces
2. Add secrets: `HF_TOKEN`, `MODEL_NAME`, `API_BASE_URL`
3. Push this repo — HF builds and deploys automatically

```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/banking-cs-openenv
git push origin main
```

---

## Submission — Meta × PyTorch × Scaler OpenEnv Hackathon 2026