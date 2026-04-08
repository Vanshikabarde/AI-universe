import json
import os
import sys
import textwrap
from typing import List, Optional

# 1. Sabse pehle dotenv load karo
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[DEBUG] python-dotenv not installed. Run: pip install python-dotenv")

from openai import OpenAI

# ── Import env directly ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from environment import BankingCSEnv
from models import BankingAction

# ── Config (Ab ye automatic .env se uthayega) ─────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")

BENCHMARK         = "banking-cs"
MAX_STEPS         = 8
TEMPERATURE       = 0.2
MAX_TOKENS        = 512
SUCCESS_THRESHOLD = 0.5
EVAL_TASKS = ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009"]

# ── Mandatory log helpers ─────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── System prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are a professional Banking Customer Service Representative.
Your goal is to solve customer queries accurately using the tools provided.

AVAILABLE TOOLS (Use EXACT names):
1. check_balance(customer_id, account_id=None)
2. get_transactions(customer_id, account_id, limit=5)
3. block_card(customer_id, card_id, reason)
4. check_loan_status(customer_id, loan_id=None)
5. raise_dispute(customer_id, txn_id, reason)

STRICT RULES:
- ALWAYS respond in valid JSON format.
- If a customer mentions a card ending in digits (e.g., '4521'), do NOT use those digits as the 'card_id'. Instead, first call 'get_transactions' or check available data to find the actual Card ID (e.g., 'CARD001').
- Use 'check_balance' for balance inquiries. (NEVER use 'get_account_balance').
- Use 'get_transactions' for history/statement requests. (NEVER use 'get_recent_transactions').
- For the 'final_response', be professional and include specific numbers/details fetched from tools.

JSON FORMATS:
- Tool Call: {"action_type": "tool_call", "tool": "tool_name", "params": {"key": "value"}}
- Final Response: {"action_type": "final_response", "response": "Your message to the customer."}
""").strip()

def build_prompt(obs_dict: dict, history: List[str]) -> str:
    customer = obs_dict["customer"]
    query    = obs_dict["query"]
    tool_results = ""
    for tr in obs_dict.get("history", []):
        tool_results += f"\nTool: {tr['tool']} → {json.dumps(tr['result'])}"

    return f"Customer: {customer['name']} (ID: {customer['id']})\nQuery: {query}\nHistory:{tool_results}\nNext Action (JSON):"

def call_llm(client: OpenAI, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return ""

def parse_action(raw: str) -> Optional[BankingAction]:
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1: return None
        data = json.loads(raw[start:end])
        return BankingAction(**data)
    except:
        return None

def run_episode(client: OpenAI, env: BankingCSEnv, task_id: str) -> float:
    obs = env.reset_task(task_id)
    obs_dict = obs.model_dump()
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards, history = [], []
    score, success = 0.0, False

    try:
        for step_n in range(1, MAX_STEPS + 1):
            if obs_dict.get("done"): break
            prompt = build_prompt(obs_dict, history)
            raw = call_llm(client, prompt)
            action = parse_action(raw)
            
            # Fallback simple logic if parsing fails
            if not action:
                action = BankingAction(action_type="final_response", response="I am processing your request.")

            action_str = json.dumps(action.model_dump(exclude_none=True)).replace(" ", "")
            result = env.step(action)
            
            log_step(step=step_n, action=action_str, reward=result.reward, done=result.done, error=result.observation.error)
            
            rewards.append(result.reward)
            obs_dict = result.observation.model_dump()
            if result.done:
                score = result.reward
                success = score >= SUCCESS_THRESHOLD
                break
    finally:
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
    return score

def main():
    if not API_KEY:
        print("ERROR: HF_TOKEN not found in .env file!")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = BankingCSEnv(seed=42)

    for task_id in EVAL_TASKS:
        run_episode(client, env, task_id)

if __name__ == "__main__":
    main()