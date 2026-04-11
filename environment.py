"""
environment.py
Banking Customer Service — full OpenEnv-compliant RL environment.

Implements:
  reset()  → BankingObservation
  step()   → StepResult (observation, reward, done, info)
  state()  → EnvState
"""

from __future__ import annotations
import random
from typing import List, Optional

from models import (
    BankingObservation, BankingAction, BankingReward,
    CustomerInfo, EnvState, RewardBreakdown,
    StepResult, ToolDefinition, ToolParam, ToolResult,
)
from bank_api import AVAILABLE_TOOLS, CUSTOMERS
from tasks import TASKS, TASKS_BY_ID, Task


# ─────────────────────────────────────────────
# Tool definitions (for observations)
# ─────────────────────────────────────────────

TOOL_DEFS: List[ToolDefinition] = [
    ToolDefinition(
        name="check_balance",
        description="Get account balance for a customer (one or all accounts).",
        params=[
            ToolParam(name="customer_id", type="str", required=True,  description="Customer ID e.g. C001"),
            ToolParam(name="account_id",  type="str", required=False, description="Account ID e.g. ACC001 (optional)"),
        ]
    ),
    ToolDefinition(
        name="get_transactions",
        description="Get recent transactions for a specific account.",
        params=[
            ToolParam(name="customer_id", type="str", required=True,  description="Customer ID"),
            ToolParam(name="account_id",  type="str", required=True,  description="Account ID"),
            ToolParam(name="limit",       type="int", required=False, description="Number of transactions (default 5)"),
        ]
    ),
    ToolDefinition(
        name="block_card",
        description="Block a debit or credit card immediately.",
        params=[
            ToolParam(name="customer_id", type="str", required=True,  description="Customer ID"),
            ToolParam(name="card_id",     type="str", required=True,  description="Card ID e.g. CARD001"),
            ToolParam(name="reason",      type="str", required=False, description="Reason for blocking"),
        ]
    ),
    ToolDefinition(
        name="check_loan_status",
        description="Get loan details: outstanding amount, EMI, next due date.",
        params=[
            ToolParam(name="customer_id", type="str", required=True,  description="Customer ID"),
            ToolParam(name="loan_id",     type="str", required=False, description="Loan ID e.g. LOAN001 (optional)"),
        ]
    ),
    ToolDefinition(
        name="raise_dispute",
        description="Raise a dispute for a suspicious or incorrect transaction.",
        params=[
            ToolParam(name="customer_id", type="str", required=True, description="Customer ID"),
            ToolParam(name="txn_id",      type="str", required=True, description="Transaction ID to dispute"),
            ToolParam(name="reason",      type="str", required=True, description="Reason for dispute"),
        ]
    ),
]


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class BankingCSEnv:
    """
    Banking Customer Service RL Environment.

    Episode flow:
      reset()              → initial BankingObservation
      step(tool_call)      → tool result, reward=0.0, done=False   (repeatable)
      step(final_response) → graded reward, done=True
    """

    MAX_TOOL_CALLS = 5
    INSTRUCTIONS = (
        "You are a banking customer service agent. "
        "Use the available tools to resolve the customer's query. "
        "You may call tools multiple times to gather information. "
        "Once you have everything needed, send action_type='final_response' "
        "with a clear, helpful reply to the customer."
    )

    def __init__(self, difficulty: str = "all", seed: Optional[int] = None):
        self._difficulty   = difficulty
        self._rng          = random.Random(seed)
        self._task: Optional[Task] = None
        self._tool_calls:  List[str]       = []
        self._tool_results: List[ToolResult] = []
        self._step_count   = 0
        self._done         = False

    # ── Public API ─────────────────────────────────────────────

    def reset(self) -> BankingObservation:
        pool = TASKS if self._difficulty == "all" else [
            t for t in TASKS if t.difficulty == self._difficulty
        ]
        self._task         = self._rng.choice(pool)
        self._tool_calls   = []
        self._tool_results = []
        self._step_count   = 0
        self._done         = False
        return self._make_obs()

    def reset_task(self, task_id: str) -> BankingObservation:
        """Reset to a specific task by ID (used by inference script)."""
        self._task         = TASKS_BY_ID[task_id]
        self._tool_calls   = []
        self._tool_results = []
        self._step_count   = 0
        self._done         = False
        return self._make_obs()

    def step(self, action: BankingAction) -> StepResult:
        assert not self._done, "Episode done — call reset() first."
        self._step_count += 1

        if action.action_type == "tool_call":
            return self._tool_step(action)
        elif action.action_type == "final_response":
            return self._response_step(action)
        else:
            obs = self._make_obs(error=f"Unknown action_type '{action.action_type}'")
            return StepResult(observation=obs, reward=0.0, done=False, truncated=False,
                              info={"error": "unknown_action_type"})

    def state(self) -> EnvState:
        if self._task is None:
            return EnvState(status="not_started")
        return EnvState(
            status="done" if self._done else "in_progress",
            task_id=self._task.id,
            difficulty=self._task.difficulty,
            description=self._task.description,
            step_count=self._step_count,
            tools_called=list(self._tool_calls),
            max_tool_calls=self.MAX_TOOL_CALLS,
        )

    # ── Internal ───────────────────────────────────────────────

    def _tool_step(self, action: BankingAction) -> StepResult:
        tool_name = action.tool or ""
        params    = action.params or {}

        if len(self._tool_calls) >= self.MAX_TOOL_CALLS:
            obs = self._make_obs(warning="Max tool calls reached. Send final_response now.")
            return StepResult(observation=obs, reward=0.0, done=False, truncated=False,
                              info={"warning": "max_tool_calls"})

        if tool_name not in AVAILABLE_TOOLS:
            obs = self._make_obs(error=f"Tool '{tool_name}' does not exist.")
            return StepResult(observation=obs, reward=-0.05, done=False, truncated=False,
                              info={"error": "invalid_tool"})

        try:
            result = AVAILABLE_TOOLS[tool_name](**params)
            success = "error" not in result
        except TypeError as e:
            result  = {"error": str(e)}
            success = False

        self._tool_calls.append(tool_name)
        tr = ToolResult(tool=tool_name, params=params, result=result, success=success)
        self._tool_results.append(tr)

        obs = self._make_obs()
        # Partial reward for using a required tool correctly
        partial = 0.04 if (tool_name in self._task.required_tools and success) else 0.0
        return StepResult(observation=obs, reward=partial, done=False, truncated=False,
                          info={"tool": tool_name, "result": result, "partial_reward": partial})

   def _response_step(self, action: BankingAction) -> StepResult:
        response = action.response or ""
        self._done = True

        grade   = self._task.grade(self._tool_calls, response)
        
        # --- PHASE 2 FIX: SQUEEZE SCORE BETWEEN 0.01 AND 0.99 ---
        raw_score = grade["score"]
        # Agar score 1.0 hai toh 0.98 kar do, agar 0.0 hai toh 0.02 kar do
        # Isse score hamesha (0, 1) ki range mein rahega
        safe_score = max(0.02, min(0.98, raw_score))
        # -------------------------------------------------------

        fb = self._build_feedback(grade)

        reward_obj = BankingReward(
            value=safe_score, # Updated to safe_score
            breakdown=RewardBreakdown(
                required_tools=grade["breakdown"]["required_tools"],
                task_resolved=grade["breakdown"]["task_resolved"],
                no_forbidden_tools=grade["breakdown"]["no_forbidden_tools"],
                response_quality=grade["breakdown"]["response_quality"],
                total=safe_score, # Updated to safe_score
            ),
            feedback=fb,
        )

        obs = self._make_obs(done=True, feedback=fb)
        return StepResult(
            observation=obs,
            reward=safe_score, # Updated to safe_score
            done=True,
            truncated=False,
            info={
                "reward_obj": reward_obj.model_dump(),
                "grade": grade,
                "tools_used": self._tool_calls,
                "steps_taken": self._step_count,
                "task_id": self._task.id,
            }
        )

        obs = self._make_obs(done=True, feedback=fb)
        return StepResult(
            observation=obs,
            reward=score,
            done=True,
            truncated=False,
            info={
                "reward_obj": reward_obj.model_dump(),
                "grade": grade,
                "tools_used": self._tool_calls,
                "steps_taken": self._step_count,
                "task_id": self._task.id,
            }
        )

    def _make_obs(
        self,
        done: bool = False,
        feedback: Optional[str] = None,
        warning: Optional[str] = None,
        error:   Optional[str] = None,
    ) -> BankingObservation:
        cust_data = CUSTOMERS[self._task.customer_id]
        return BankingObservation(
            task_id=self._task.id,
            difficulty=self._task.difficulty,
            customer=CustomerInfo(
                id=self._task.customer_id,
                name=cust_data["name"],
                email=cust_data["email"],
            ),
            query=self._task.query,
            tools_available=TOOL_DEFS,
            instructions=self.INSTRUCTIONS,
            history=list(self._tool_results),
            step_count=self._step_count,
            done=done,
            feedback=feedback,
            warning=warning,
            error=error,
        )

    @staticmethod
    def _build_feedback(grade: dict) -> str:
        parts = []
        if not grade["missing_tools"]:
            parts.append("✓ All required tools used")
        else:
            parts.append(f"✗ Missing tools: {grade['missing_tools']}")
        if not grade["missing_keywords"]:
            parts.append("✓ Response has all key information")
        else:
            parts.append(f"✗ Missing in response: {grade['missing_keywords']}")
        if not grade["forbidden_used"]:
            parts.append("✓ No irrelevant tools called")
        else:
            parts.append(f"✗ Irrelevant tools: {grade['forbidden_used']}")
        return " | ".join(parts)
