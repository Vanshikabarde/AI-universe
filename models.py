"""
models.py
Typed Pydantic models for the Banking Customer Service OpenEnv environment.
Full OpenEnv spec: Observation, Action, Reward
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────

class ToolParam(BaseModel):
    name: str
    type: str
    required: bool
    description: str


class ToolDefinition(BaseModel):
    name: str
    description: str
    params: List[ToolParam]


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class CustomerInfo(BaseModel):
    id: str
    name: str
    email: str


class ToolResult(BaseModel):
    tool: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    success: bool


class BankingObservation(BaseModel):
    """Returned by reset() and step()."""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    customer: CustomerInfo
    query: str
    tools_available: List[ToolDefinition]
    instructions: str
    history: List[ToolResult] = Field(default_factory=list)
    step_count: int = 0
    done: bool = False
    feedback: Optional[str] = None          # populated after final_response
    warning: Optional[str] = None           # e.g. max tool calls reached
    error: Optional[str] = None             # bad action type


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class BankingAction(BaseModel):
    """
    Two valid action types:

    1. tool_call  — call a banking tool
       action_type = "tool_call"
       tool        = "check_balance" | "get_transactions" | "block_card" |
                     "check_loan_status" | "raise_dispute"
       params      = dict of tool arguments

    2. final_response — send the final reply to the customer
       action_type = "final_response"
       response    = the text reply
    """
    action_type: Literal["tool_call", "final_response"]
    tool: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    response: Optional[str] = None


# ─────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    required_tools:     float = 0.0   # 0.40 weight
    task_resolved:      float = 0.0   # 0.35 weight
    no_forbidden_tools: float = 0.0   # 0.10 weight
    response_quality:   float = 0.0   # 0.15 weight
    total:              float = 0.0


class BankingReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward in [0, 1]")
    breakdown: RewardBreakdown
    feedback: str


# ─────────────────────────────────────────────
# Step result
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    observation: BankingObservation
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# State (for GET /state)
# ─────────────────────────────────────────────

class EnvState(BaseModel):
    status: Literal["not_started", "in_progress", "done"]
    task_id: Optional[str] = None
    difficulty: Optional[str] = None
    description: Optional[str] = None
    step_count: int = 0
    tools_called: List[str] = Field(default_factory=list)
    max_tool_calls: int = 5