"""
tasks.py
Task bank — 9 scenarios (easy / medium / hard) each with a
deterministic grader that scores agent performance 0.0–1.0.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ─────────────────────────────────────────────
# Task schema
# ─────────────────────────────────────────────

class Task:
    def __init__(
        self,
        id: str,
        difficulty: str,
        customer_id: str,
        query: str,
        required_tools: List[str],
        forbidden_tools: List[str],
        response_keywords: List[str],
        description: str,
        resolution: str,
    ):
        self.id                = id
        self.difficulty        = difficulty
        self.customer_id       = customer_id
        self.query             = query
        self.required_tools    = required_tools
        self.forbidden_tools   = forbidden_tools
        self.response_keywords = response_keywords   # must appear in final response
        self.description       = description
        self.resolution        = resolution

    def grade(
        self,
        tools_called: List[str],
        final_response: str,
    ) -> Dict[str, Any]:
        """
        Deterministic grader. Returns score 0.0–1.0 with breakdown.
        Weights: required_tools=0.40, task_resolved=0.35,
                 no_forbidden_tools=0.10, response_quality=0.15
        """
        resp_lower = final_response.lower()

        # 1. Required tools (0.40)
        required  = set(self.required_tools)
        used      = set(tools_called)
        hit_ratio = len(required & used) / len(required) if required else 1.0
        tools_score = round(hit_ratio * 0.40, 4)

        # 2. Task resolved — keywords in response (0.35)
        kws = self.response_keywords
        kw_hits = sum(1 for k in kws if k.lower() in resp_lower)
        kw_ratio = kw_hits / len(kws) if kws else 1.0
        resolved_score = round(kw_ratio * 0.35, 4)

        # 3. No forbidden tools (0.10)
        bad = set(self.forbidden_tools) & used
        no_bad_score = 0.0 if bad else 0.10

        # 4. Response quality (0.15)
        quality = 0.0
        if len(final_response) >= 60:
            quality += 0.075
        polite = ["please", "sorry", "thank", "help", "resolve",
                  "assist", "apolog", "understand"]
        if any(w in resp_lower for w in polite):
            quality += 0.075
        quality_score = round(quality, 4)

        total = round(tools_score + resolved_score + no_bad_score + quality_score, 4)

        missing_tools = list(required - used)
        missing_kws   = [k for k in kws if k.lower() not in resp_lower]

        total = round(tools_score + resolved_score + no_bad_score + quality_score, 4)

        if total >= 1.0:
            total = 0.98
        elif total <= 0.0:
            total = 0.02

        missing_tools = list(required - used)
        missing_kws   = [k for k in kws if k.lower() not in resp_lower]

        return {
            "score": total,
            "breakdown": {
                "required_tools":      tools_score,
                "task_resolved":       resolved_score,
                "no_forbidden_tools":  no_bad_score,
                "response_quality":    quality_score,
            },
            "missing_tools":    missing_tools,
            "missing_keywords": missing_kws,
            "forbidden_used":   list(bad),
        }


# ─────────────────────────────────────────────
# Task bank
# ─────────────────────────────────────────────

TASKS: List[Task] = [

    # ── EASY ──────────────────────────────────────────────────
    Task(
        id="T001", difficulty="easy",
        customer_id="C001",
        query="Hi, I want to check my savings account balance. My account is ACC001.",
        required_tools=["check_balance"],
        forbidden_tools=[],
        response_keywords=["45230", "savings"],
        description="Simple savings account balance enquiry",
        resolution="balance_check",
    ),
    Task(
        id="T002", difficulty="easy",
        customer_id="C002",
        query="Can you show me my last 5 transactions on account ACC003?",
        required_tools=["get_transactions"],
        forbidden_tools=[],
        response_keywords=["transaction", "ACC003"],
        description="Recent transaction history lookup",
        resolution="transaction_history",
    ),
    Task(
        id="T003", difficulty="easy",
        customer_id="C001",
        query="What is the outstanding amount and next due date for my home loan LOAN001?",
        required_tools=["check_loan_status"],
        forbidden_tools=[],
        response_keywords=["1820000", "2026-04-15"],
        description="Loan status and next EMI enquiry",
        resolution="loan_status_check",
    ),

    # ── MEDIUM ────────────────────────────────────────────────
    Task(
        id="T004", difficulty="medium",
        customer_id="C002",
        query="My debit card ending 9910 was stolen! Please block it immediately.",
        required_tools=["block_card"],
        forbidden_tools=[],
        response_keywords=["blocked", "9910"],
        description="Emergency card block",
        resolution="card_blocked",
    ),
    Task(
        id="T005", difficulty="medium",
        customer_id="C001",
        query="I see a suspicious transaction on account ACC001. Show me recent transactions and raise a dispute.",
        required_tools=["get_transactions", "raise_dispute"],
        forbidden_tools=[],
        response_keywords=["dispute", "ticket"],
        description="Suspicious transaction — look up then dispute",
        resolution="dispute_raised",
    ),
    Task(
        id="T006", difficulty="medium",
        customer_id="C003",
        query="What is my total balance across all accounts, and when is my next car loan EMI due?",
        required_tools=["check_balance", "check_loan_status"],
        forbidden_tools=[],
        response_keywords=["3200", "67500", "2026-04-20"],
        description="Multi-account balance + loan EMI date",
        resolution="balance_and_loan",
    ),

    # ── HARD ──────────────────────────────────────────────────
    Task(
        id="T007", difficulty="hard",
        customer_id="C001",
        query=(
            "I lost my wallet. My debit card ending 4521 might be compromised. "
            "Please block it, also tell me my savings balance, "
            "and when is my next home loan EMI due?"
        ),
        required_tools=["block_card", "check_balance", "check_loan_status"],
        forbidden_tools=[],
        response_keywords=["blocked", "4521", "45230", "2026-04-15"],
        description="Three tasks in one query — card block + balance + loan",
        resolution="multi_task_resolution",
    ),
    Task(
        id="T008", difficulty="hard",
        customer_id="C002",
        query=(
            "I noticed two suspicious debits on my savings account ACC003. "
            "Show me recent transactions, raise a dispute for any suspicious one, "
            "and also check my current balance."
        ),
        required_tools=["get_transactions", "raise_dispute", "check_balance"],
        forbidden_tools=[],
        response_keywords=["dispute", "ticket", "8750"],
        description="Full fraud investigation — transactions + dispute + balance",
        resolution="full_investigation",
    ),
    Task(
        id="T009", difficulty="hard",
        customer_id="C001",
        query=(
            "Someone made an unauthorised transaction using my credit card ending 8832. "
            "Block the card immediately, show me the recent transactions on ACC002, "
            "and tell me how much I still owe on my home loan."
        ),
        required_tools=["block_card", "get_transactions", "check_loan_status"],
        forbidden_tools=[],
        response_keywords=["blocked", "8832", "1820000"],
        description="Card fraud + account investigation + loan check",
        resolution="fraud_and_loan_check",
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
