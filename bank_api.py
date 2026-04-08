"""
bank_api.py
Simulated banking backend — in-memory customer data + 5 tool functions.
"""

from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────
# In-memory customer database
# ─────────────────────────────────────────────

CUSTOMERS = {
    "C001": {
        "name": "Arjun Mehta",
        "email": "arjun.mehta@email.com",
        "accounts": {
            "ACC001": {"type": "savings",  "balance": 45230.75, "currency": "INR"},
            "ACC002": {"type": "current",  "balance": 12800.00, "currency": "INR"},
        },
        "cards": {
            "CARD001": {"type": "debit",  "last4": "4521", "status": "active",  "limit": None},
            "CARD002": {"type": "credit", "last4": "8832", "status": "active",  "limit": 100000},
        },
        "loans": {
            "LOAN001": {"type": "home_loan",     "amount": 2500000, "outstanding": 1820000,
                        "emi": 22500, "next_due": "2026-04-15", "status": "active"},
            "LOAN002": {"type": "personal_loan", "amount": 200000,  "outstanding": 0,
                        "emi": 0,     "next_due": None,          "status": "closed"},
        },
    },
    "C002": {
        "name": "Priya Sharma",
        "email": "priya.sharma@email.com",
        "accounts": {
            "ACC003": {"type": "savings", "balance": 8750.50, "currency": "INR"},
        },
        "cards": {
            "CARD003": {"type": "debit",  "last4": "9910", "status": "active",  "limit": None},
            "CARD004": {"type": "credit", "last4": "2231", "status": "active",  "limit": 50000},
        },
        "loans": {},
    },
    "C003": {
        "name": "Rahul Verma",
        "email": "rahul.verma@email.com",
        "accounts": {
            "ACC004": {"type": "savings", "balance": 3200.00, "currency": "INR"},
            "ACC005": {"type": "current", "balance": 67500.00, "currency": "INR"},
        },
        "cards": {
            "CARD005": {"type": "debit",  "last4": "1122", "status": "active",  "limit": None},
        },
        "loans": {
            "LOAN003": {"type": "car_loan", "amount": 800000, "outstanding": 420000,
                        "emi": 15000, "next_due": "2026-04-20", "status": "active"},
        },
    },
}

# Seeded random transactions
_CATEGORIES = ["grocery","fuel","restaurant","online_shopping","emi","utility","atm_withdrawal","transfer"]
_MERCHANTS  = ["BigBasket","HP Petrol","Swiggy","Amazon","HDFC EMI","BESCOM","ATM","NEFT Transfer"]

def _generate_transactions(account_id: str, n: int = 10, seed: int = 42):
    rng = random.Random(seed + hash(account_id) % 1000)
    txns = []
    for i in range(n):
        days_ago = rng.randint(0, 30)
        amount   = round(rng.uniform(100, 8000), 2)
        cat_idx  = rng.randint(0, len(_CATEGORIES) - 1)
        txns.append({
            "txn_id":       f"TXN{account_id[-3:]}{i:04d}",
            "date":         (datetime(2026, 4, 8) - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "amount":       amount,
            "type":         rng.choice(["debit", "credit"]),
            "merchant":     _MERCHANTS[cat_idx],
            "category":     _CATEGORIES[cat_idx],
        })
    return sorted(txns, key=lambda x: x["date"], reverse=True)


# ─────────────────────────────────────────────
# Tool functions
# ─────────────────────────────────────────────

def check_balance(customer_id: str, account_id: str = None) -> dict:
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found"}
    accounts = customer["accounts"]
    if account_id:
        acct = accounts.get(account_id)
        if not acct:
            return {"error": f"Account {account_id} not found"}
        return {"customer": customer["name"], "account_id": account_id, **acct}
    return {
        "customer": customer["name"],
        "accounts": [{"account_id": aid, **info} for aid, info in accounts.items()]
    }


def get_transactions(customer_id: str, account_id: str, limit: int = 5) -> dict:
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found"}
    if account_id not in customer["accounts"]:
        return {"error": f"Account {account_id} not found"}
    txns = _generate_transactions(account_id)[:limit]
    return {"customer": customer["name"], "account_id": account_id, "transactions": txns}


def block_card(customer_id: str, card_id: str, reason: str = "customer_request") -> dict:
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found"}
    card = customer["cards"].get(card_id)
    if not card:
        return {"error": f"Card {card_id} not found"}
    if card["status"] == "blocked":
        return {"message": f"Card ending {card['last4']} is already blocked.",
                "card_id": card_id, "status": "blocked"}
    card["status"] = "blocked"
    return {
        "success": True,
        "message": f"Card ending {card['last4']} has been successfully blocked.",
        "card_id": card_id, "reason": reason,
        "timestamp": "2026-04-08 10:00:00",
    }


def check_loan_status(customer_id: str, loan_id: str = None) -> dict:
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found"}
    loans = customer["loans"]
    if not loans:
        return {"message": "No loans found for this customer."}
    if loan_id:
        loan = loans.get(loan_id)
        if not loan:
            return {"error": f"Loan {loan_id} not found"}
        return {"customer": customer["name"], "loan_id": loan_id, **loan}
    return {"customer": customer["name"],
            "loans": [{"loan_id": lid, **info} for lid, info in loans.items()]}


def raise_dispute(customer_id: str, txn_id: str, reason: str) -> dict:
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found"}
    ticket_id = f"DIS{abs(hash(txn_id + customer_id)) % 900000 + 100000}"
    return {
        "success": True,
        "ticket_id": ticket_id,
        "txn_id": txn_id,
        "status": "under_review",
        "message": f"Dispute raised. Ticket: {ticket_id}. Resolution within 7 working days.",
        "reason": reason,
        "timestamp": "2026-04-08 10:00:00",
    }


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

AVAILABLE_TOOLS = {
    "check_balance":     check_balance,
    "get_transactions":  get_transactions,
    "block_card":        block_card,
    "check_loan_status": check_loan_status,
    "raise_dispute":     raise_dispute,
}