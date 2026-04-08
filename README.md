---
title: Universe
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🏦 Banking Customer Service - OpenEnv RL Environment

**Meta × PyTorch × Scaler OpenEnv Hackathon 2026 Submission**

## 🌟 Project Overview
This is a high-fidelity Reinforcement Learning (RL) environment where an LLM agent acts as a Banking Customer Service Representative. The agent resolves complex, multi-step customer queries by calling specific banking tools, graded on tool correctness and resolution quality.

### Key Features:
- **9 Real-World Tasks:** Ranging from simple balance checks to hard, multi-intent fraud investigations.
- **Tool-Augmented AI:** Uses banking APIs like `check_balance`, `block_card`, and `raise_dispute`.
- **Sophisticated Reward Signal:** A weighted composite reward function (Weights: 0.40 Tools, 0.35 Resolution, 0.10 Safety, 0.15 Quality).
- **OpenEnv Compliant:** Fully supports `step()`, `reset()`, and `state()` endpoints via FastAPI.

---

## 🛠 Action & Observation Spaces

### Action Space
- **`tool_call`**: Executing banking functions with specific parameters.
- **`final_response`**: Delivering the resolution to the customer in professional natural language.

### Observation Space
The agent receives structured JSON containing customer profiles, task metadata, and full tool execution history to maintain context.

---

## 📈 Benchmark Results (Baseline)
Tested with **Qwen/Qwen2.5-72B-Instruct**, achieving a high success rate:

| Task ID | Difficulty | Description | Reward | Status |
| :--- | :--- | :--- | :--- | :--- |
| **T001** | Easy | Balance Enquiry | **0.75** | ✅ Pass |
| **T005** | Medium | Transaction Dispute | **1.00** | ✅ Pass |
| **T007** | Hard | Multi-Intent (Block+Loan+Balance) | **0.82** | ✅ Pass |
| **T008** | Hard | Fraud Investigation | **1.00** | ✅ Pass |

**Average Success Score: 0.85+**

---

## 🚀 Deployment & Usage
- **Platform:** Hugging Face Spaces (Docker)
- **Framework:** FastAPI, Uvicorn, OpenAI SDK
- **Port:** 7860

### How to test:
1. Access the API documentation at the `/docs` endpoint.
2. Use the `inference.py` script to run the baseline evaluation.

---

## 🛡 Security & Ethics
Developed by a **Cyber Security student**, this environment emphasizes:
- **Mock Data Anonymization:** No real user data is used.
- **Strict Validation:** Prevents hallucinated tool calls through deterministic environment checks.

---

## 📄 License
This project is licensed under the MIT License.