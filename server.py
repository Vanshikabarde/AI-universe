import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.responses import RedirectResponse

# Inhi files se import ho raha hai
from environment import BankingCSEnv
from models import BankingAction, BankingObservation, EnvState, StepResult

app = FastAPI(
    title="Banking Customer Service — OpenEnv",
    description="RL environment for banking customer service agents. Meta × PyTorch Hackathon.",
    version="1.0.0",
)

env = BankingCSEnv(difficulty="all", seed=42)

# ── Root Route (Status Page) ──────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "Running",
        "project": "Banking Customer Service OpenEnv",
        "organization": "SIRT Bhopal",
        "tasks_available": ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009"],
        "docs": "/docs"
    }

# ── Request models ────────────────────────────────────────────────
class ResetRequest(BaseModel):
    difficulty: Optional[str] = "all"
    task_id:    Optional[str] = None
    seed:       Optional[int] = None

# ── Endpoints ─────────────────────────────────────────────────────
@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> dict:
    """Start a new episode."""
    global env
    seed = req.seed if req.seed is not None else 42
    env  = BankingCSEnv(difficulty=req.difficulty or "all", seed=seed)
    if req.task_id:
        obs = env.reset_task(req.task_id)
    else:
        obs = env.reset()
    return {"observation": obs.model_dump(), "done": False}

@app.post("/step")
def step(action: BankingAction) -> dict:
    """Take one step."""
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset first.")
    result: StepResult = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward":      result.reward,
        "done":        result.done,
        "truncated":   result.truncated,
        "info":        result.info,
    }

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "BankingCSEnv", "version": "1.0.0"}

# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hugging Face PORT environment variable use karta hai
    port = int(os.getenv("PORT", 7860)) 
    print(f"Banking CS OpenEnv server starting on port {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)