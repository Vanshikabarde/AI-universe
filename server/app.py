import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import RedirectResponse

# Import paths updated for sub-folder structure
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import BankingCSEnv
from models import BankingAction, BankingObservation, EnvState, StepResult

app = FastAPI(
    title="Banking Customer Service — OpenEnv",
    description="RL environment for banking customer service agents.",
    version="1.0.0",
)

# Global env instance
env = BankingCSEnv(difficulty="all", seed=42)

@app.get("/")
def root():
    return {
        "status": "Running",
        "project": "Banking Customer Service OpenEnv",
        "organization": "SIRT Bhopal",
        "tasks_available": ["T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009"],
        "docs": "/docs"
    }

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "all"
    task_id:    Optional[str] = None
    seed:       Optional[int] = None

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> dict:
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
