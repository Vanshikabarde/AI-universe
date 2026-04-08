import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys

# Path fix for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import BankingCSEnv
from models import BankingAction, BankingObservation, EnvState, StepResult

app = FastAPI(title="Banking CS — OpenEnv")
env = BankingCSEnv(difficulty="all", seed=42)

@app.get("/")
def root():
    return {"status": "Running", "tasks": ["T001-T009"]}

# --- Endpoints (Reset/Step/Health) yahan purane wale hi rahenge ---
@app.post("/reset")
def reset(req: BaseModel = None):
    global env
    obs = env.reset()
    return {"observation": obs.model_dump(), "done": False}

@app.post("/step")
def step(action: BankingAction):
    result = env.step(action)
    return {"observation": result.observation.model_dump(), "reward": result.reward, "done": result.done}

@app.get("/health")
def health():
    return {"status": "ok"}

# --- YE VALA PART ZAROORI HAI ---
def main():
    """Main entry point for the validator"""
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
