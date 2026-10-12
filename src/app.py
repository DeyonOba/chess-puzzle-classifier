"""FastAPI service for puzzle theme prediction.

    CPC_CKPT=logs/train/runs/NAME/checkpoints/best.ckpt uv run uvicorn src.app:app --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
         -d '{"fen": "...", "moves": ["e6e7", "b2b1"], "cp": 850}'

Env: CPC_CKPT (required), CPC_POSITION_CKPT (optional position-only model for
FEN-only requests), CPC_EVAL_RESULTS (optional eval_results.json for val-tuned
thresholds), CPC_DEVICE (auto|cpu|mps|cuda).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.inference import Predictor


class PredictRequest(BaseModel):
    fen: str = Field(..., description="position the solver sees (side to move = solver)")
    moves: list[str] | None = Field(None, description="solution line in UCI from `fen`")
    setup_move: str | None = Field(None, description="if `fen` is the Lichess CSV FEN, the first move of the Moves column")
    cp: int | None = Field(None, description="engine centipawn score after the solution, solver POV; sets the outcome tag by rule")
    decoding: str = Field("argmax_groups", pattern="^(argmax_groups|independent)$")
    apply_rules: bool = True

    @field_validator("fen")
    @classmethod
    def _valid_fen(cls, v: str) -> str:
        try:
            chess.Board(v)
        except ValueError as e:
            raise ValueError(f"invalid FEN: {e}")
        return v


class PredictResponse(BaseModel):
    tags: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    rule_tags: dict[str, str]
    decoding: str
    line_provided: bool
    positions: int
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("CPC_CKPT")
    if not ckpt:
        raise RuntimeError("set CPC_CKPT to a Lightning checkpoint (logs/train/runs/NAME/checkpoints/best.ckpt)")
    app.state.predictor = Predictor(ckpt, os.environ.get("CPC_POSITION_CKPT"), os.environ.get("CPC_DEVICE", "auto"),
                                    os.environ.get("CPC_EVAL_RESULTS"))
    yield


app = FastAPI(title="chess-puzzle-classifier", lifespan=lifespan)


@app.get("/health")
def health():
    p: Predictor = app.state.predictor
    return {"status": "ok", "labels": len(p.labels), "max_positions": p.max_positions, "device": str(p.device),
            "position_model": p.position_model is not None}


@app.get("/labels")
def labels():
    return {"labels": app.state.predictor.labels}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p: Predictor = app.state.predictor
    try:
        out = p.predict(req.fen, req.moves, req.cp, req.setup_move, req.decoding, req.apply_rules)
    except ValueError as e:  # illegal move in the line, etc.
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**out.__dict__)
