"""FastAPI service for puzzle theme prediction.

    CPC_CKPT=logs/train/runs/NAME/checkpoints/best.ckpt uv run uvicorn src.app:app --port 8000
    curl -X POST localhost:8000/predict -H 'content-type: application/json' \
         -d '{"fen": "...", "moves": ["e6e7", "b2b1"], "cp": 850}'
    curl -X POST localhost:8000/predict_batch -H 'content-type: application/json' \
         -d '{"items": [{"fen": "...", "moves": ["e6e7"]}], "include_embedding": true}'

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

from src.inference import PredictItem, Predictor

MAX_BATCH = 512


class PuzzleInput(BaseModel):
    fen: str = Field(..., description="position the solver sees (side to move = solver)")
    moves: list[str] | None = Field(None, description="solution line in UCI from `fen`")
    setup_move: str | None = Field(None, description="if `fen` is the Lichess CSV FEN, the first move of the Moves column")
    cp: int | None = Field(None, description="engine centipawn score after the solution, solver POV; sets the outcome tag by rule")

    @field_validator("fen")
    @classmethod
    def _valid_fen(cls, v: str) -> str:
        try:
            chess.Board(v)
        except ValueError as e:
            raise ValueError(f"invalid FEN: {e}")
        return v

    def item(self) -> PredictItem:
        return PredictItem(self.fen, self.moves, self.setup_move, self.cp)


class PredictRequest(PuzzleInput):
    decoding: str = Field("argmax_groups", pattern="^(argmax_groups|independent)$")
    apply_rules: bool = True
    include_embedding: bool = False


class PredictResponse(BaseModel):
    tags: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    rule_tags: dict[str, str]
    decoding: str
    line_provided: bool
    positions: int
    model: str
    embedding: list[float] | None = None


class PredictBatchRequest(BaseModel):
    items: list[PuzzleInput] = Field(..., min_length=1, max_length=MAX_BATCH)
    decoding: str = Field("argmax_groups", pattern="^(argmax_groups|independent)$")
    apply_rules: bool = True
    include_embedding: bool = False


class BatchItemResponse(BaseModel):
    prediction: PredictResponse | None = None
    error: str | None = None


class PredictBatchResponse(BaseModel):
    checkpoint: str
    embedding_dim: int
    results: list[BatchItemResponse]


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
            "position_model": p.position_model is not None, "checkpoint": p.checkpoint, "embedding_dim": p.embedding_dim,
            "max_batch": MAX_BATCH}


@app.get("/labels")
def labels():
    return {"labels": app.state.predictor.labels}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    p: Predictor = app.state.predictor
    try:
        out = p.predict(req.fen, req.moves, req.cp, req.setup_move, req.decoding, req.apply_rules, req.include_embedding)
    except ValueError as e:  # illegal move in the line, etc.
        raise HTTPException(status_code=422, detail=str(e))
    return PredictResponse(**out.__dict__)


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    """One forward pass for up to MAX_BATCH puzzles. A bad item (illegal move) gets an `error`
    instead of failing the batch."""
    p: Predictor = app.state.predictor
    items = [inp.item() for inp in req.items]
    errors = [item.check() for item in items]
    valid = [item for item, err in zip(items, errors) if err is None]
    predictions = iter(p.predict_batch(valid, req.decoding, req.apply_rules, req.include_embedding) if valid else [])
    results = [BatchItemResponse(error=err) if err else BatchItemResponse(prediction=PredictResponse(**next(predictions).__dict__))
               for err in errors]
    return PredictBatchResponse(checkpoint=p.checkpoint, embedding_dim=p.embedding_dim, results=results)
