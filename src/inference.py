"""Inference for puzzles: encode, predict, decode, optionally apply cook.py rules.

    from src.inference import Predictor, PredictItem
    p = Predictor("logs/train/runs/NAME/checkpoints/best.ckpt")
    p.predict(fen, moves=["e6e7", "b2b1", ...], cp=850)
    p.predict_batch([PredictItem(fen, moves), ...], include_embedding=True)

`fen` is the position the solver sees (side to move = solver). In the Lichess
CSV that is the FEN *after* the first move of the Moves column; pass
`setup_move=` to apply it for you. `moves` is the solution line in UCI from
that position. Without `moves` the model sees a single position (T=1); if a
position-only checkpoint is configured it is used for those requests.

Rules (opt-out with apply_rules=False) override predictions where cook.py's
tag is a pure function of the inputs:
  - length tags from the number of solution plies
  - mate / mateInN when the line ends in checkmate (and no mate tags otherwise)
  - outcome (mate / crushing / advantage / equality) from `cp` when given

`include_embedding=True` returns the encoder's pooled feature vector per
puzzle (see PuzzleSequenceClassifier.forward_with_embedding), for similarity
search over puzzles.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from src.models.puzzle_module import PuzzleLitModule
from src.utils.bitboard_encoder import NUM_PLANES_EXTENDED, board_to_bitboards, unpack_bitboards
from src.utils.label_groups import EXCLUSIVE_GROUPS, LabelLayout
from src.utils.thresholds import decode_grouped

LENGTH_TAGS = {2: "oneMove", 4: "short"}  # by number of plies incl. the setup move, as cook.py counts
OUTCOME_TAGS = EXCLUSIVE_GROUPS["outcome"]
MATE_IN_TAGS = EXCLUSIVE_GROUPS["mateIn"]
DECODINGS = ("argmax_groups", "independent")


def length_tag(solution_plies: int) -> str:
    """cook.py counts mainline length including the setup move: 2 -> oneMove, 4 -> short, >=8 -> veryLong, else long."""
    n = solution_plies + 1
    if n == 2:
        return "oneMove"
    if n == 4:
        return "short"
    if n >= 8:
        return "veryLong"
    return "long"


def outcome_from_cp(cp: int) -> str:
    return "crushing" if cp > 600 else "advantage" if cp > 200 else "equality"


@dataclass
class PredictItem:
    fen: str
    moves: list[str] | None = None
    setup_move: str | None = None
    cp: int | None = None

    def check(self) -> str | None:
        """Why this item cannot be predicted (invalid FEN, illegal move), or None."""
        try:
            board = chess.Board(self.fen)
            if self.setup_move:
                board.push_uci(self.setup_move)
            for u in self.moves or []:
                board.push_uci(u)
        except ValueError as e:
            return str(e)
        return None


@dataclass
class Prediction:
    tags: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    rule_tags: dict[str, str] = field(default_factory=dict)   # tags set by rule, with the reason
    decoding: str = "argmax_groups"
    line_provided: bool = False
    positions: int = 1
    model: str = "line"
    embedding: list[float] | None = None


@dataclass
class _Encoded:
    packed: np.ndarray          # (T_max, planes) uint64
    positions: int
    final_board: chess.Board
    moves: list[str]
    cp: int | None


class Predictor:
    def __init__(self, ckpt_path: str, position_ckpt_path: str | None = None, device: str = "auto",
                 eval_results: str | None = None):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.checkpoint = pathlib.Path(ckpt_path).resolve().parent.parent.name
        self.model = PuzzleLitModule.load_from_checkpoint(ckpt_path, map_location="cpu").to(self.device).eval()
        self.position_model = (PuzzleLitModule.load_from_checkpoint(position_ckpt_path, map_location="cpu").to(self.device).eval()
                               if position_ckpt_path else None)
        if self.position_model is not None and list(self.position_model.labels) != list(self.model.labels):
            raise ValueError("the position checkpoint has a different label list than the line checkpoint")
        self.labels = self.model.labels
        self.layout = LabelLayout(self.labels)
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.max_positions = self.model.net.ply_embedding.num_embeddings if self.model.net.ply_embedding is not None \
            else self.model.net.head.ply_embedding.num_embeddings
        self.extended = self.model.net.in_planes == NUM_PLANES_EXTENDED  # checkpoint trained on attack-map planes
        self.embedding_dim = self.model.net.encoder.hidden_dim
        self.thresholds = self.model.thresholds.detach().cpu().numpy().copy()
        if eval_results:
            ev = json.loads(pathlib.Path(eval_results).read_text())
            self.thresholds = np.array([ev["per_class"][l]["threshold"] for l in self.labels], dtype=np.float32)
        self.position_thresholds = (self.position_model.thresholds.detach().cpu().numpy().copy()
                                    if self.position_model is not None else self.thresholds)

    # -- encoding --------------------------------------------------------------------------------
    def encode(self, fen: str, moves: list[str] | None) -> tuple[torch.Tensor, torch.Tensor, chess.Board]:
        enc = self._encode(PredictItem(fen, moves))
        boards = torch.from_numpy(unpack_bitboards(enc.packed)).float()[None]  # (1, T_max, planes, 8, 8)
        return boards, torch.tensor([enc.positions]), enc.final_board

    def _encode(self, item: PredictItem) -> _Encoded:
        board = chess.Board(item.fen)
        if item.setup_move:
            board.push_uci(item.setup_move)
        moves = list(item.moves or [])
        packed = [board_to_bitboards(board, self.extended)]
        for u in moves[: self.max_positions - 1]:
            board.push_uci(u)
            packed.append(board_to_bitboards(board, self.extended))
        for u in moves[self.max_positions - 1:]:  # beyond T: still play out, for the mate rule
            board.push_uci(u)
        T = len(packed)
        arr = np.zeros((self.max_positions, packed[0].shape[0]), dtype=np.uint64); arr[:T] = np.stack(packed)
        return _Encoded(arr, T, board, moves, item.cp)

    # -- prediction ------------------------------------------------------------------------------
    def predict(self, fen: str, moves: list[str] | None = None, cp: int | None = None, setup_move: str | None = None,
                decoding: str = "argmax_groups", apply_rules: bool = True, include_embedding: bool = False) -> Prediction:
        return self.predict_batch([PredictItem(fen, moves, setup_move, cp)], decoding, apply_rules, include_embedding)[0]

    @torch.no_grad()
    def predict_batch(self, items: list[PredictItem], decoding: str = "argmax_groups", apply_rules: bool = True,
                      include_embedding: bool = False) -> list[Prediction]:
        """One forward pass per model for the whole batch. Items with a line go to the line model,
        FEN-only items to the position model when one is loaded."""
        if decoding not in DECODINGS:
            raise ValueError(f"decoding must be one of {DECODINGS}")
        encoded = [self._encode(item) for item in items]
        out: list[Prediction | None] = [None] * len(items)
        use_position = [not e.moves and self.position_model is not None for e in encoded]
        for model_name, model, thresholds in (("line", self.model, self.thresholds),
                                              ("position", self.position_model, self.position_thresholds)):
            idx = [i for i, p in enumerate(use_position) if (p if model_name == "position" else not p)]
            if not idx or model is None:
                continue
            packed = np.stack([encoded[i].packed for i in idx])
            lengths = np.array([encoded[i].positions for i in idx])
            if model_name == "position":
                packed, lengths = packed[:, :1], np.ones_like(lengths)
            boards = torch.from_numpy(unpack_bitboards(packed)).float().to(self.device)
            raw, emb = model.net.forward_with_embedding(boards, torch.from_numpy(lengths).to(self.device))
            probs = model.probs(raw.float()).cpu().numpy()
            embeddings = emb.float().cpu().numpy()
            preds = decode_grouped(probs, thresholds, self.layout) if decoding == "argmax_groups" else probs >= thresholds
            for row, i in enumerate(idx):
                e = encoded[i]
                p, rule_tags, extra = (self._apply_rules(preds[row], e.moves, e.final_board, e.cp) if apply_rules
                                       else (preds[row], {}, []))
                out[i] = Prediction(
                    tags=[l for l, on in zip(self.labels, p) if on] + extra,
                    probabilities={l: float(v) for l, v in zip(self.labels, probs[row])},
                    thresholds={l: float(t) for l, t in zip(self.labels, thresholds)}, rule_tags=rule_tags,
                    decoding=decoding, line_provided=bool(e.moves), positions=int(lengths[row]),
                    model=model_name, embedding=embeddings[row].tolist() if include_embedding else None,
                )
        return [p for p in out if p is not None]

    def _set_group(self, preds: np.ndarray, members: list[str], chosen: str | None):
        for m in members:
            if m in self.index:
                preds[self.index[m]] = (m == chosen)

    def _apply_rules(self, preds: np.ndarray, moves: list[str], final_board: chess.Board, cp: int | None):
        preds = preds.copy(); why: dict[str, str] = {}; extra: list[str] = []
        if moves:
            lt = length_tag(len(moves))
            self._set_group(preds, EXCLUSIVE_GROUPS["length"], lt); why[lt] = f"rule: {len(moves)} solution plies"
            if final_board.is_checkmate():
                depth = (len(moves) + 1) // 2
                mate_in = f"mateIn{min(depth, 5)}"
                self._set_group(preds, MATE_IN_TAGS, mate_in); why[mate_in] = "rule: line ends in checkmate"
                self._set_group(preds, OUTCOME_TAGS, "mate"); why["mate"] = "rule: line ends in checkmate"
            else:
                self._set_group(preds, MATE_IN_TAGS, None)
                if cp is None and "mate" in self.index and preds[self.index["mate"]]:
                    preds[self.index["mate"]] = False  # a full line that does not mate cannot be a mate puzzle
                    for m in EXCLUSIVE_GROUPS["matePattern"]:
                        if m in self.index:
                            preds[self.index[m]] = False
        if cp is not None and not (moves and final_board.is_checkmate()):
            oc = outcome_from_cp(cp)
            if oc in self.index:
                self._set_group(preds, OUTCOME_TAGS, oc)
            else:  # merged vocabulary: the model label is 'winning'; the fine-grained tag is reported as an extra
                self._set_group(preds, OUTCOME_TAGS, "winning" if oc != "equality" else "equality")
                extra = [oc]
            why[oc] = f"rule: cp={cp}"
            for m in MATE_IN_TAGS + EXCLUSIVE_GROUPS["matePattern"]:
                if m in self.index:
                    preds[self.index[m]] = False
        return preds, why, extra
