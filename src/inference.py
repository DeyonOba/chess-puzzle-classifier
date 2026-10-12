"""Inference for one puzzle: encode, predict, decode, optionally apply cook.py rules.

    from src.inference import Predictor
    p = Predictor("logs/train/runs/NAME/checkpoints/best.ckpt")
    p.predict(fen, moves=["e6e7", "b2b1", ...], cp=850)

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
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from src.models.puzzle_module import PuzzleLitModule
from src.utils.bitboard_encoder import board_to_bitboards, unpack_bitboards
from src.utils.label_groups import EXCLUSIVE_GROUPS, LabelLayout
from src.utils.thresholds import decode_grouped

LENGTH_TAGS = {2: "oneMove", 4: "short"}  # by number of plies incl. the setup move, as cook.py counts
OUTCOME_TAGS = EXCLUSIVE_GROUPS["outcome"]
MATE_IN_TAGS = EXCLUSIVE_GROUPS["mateIn"]


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
class Prediction:
    tags: list[str]
    probabilities: dict[str, float]
    thresholds: dict[str, float]
    rule_tags: dict[str, str] = field(default_factory=dict)   # tags set by rule, with the reason
    decoding: str = "argmax_groups"
    line_provided: bool = False
    positions: int = 1
    model: str = "line"


class Predictor:
    def __init__(self, ckpt_path: str, position_ckpt_path: str | None = None, device: str = "auto",
                 eval_results: str | None = None):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = PuzzleLitModule.load_from_checkpoint(ckpt_path, map_location="cpu").to(self.device).eval()
        self.position_model = (PuzzleLitModule.load_from_checkpoint(position_ckpt_path, map_location="cpu").to(self.device).eval()
                               if position_ckpt_path else None)
        self.labels = self.model.labels
        self.layout = LabelLayout(self.labels)
        self.index = {l: i for i, l in enumerate(self.labels)}
        self.max_positions = self.model.net.ply_embedding.num_embeddings if self.model.net.ply_embedding is not None \
            else self.model.net.head.ply_embedding.num_embeddings
        self.thresholds = self.model.thresholds.detach().cpu().numpy().copy()
        if eval_results:
            ev = json.loads(pathlib.Path(eval_results).read_text())
            self.thresholds = np.array([ev["per_class"][l]["threshold"] for l in self.labels], dtype=np.float32)

    # -- encoding --------------------------------------------------------------------------------
    def encode(self, fen: str, moves: list[str] | None) -> tuple[torch.Tensor, torch.Tensor, chess.Board]:
        board = chess.Board(fen)
        moves = moves or []
        packed = [board_to_bitboards(board)]
        for u in moves[: self.max_positions - 1]:
            board.push_uci(u)
            packed.append(board_to_bitboards(board))
        for u in moves[self.max_positions - 1:]:  # beyond T: still play out, for the mate rule
            board.push_uci(u)
        T = len(packed)
        arr = np.zeros((self.max_positions, 18), dtype=np.uint64); arr[:T] = np.stack(packed)
        boards = torch.from_numpy(unpack_bitboards(arr)).float()[None]  # (1, T_max, 18, 8, 8)
        return boards, torch.tensor([T]), board

    # -- prediction ------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, fen: str, moves: list[str] | None = None, cp: int | None = None, setup_move: str | None = None,
                decoding: str = "argmax_groups", apply_rules: bool = True) -> Prediction:
        if setup_move:
            b = chess.Board(fen); b.push_uci(setup_move); fen = b.fen()
        moves = list(moves or [])
        line = bool(moves)
        model = self.position_model if (not line and self.position_model is not None) else self.model
        boards, lengths, final_board = self.encode(fen, moves)
        if model is self.position_model:
            boards, lengths = boards[:, :1], torch.tensor([1])
        probs = model.probs(model(boards.to(self.device), lengths.to(self.device)).float()).cpu().numpy()[0]

        if decoding == "argmax_groups":
            preds = decode_grouped(probs[None], self.thresholds, self.layout)[0]
        elif decoding == "independent":
            preds = probs >= self.thresholds
        else:
            raise ValueError("decoding must be 'argmax_groups' or 'independent'")

        rule_tags: dict[str, str] = {}
        self.extra_tags: list[str] = []
        if apply_rules:
            preds, rule_tags = self._apply_rules(preds, moves, final_board, cp)

        tags = [l for l, on in zip(self.labels, preds) if on] + self.extra_tags
        return Prediction(
            tags=tags, probabilities={l: float(p) for l, p in zip(self.labels, probs)},
            thresholds={l: float(t) for l, t in zip(self.labels, self.thresholds)}, rule_tags=rule_tags,
            decoding=decoding, line_provided=line, positions=int(lengths[0]),
            model="position" if model is self.position_model else "line",
        )

    def _set_group(self, preds: np.ndarray, members: list[str], chosen: str | None):
        for m in members:
            if m in self.index:
                preds[self.index[m]] = (m == chosen)

    def _apply_rules(self, preds: np.ndarray, moves: list[str], final_board: chess.Board, cp: int | None):
        preds = preds.copy(); why: dict[str, str] = {}
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
                self.extra_tags = [oc]
            why[oc] = f"rule: cp={cp}"
            for m in MATE_IN_TAGS + EXCLUSIVE_GROUPS["matePattern"]:
                if m in self.index:
                    preds[self.index[m]] = False
        return preds, why
