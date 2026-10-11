"""Board encoding as packed 64-bit bitboards.

Each position is encoded as NUM_PLANES uint64 masks (one bit per square,
square index = rank * 8 + file, a1 = bit 0). This is 8x smaller than storing
unpacked (18, 8, 8) uint8 planes and unpacks vectorised with np.unpackbits.

Plane layout (index -> meaning):
    0-5   white pawn, knight, bishop, rook, queen, king
    6-11  black pawn, knight, bishop, rook, queen, king
    12    side to move is white (all ones / all zeros)
    13-16 castling rights: W-kingside, W-queenside, B-kingside, B-queenside
    17    en passant target square (single bit)

Unpacked layout is (plane, rank, file): arr[p, r, f] is square chess.square(f, r),
so arr[0, 1, :] are white pawns on rank 2 and arr[5, 0, 4] is a white king on e1.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

NUM_PLANES = 18
_FULL = np.uint64(chess.BB_ALL)
_EMPTY = np.uint64(0)


def board_to_bitboards(board: chess.Board) -> np.ndarray:
    """Encode a python-chess Board as a (NUM_PLANES,) uint64 array of bitboards."""
    out = np.empty(NUM_PLANES, dtype=np.uint64)
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    piece_masks = (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    for idx, mask in enumerate(piece_masks):
        out[idx] = mask & white
        out[idx + 6] = mask & black
    out[12] = _FULL if board.turn == chess.WHITE else _EMPTY
    out[13] = _FULL if board.has_kingside_castling_rights(chess.WHITE) else _EMPTY
    out[14] = _FULL if board.has_queenside_castling_rights(chess.WHITE) else _EMPTY
    out[15] = _FULL if board.has_kingside_castling_rights(chess.BLACK) else _EMPTY
    out[16] = _FULL if board.has_queenside_castling_rights(chess.BLACK) else _EMPTY
    out[17] = np.uint64(chess.BB_SQUARES[board.ep_square]) if board.ep_square is not None else _EMPTY
    return out


def unpack_bitboards(packed: np.ndarray) -> np.ndarray:
    """Unpack (..., NUM_PLANES) uint64 bitboards into (..., NUM_PLANES, 8, 8) uint8 planes.

    Output index order is (plane, rank, file). Works on any leading batch shape,
    so a whole (T, NUM_PLANES) sequence or (B, T, NUM_PLANES) batch unpacks in one call.
    """
    packed = np.ascontiguousarray(packed, dtype="<u8")
    bits = np.unpackbits(packed.view(np.uint8), axis=-1, bitorder="little")
    return bits.reshape(*packed.shape, 8, 8)


def board_to_tensor_fast(board: chess.Board) -> torch.Tensor:
    """Convert a Board into an (NUM_PLANES, 8, 8) uint8 tensor (rank-major, a1 at [0, 0])."""
    return torch.from_numpy(unpack_bitboards(board_to_bitboards(board)))


def encode_solution_line(fen: str, moves: str, max_positions: int) -> tuple[np.ndarray, int]:
    """Encode a Lichess puzzle as a sequence of packed positions.

    The CSV FEN is the position *before* the opponent's setup move, which is the
    first token of `moves`. Position 0 is the board after that setup move (what
    the solver sees); position k is the board after the k-th solution ply.
    Sequences longer than `max_positions` are truncated.

    Returns (packed, length) where packed is (max_positions, NUM_PLANES) uint64,
    zero-padded beyond `length`.
    """
    uci_moves = moves.split()
    board = chess.Board(fen)
    board.push_uci(uci_moves[0])

    length = min(len(uci_moves), max_positions)
    packed = np.zeros((max_positions, NUM_PLANES), dtype=np.uint64)
    packed[0] = board_to_bitboards(board)
    for k in range(1, length):
        board.push_uci(uci_moves[k])
        packed[k] = board_to_bitboards(board)
    return packed, length
