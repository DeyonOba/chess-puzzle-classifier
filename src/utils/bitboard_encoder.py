import chess
import numpy as np
import torch

def board_to_tensor_fast(board: chess.Board) -> torch.Tensor:
    """Converts a python-chess Board into an 18x8x8 uint8 tensor using bitwise operations."""
    arr = np.zeros((18, 8, 8), dtype=np.uint8)

    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]

    piece_masks = [
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
    ]

    for idx, p_mask in enumerate(piece_masks):
        w_mask = p_mask & white
        if w_mask:
            arr[idx] = np.unpackbits(
                np.array([w_mask], dtype='>u8').view(np.uint8)
            ).reshape(8, 8)[::-1]

        b_mask = p_mask & black
        if b_mask:
            arr[idx + 6] = np.unpackbits(
                np.array([b_mask], dtype='>u8').view(np.uint8)
            ).reshape(8, 8)[::-1]

    if board.turn == chess.WHITE:
        arr[12, :, :] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        arr[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        arr[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        arr[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        arr[16, :, :] = 1

    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        arr[17, row, col] = 1

    return torch.from_numpy(arr)