import chess
import numpy as np

from src.utils.bitboard_encoder import (
    NUM_PLANES,
    board_to_bitboards,
    board_to_tensor_fast,
    encode_solution_line,
    unpack_bitboards,
)

PIECE_PLANE = {
    (chess.WHITE, chess.PAWN): 0, (chess.WHITE, chess.KNIGHT): 1, (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3, (chess.WHITE, chess.QUEEN): 4, (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6, (chess.BLACK, chess.KNIGHT): 7, (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9, (chess.BLACK, chess.QUEEN): 10, (chess.BLACK, chess.KING): 11,
}


def reference_encode(board: chess.Board) -> np.ndarray:
    """Slow square-by-square encoder used as ground truth."""
    arr = np.zeros((NUM_PLANES, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        arr[PIECE_PLANE[(piece.color, piece.piece_type)], chess.square_rank(sq), chess.square_file(sq)] = 1
    arr[12] = board.turn == chess.WHITE
    arr[13] = board.has_kingside_castling_rights(chess.WHITE)
    arr[14] = board.has_queenside_castling_rights(chess.WHITE)
    arr[15] = board.has_kingside_castling_rights(chess.BLACK)
    arr[16] = board.has_queenside_castling_rights(chess.BLACK)
    if board.ep_square is not None:
        arr[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    return arr


FENS = [
    chess.STARTING_FEN,
    "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
    "8/8/8/3k4/8/8/4K3/8 w - - 0 1",
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # en passant available on d6
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",  # partial castling rights
]


def test_pack_unpack_matches_reference():
    for fen in FENS:
        board = chess.Board(fen)
        got = unpack_bitboards(board_to_bitboards(board))
        assert got.shape == (NUM_PLANES, 8, 8)
        np.testing.assert_array_equal(got, reference_encode(board), err_msg=fen)


def test_orientation_rank_major_file_minor():
    arr = board_to_tensor_fast(chess.Board()).numpy()
    assert arr[0, 1].sum() == 8 and arr[0].sum() == 8  # white pawns on rank 2
    assert arr[6, 6].sum() == 8  # black pawns on rank 7
    assert arr[5, 0, 4] == 1  # white king e1
    assert arr[11, 7, 4] == 1  # black king e8
    assert arr[3, 0, 0] == 1 and arr[3, 0, 7] == 1  # white rooks a1, h1


def test_en_passant_plane_is_consistent_with_piece_planes():
    board = chess.Board()
    board.push_uci("e2e4")
    arr = board_to_tensor_fast(board).numpy()
    assert arr[17, 2, 4] == 1 and arr[17].sum() == 1  # e3
    assert arr[0, 3, 4] == 1  # pawn now on e4, same file column as ep square


def test_unpack_batched_shapes():
    packed = np.stack([board_to_bitboards(chess.Board(f)) for f in FENS])  # (5, 18)
    out = unpack_bitboards(packed)
    assert out.shape == (len(FENS), NUM_PLANES, 8, 8)
    out3 = unpack_bitboards(packed.reshape(1, len(FENS), NUM_PLANES))
    assert out3.shape == (1, len(FENS), NUM_PLANES, 8, 8)


def test_encode_solution_line_starts_after_setup_move():
    fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
    moves = "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1"
    packed, length = encode_solution_line(fen, moves, max_positions=12)
    assert packed.shape == (12, NUM_PLANES) and length == 6
    board = chess.Board(fen)
    board.push_uci("f2g3")
    np.testing.assert_array_equal(packed[0], board_to_bitboards(board))
    for k, uci in enumerate(moves.split()[1:], start=1):
        board.push_uci(uci)
        np.testing.assert_array_equal(packed[k], board_to_bitboards(board))
    assert not packed[6:].any()


def test_encode_solution_line_truncates():
    fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
    moves = "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1"
    packed, length = encode_solution_line(fen, moves, max_positions=3)
    assert packed.shape == (3, NUM_PLANES) and length == 3
    assert packed.any(axis=1).all()


def test_extra_planes_attack_pin_hanging_check():
    from src.utils.bitboard_encoder import NUM_PLANES_EXTENDED, extra_bitboards
    # white Re1 pins black Ne4 against Ke8? use a clean one: white Bb5 pins black Nc6 to Ke8
    board = chess.Board("4k3/8/2n5/1B6/8/8/8/4K3 b - - 0 1")
    x = unpack_bitboards(board_to_bitboards(board, extended=True))
    assert x.shape == (NUM_PLANES_EXTENDED, 8, 8)
    np.testing.assert_array_equal(x[:18], unpack_bitboards(board_to_bitboards(board)))  # base planes unchanged
    assert x[18, 5, 2] == 1          # c6 attacked by white (the bishop)
    assert x[21, 5, 2] == 1 and x[21].sum() == 1   # black knight on c6 is pinned
    assert x[23, 5, 2] == 1          # ...and hanging (attacked, undefended)
    assert x[22].sum() == 0          # white bishop is not attacked
    assert x[24].sum() == 0 and x[25].sum() == 0   # no check
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R b K - 0 1"); board.push_uci("e8d8"); board.push_uci("h1h8")
    x = unpack_bitboards(board_to_bitboards(board, extended=True))
    assert x[24, 7, 7] == 1 and x[25].all()        # rook on h8 gives check; in_check plane is all ones


def test_encode_solution_line_extended_shape():
    packed, length = encode_solution_line("r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24", "f2g3 e6e7 b2b1", 12, extended=True)
    assert packed.shape == (12, 26) and length == 3 and packed[:3, 18:].any()
