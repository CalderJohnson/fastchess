"""Miscellaneous utility functions for chess board and move encoding/decoding."""
import numpy as np
import chess

def encode_board(board: chess.Board):
    """Encodes a chess.Board into a 18x8x8 numpy array for neural network input."""
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        row = 7 - chess.square_rank(sq)
        col = chess.square_file(sq)

        offset = 0 if piece.color == chess.WHITE else 6
        piece_type = piece.piece_type - 1
        planes[offset + piece_type, row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0

    # Halfmove clock
    planes[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)

    return planes


SLIDING_DIRS = [
    (0, 1),  (0, -1), (1, 0),  (-1, 0),
    (1, 1),  (1, -1), (-1, 1), (-1, -1)
]

KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

def move_to_index(move: chess.Move):
    """Encodes a chess.Move into an index for policy output."""
    from_sq = move.from_square
    to_sq = move.to_square

    # From row/col to row/col
    fr = 7 - chess.square_rank(from_sq)
    fc = chess.square_file(from_sq)
    tr = 7 - chess.square_rank(to_sq)
    tc = chess.square_file(to_sq)

    # Diff row/col
    dr = tr - fr
    dc = tc - fc

    # Knight moves
    for i, (r, c) in enumerate(KNIGHT_DIRS):
        if (dr, dc) == (r, c):
            return fr, fc, 56 + i

    # Underpromotions
    if move.promotion and move.promotion != chess.QUEEN:
        promo_map = {chess.ROOK:0, chess.BISHOP:1, chess.KNIGHT:2}
        promo = promo_map[move.promotion]
        if dc == 0:
            return fr, fc, 64 + promo
        elif dc == -1:
            return fr, fc, 67 + promo
        elif dc == 1:
            return fr, fc, 70 + promo

    # Sliding moves
    for i, (r, c) in enumerate(SLIDING_DIRS):
        if r == 0: dist = dc // c if c != 0 else None
        elif c == 0: dist = dr // r if r != 0 else None
        else: dist = dr // r if dr == dc * (r/c) else None

        if dist and 1 <= dist <= 7:
            return fr, fc, i*7 + (dist - 1)

    raise ValueError("Illegal move encoding")

def index_to_move(fr, fc, plane, board: chess.Board):
    """Decodes an index from policy output into a chess.Move."""
    # Convert array coords → square
    from_sq = chess.square(fc, 7 - fr)

    # Sliding moves
    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        dr, dc = SLIDING_DIRS[dir_idx]

        tr = fr + dr * dist
        tc = fc + dc * dist

        if 0 <= tr < 8 and 0 <= tc < 8:
            to_sq = chess.square(tc, 7 - tr)
            return chess.Move(from_sq, to_sq)

    # Knight moves
    elif plane < 64:
        dr, dc = KNIGHT_DIRS[plane - 56]
        tr = fr + dr
        tc = fc + dc

        if 0 <= tr < 8 and 0 <= tc < 8:
            to_sq = chess.square(tc, 7 - tr)
            return chess.Move(from_sq, to_sq)

    # Underpromotions
    else:
        promo = (plane - 64) % 3
        promo_piece = [chess.ROOK, chess.BISHOP, chess.KNIGHT][promo]

        group = (plane - 64) // 3
        dc = [-1, 0, 1][group]

        tr = fr - 1 if board.turn == chess.WHITE else fr + 1
        tc = fc + dc

        if 0 <= tr < 8 and 0 <= tc < 8:
            to_sq = chess.square(tc, 7 - tr)
            return chess.Move(from_sq, to_sq, promotion=promo_piece)

    return None

def print_board(board):
    """Prints the board in a human-readable format."""
    print("\n" + board.unicode(borders=True))

def parse_move(board, move_str):
    """Parses a UCI move string and returns a chess.Move if legal."""
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
    except:
        pass
    return None

def get_legal_mask(board: chess.Board):
    """Returns a boolean mask of legal moves for the given board."""
    legal_mask = np.zeros(4672, dtype=bool)
    for move in board.legal_moves:
        fr, fc, p = move_to_index(move)
        idx = fr*8*73 + fc*73 + p
        legal_mask[idx] = True
    return legal_mask
