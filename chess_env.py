"""
Chess <-> tensor encoding.

Board encoding (18 planes, 8x8):
  0-5   : my pieces   (pawn, knight, bishop, rook, queen, king)
  6-11  : opp pieces  (pawn, knight, bishop, rook, queen, king)
  12    : my kingside castling right
  13    : my queenside castling right
  14    : opp kingside castling right
  15    : opp queenside castling right
  16    : en-passant target square
  17    : no-progress (halfmove clock) count, normalized to [0,1]

The board is always encoded from the perspective of the side to move:
if it's Black's turn we use board.mirror() (python-chess flips the board
vertically AND swaps piece colors), so the network only ever sees
"White to move" - style positions. This halves the input space the
network has to learn and matches the spirit of the AlphaZero encoding
(we use a single current position instead of the full 8-position history
used in the paper, to keep this implementation compact).

Move encoding (4672 = 8*8*73):
  For each origin square (64) there are 73 possible move "planes":
    0-55  : queen-like moves - 8 directions x 7 distances
    56-63 : knight moves - 8 L-shaped jumps
    64-72 : underpromotions - 3 directions x 3 piece types (N, B, R)
  Normal promotions to queen are encoded as an ordinary queen-like move
  of magnitude 1 in the forward/diagonal direction; index_to_move fills
  in promotion=QUEEN automatically when a pawn reaches the last rank.
"""

import numpy as np
import chess

from config import NUM_PLANES, ACTION_SIZE

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# 8 queen-like directions as (delta_file, delta_rank)
QUEEN_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]

KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# underpromotion directions (delta_file only; pawn always moves +1 rank
# in mirrored/"my perspective" space) and piece order
UNDERPROMO_DIRS = [0, -1, 1]  # forward, capture-left, capture-right
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _mirrored_view(board: chess.Board):
    """Return (view_board, was_mirrored). view_board always has White to move."""
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a board into an (18, 8, 8) float32 tensor, side-to-move perspective."""
    view, _ = _mirrored_view(board)
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for i, pt in enumerate(PIECE_TYPES):
        for sq in view.pieces(pt, chess.WHITE):
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[i, r, f] = 1.0
        for sq in view.pieces(pt, chess.BLACK):
            r, f = chess.square_rank(sq), chess.square_file(sq)
            planes[6 + i, r, f] = 1.0

    if view.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if view.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if view.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if view.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0

    if view.ep_square is not None:
        r, f = chess.square_rank(view.ep_square), chess.square_file(view.ep_square)
        planes[16, r, f] = 1.0

    planes[17, :, :] = min(view.halfmove_clock, 100) / 100.0

    return planes


def _direction_index(dx: int, dy: int):
    step = max(abs(dx), abs(dy))
    ux, uy = dx // step, dy // step
    return QUEEN_DIRECTIONS.index((ux, uy)), step


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    """Encode a legal move (in the ORIGINAL board's frame) into an action index."""
    _, mirrored = _mirrored_view(board)
    if mirrored:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )

    from_sq, to_sq = move.from_square, move.to_square
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    dx, dy = tf - ff, tr - fr

    if move.promotion is not None and move.promotion != chess.QUEEN:
        dir_idx = UNDERPROMO_DIRS.index(dx)
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        plane = 64 + dir_idx * 3 + piece_idx
    elif (dx, dy) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((dx, dy))
    else:
        dir_idx, dist = _direction_index(dx, dy)
        plane = dir_idx * 7 + (dist - 1)

    return from_sq * 73 + plane


def index_to_move(board: chess.Board, action: int) -> chess.Move:
    """Decode an action index into a legal chess.Move on the ORIGINAL board."""
    view, mirrored = _mirrored_view(board)

    from_sq, plane = divmod(action, 73)
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    promotion = None

    if plane < 56:
        dir_idx, dist = divmod(plane, 7)
        dx, dy = QUEEN_DIRECTIONS[dir_idx]
        tf, tr = ff + dx * (dist + 1), fr + dy * (dist + 1)
    elif plane < 64:
        dx, dy = KNIGHT_DELTAS[plane - 56]
        tf, tr = ff + dx, fr + dy
    else:
        u = plane - 64
        dir_idx, piece_idx = divmod(u, 3)
        dx = UNDERPROMO_DIRS[dir_idx]
        tf, tr = ff + dx, fr + 1
        promotion = UNDERPROMO_PIECES[piece_idx]

    to_sq = chess.square(tf, tr)

    # queen promotion is implicit: pawn reaching the back rank without
    # an explicit underpromotion plane always promotes to queen
    piece = view.piece_at(from_sq)
    if promotion is None and piece is not None and piece.piece_type == chess.PAWN and tr == 7:
        promotion = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promotion)

    if mirrored:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
    return move


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of shape (ACTION_SIZE,) marking legal moves."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_action(board, move)] = 1.0
    return mask


def game_result(board: chess.Board) -> float:
    """Return result from the perspective of the player to move BEFORE this
    call was reached being checked as terminal, i.e. call only when
    board.is_game_over() is True. Returns +1/-1/0 from the perspective of
    the side that is about to move (the side that just got mated loses)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
