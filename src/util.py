"""Miscellaneous utility functions for chess board and move encoding/decoding."""
import torch
import numpy as np
import chess


class ChessPositionEncoder:
    """Encodes chess positions into 18-channel tensors."""
    
    PIECE_TO_CHANNEL = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    @staticmethod
    def encode_board(board):
        """
        Encode board state as 18x8x8 tensor:
        - Channels 0-5: White pieces (P,N,B,R,Q,K)
        - Channels 6-11: Black pieces
        - Channel 12: Repetition counter (1 if position seen before)
        - Channel 13: Current player (1 if white to move)
        - Channel 14: White castling kingside
        - Channel 15: White castling queenside
        - Channel 16: Black castling kingside
        - Channel 17: Black castling queenside
        """
        state = np.zeros((18, 8, 8), dtype=np.float32)
        
        # Encode pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                channel = ChessPositionEncoder.PIECE_TO_CHANNEL[piece.piece_type]
                if not piece.color:  # Black
                    channel += 6
                state[channel, rank, file] = 1.0
        
        # Encode metadata
        state[13, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
        state[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        state[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        state[16, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        state[17, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        return state


class MoveEncoder:
    """Encodes chess moves as indices for policy head."""
    
    # Move type definitions
    # (dr, dc) where row 0 = rank 7, row increases downward
    SLIDING_DIRS = [
        (0, 1),   # East (right)
        (0, -1),  # West (left)
        (1, 0),   # South (down, decreasing rank)
        (-1, 0),  # North (up, increasing rank)
        (1, 1),   # SE
        (1, -1),  # SW
        (-1, 1),  # NE
        (-1, -1)  # NW
    ]
    KNIGHT_DIRS = [
        (1, 2), (2, 1), (2, -1), (1, -2),
        (-1, -2), (-2, -1), (-2, 1), (-1, 2)
    ]
    
    def __init__(self):
        self.num_moves = 4672  # 8 * 8 * 73
    
    def encode_move(self, move):
        """
        Convert chess.Move to policy index.
        Returns index in range [0, 4672) as (from_row * 8 * 73) + (from_col * 73) + move_type
        
        Move types (73 total):
        - 0-55: Sliding moves (8 directions × 7 distances)
        - 56-63: Knight moves (8 possible L-shapes)
        - 64-66: Underpromotion straight (rook, bishop, knight)
        - 67-69: Underpromotion capture left (rook, bishop, knight)
        - 70-72: Underpromotion capture right (rook, bishop, knight)
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        # Convert to row/col (from white's perspective, rank 0 = row 7)
        fr = 7 - chess.square_rank(from_sq)
        fc = chess.square_file(from_sq)
        tr = 7 - chess.square_rank(to_sq)
        tc = chess.square_file(to_sq)
        
        # Calculate deltas
        dr = tr - fr
        dc = tc - fc
        
        # Check knight moves
        for i, (r, c) in enumerate(self.KNIGHT_DIRS):
            if (dr, dc) == (r, c):
                move_type = 56 + i
                return fr * 8 * 73 + fc * 73 + move_type
        
        # Check underpromotions (pawn moves to rank 0 or 7)
        if move.promotion and move.promotion != chess.QUEEN:
            promo_map = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}
            promo_idx = promo_map[move.promotion]
            
            if dc == 0:  # Straight
                move_type = 64 + promo_idx
            elif dc == -1:  # Capture left
                move_type = 67 + promo_idx
            elif dc == 1:  # Capture right
                move_type = 70 + promo_idx
            else:
                return -1  # Invalid underpromotion
            
            return fr * 8 * 73 + fc * 73 + move_type
        
        # Check sliding moves (including queen promotions and regular moves)
        for i, (r, c) in enumerate(self.SLIDING_DIRS):
            # Check if move aligns with this direction
            if r == 0 and c != 0:
                # Horizontal move
                if dr == 0 and dc != 0:
                    if (dc > 0 and c > 0) or (dc < 0 and c < 0):
                        dist = abs(dc)
                        if 1 <= dist <= 7:
                            move_type = i * 7 + (dist - 1)
                            return fr * 8 * 73 + fc * 73 + move_type
            elif c == 0 and r != 0:
                # Vertical move
                if dc == 0 and dr != 0:
                    if (dr > 0 and r > 0) or (dr < 0 and r < 0):
                        dist = abs(dr)
                        if 1 <= dist <= 7:
                            move_type = i * 7 + (dist - 1)
                            return fr * 8 * 73 + fc * 73 + move_type
            else:
                # Diagonal move
                if dr != 0 and dc != 0 and abs(dr) == abs(dc):
                    if (dr > 0) == (r > 0) and (dc > 0) == (c > 0):
                        dist = abs(dr)
                        if 1 <= dist <= 7:
                            move_type = i * 7 + (dist - 1)
                            return fr * 8 * 73 + fc * 73 + move_type
        
        # Should not reach here for valid chess moves
        return -1
    
    def decode_move(self, policy_idx, board=None):
        """
        Convert policy index back to chess.Move.
        
        Args:
            policy_idx: Index in range [0, 4672)
            board: Optional chess.Board to validate the move
            
        Returns:
            chess.Move object, or None if invalid
        """
        if policy_idx < 0 or policy_idx >= self.num_moves:
            return None
        
        # Decompose index into from_row, from_col, move_type
        from_row = policy_idx // (8 * 73)
        remainder = policy_idx % (8 * 73)
        from_col = remainder // 73
        move_type = remainder % 73
        
        # Convert from_row, from_col to square (remember row 0 = rank 7)
        from_rank = 7 - from_row
        from_file = from_col
        from_sq = chess.square(from_file, from_rank)
        
        # Decode move_type to get to_square and promotion
        to_sq = None
        promotion = None
        
        # Knight moves (56-63)
        if 56 <= move_type <= 63:
            knight_idx = move_type - 56
            dr, dc = self.KNIGHT_DIRS[knight_idx]
            to_row = from_row + dr
            to_col = from_col + dc
            
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_rank = 7 - to_row
                to_file = to_col
                to_sq = chess.square(to_file, to_rank)
        
        # Underpromotions (64-72)
        elif 64 <= move_type <= 72:
            promo_types = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
            
            if 64 <= move_type <= 66:  # Straight
                promo_idx = move_type - 64
                # White pawns move up (row decreases), black pawns move down (row increases)
                # from_rank tells us which side: rank 1 = white pawn, rank 6 = black pawn
                dr = -1 if from_rank == 6 else 1
                dc = 0
            elif 67 <= move_type <= 69:  # Capture left
                promo_idx = move_type - 67
                dr = -1 if from_rank == 6 else 1
                dc = -1
            else:  # 70-72: Capture right
                promo_idx = move_type - 70
                dr = -1 if from_rank == 6 else 1
                dc = 1
            
            promotion = promo_types[promo_idx]
            to_row = from_row + dr
            to_col = from_col + dc
            
            if 0 <= to_row < 8 and 0 <= to_col < 8:
                to_rank = 7 - to_row
                to_file = to_col
                to_sq = chess.square(to_file, to_rank)
        
        # Sliding moves (0-55)
        else:
            direction_idx = move_type // 7
            distance = (move_type % 7) + 1
            
            if direction_idx < len(self.SLIDING_DIRS):
                dr, dc = self.SLIDING_DIRS[direction_idx]
                to_row = from_row + dr * distance
                to_col = from_col + dc * distance
                
                if 0 <= to_row < 8 and 0 <= to_col < 8:
                    to_rank = 7 - to_row
                    to_file = to_col
                    to_sq = chess.square(to_file, to_rank)
                    
                    # Check for queen promotion (pawn reaching last rank)
                    if board:
                        piece = board.piece_at(from_sq)
                        if piece and piece.piece_type == chess.PAWN:
                            if (piece.color == chess.WHITE and to_rank == 7) or \
                               (piece.color == chess.BLACK and to_rank == 0):
                                promotion = chess.QUEEN
        
        if to_sq is None:
            return None
        
        move = chess.Move(from_sq, to_sq, promotion=promotion)
        
        # Validate against board if provided
        if board and move not in board.legal_moves:
            return None
        
        return move
    
    def get_legal_mask(self, board):
        """Return boolean mask of legal moves."""
        mask = torch.zeros(self.num_moves, dtype=torch.bool)
        for move in board.legal_moves:
            idx = self.encode_move(move)
            if idx >= 0:
                mask[idx] = True
        return mask


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
