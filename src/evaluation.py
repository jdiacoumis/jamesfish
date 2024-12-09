import chess
from typing import Dict

class Evaluator:
    # Piece values in centipawns (100 = 1 pawn)
    PIECE_VALUES: Dict[chess.PieceType, int] = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Bonus for controlling center squares
    CENTER_SQUARES = {chess.E4, chess.E5, chess.D4, chess.D5}
    CENTER_CONTROL_BONUS = 10
    
    # Bonus for piece mobility (per legal move)
    MOBILITY_BONUS = 5
    
    # Penalty for doubled pawns (same file)
    DOUBLED_PAWN_PENALTY = -20
    
    def __init__(self):
        pass
    
    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate the current position. Returns score in centipawns.
        Positive scores favor white, negative scores favor black.
        """
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        score = 0
        
        # Material counting
        score += self._evaluate_material(board)
        
        # Positional evaluation
        score += self._evaluate_piece_mobility(board)
        score += self._evaluate_center_control(board)
        score += self._evaluate_pawn_structure(board)
        
        # Flip score if it's black's turn
        return score if board.turn else -score
    
    def _evaluate_material(self, board: chess.Board) -> int:
        """Calculate material balance"""
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = self.PIECE_VALUES[piece.piece_type]
            score += value if piece.color else -value
        return score
    
    def _evaluate_piece_mobility(self, board: chess.Board) -> int:
        """Evaluate piece mobility by counting legal moves"""
        original_turn = board.turn
        
        # Count white's moves
        board.turn = chess.WHITE
        white_moves = len(list(board.legal_moves))
        
        # Count black's moves
        board.turn = chess.BLACK
        black_moves = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = original_turn
        
        return (white_moves - black_moves) * self.MOBILITY_BONUS
    
    def _evaluate_center_control(self, board: chess.Board) -> int:
        """Evaluate center square control"""
        score = 0
        for square in self.CENTER_SQUARES:
            # White attackers of this square
            white_control = len(board.attackers(chess.WHITE, square))
            # Black attackers of this square
            black_control = len(board.attackers(chess.BLACK, square))
            score += (white_control - black_control) * self.CENTER_CONTROL_BONUS
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> int:
        """Evaluate pawn structure (currently just checking for doubled pawns)"""
        score = 0
        for file_idx in range(8):
            # Count pawns on this file
            white_pawns = 0
            black_pawns = 0
            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            
            # Penalize doubled pawns
            if white_pawns > 1:
                score += (white_pawns - 1) * self.DOUBLED_PAWN_PENALTY
            if black_pawns > 1:
                score -= (black_pawns - 1) * self.DOUBLED_PAWN_PENALTY
                
        return score

if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    evaluator = Evaluator()
    print(f"Initial position evaluation: {evaluator.evaluate(board)}")
    
    # Make a move and evaluate again
    board.push_san("e4")
    print(f"After 1.e4: {evaluator.evaluate(board)}")