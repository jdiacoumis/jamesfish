import chess
from typing import Dict, Set

class Evaluator:
    # Piece values in centipawns
    PIECE_VALUES: Dict[chess.PieceType, int] = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Bonuses and penalties
    CENTER_CONTROL_BONUS = 10
    MOBILITY_BONUS = 5
    DOUBLED_PAWN_PENALTY = -20
    ISOLATED_PAWN_PENALTY = -15
    BISHOP_PAIR_BONUS = 30
    ROOK_OPEN_FILE_BONUS = 25
    ROOK_SEMI_OPEN_FILE_BONUS = 10
    PAWN_CHAIN_BONUS = 10
    
    # King safety
    KING_SHIELD_BONUS = 10  # Bonus for each pawn near king
    KING_ATTACKER_PENALTY = -15  # Penalty for each enemy piece attacking king zone
    
    # Piece-square tables (midgame values)
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
    
    BISHOP_TABLE = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ]
    
    def __init__(self):
        self._init_king_zone_squares()
    
    def _init_king_zone_squares(self):
        """Pre-calculate king zones for all possible king positions"""
        self.king_zones = {}
        for square in chess.SQUARES:
            zone = set()
            rank, file = chess.square_rank(square), chess.square_file(square)
            # Add squares around king
            for r in range(max(0, rank - 1), min(8, rank + 2)):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    zone.add(chess.square(f, r))
            self.king_zones[square] = zone
    
    def evaluate(self, board: chess.Board) -> int:
        """Evaluate the current position"""
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        score = 0
        
        # Core evaluation components
        score += self._evaluate_material(board)
        score += self._evaluate_piece_mobility(board)
        score += self._evaluate_center_control(board)
        score += self._evaluate_pawn_structure(board)
        
        # New evaluation components
        score += self._evaluate_king_safety(board)
        score += self._evaluate_piece_positioning(board)
        score += self._evaluate_bishop_pair(board)
        score += self._evaluate_rook_placement(board)
        
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
    
    def _evaluate_piece_positioning(self, board: chess.Board) -> int:
        """Evaluate piece positions using piece-square tables"""
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Get square index from white's perspective
            square_idx = square if piece.color else chess.square_mirror(square)
            
            if piece.piece_type == chess.PAWN:
                value = self.PAWN_TABLE[square_idx]
            elif piece.piece_type == chess.KNIGHT:
                value = self.KNIGHT_TABLE[square_idx]
            elif piece.piece_type == chess.BISHOP:
                value = self.BISHOP_TABLE[square_idx]
            else:
                continue  # Skip other pieces for now
                
            score += value if piece.color else -value
            
        return score
    
    def _evaluate_king_safety(self, board: chess.Board) -> int:
        """Evaluate king safety based on pawn shield and attacking pieces"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
                
            # Evaluate pawn shield
            king_zone = self.king_zones[king_square]
            friendly_pawns = 0
            for square in king_zone:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    friendly_pawns += 1
            
            # Count attackers
            attackers = 0
            for square in king_zone:
                attackers += len(board.attackers(not color, square))
            
            zone_safety = (friendly_pawns * self.KING_SHIELD_BONUS + 
                         attackers * self.KING_ATTACKER_PENALTY)
            score += zone_safety if color else -zone_safety
            
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> int:
        """Evaluate pawn structure including chains, doubled and isolated pawns"""
        score = 0
        pawn_files = {chess.WHITE: set(), chess.BLACK: set()}
        
        # First pass: record pawn files and check for doubles
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN:
                file_idx = chess.square_file(square)
                pawn_files[piece.color].add(file_idx)
                
                # Count pawns on this file
                file_pawns = sum(1 for rank in range(8)
                               if board.piece_at(chess.square(file_idx, rank)) is not None
                               and board.piece_at(chess.square(file_idx, rank)).piece_type == chess.PAWN
                               and board.piece_at(chess.square(file_idx, rank)).color == piece.color)
                
                if file_pawns > 1:
                    score += (file_pawns - 1) * self.DOUBLED_PAWN_PENALTY if piece.color else -(file_pawns - 1) * self.DOUBLED_PAWN_PENALTY
        
        # Check for isolated pawns and pawn chains
        for color in [chess.WHITE, chess.BLACK]:
            for file_idx in pawn_files[color]:
                # Isolated pawns check
                if (file_idx - 1 not in pawn_files[color] and 
                    file_idx + 1 not in pawn_files[color]):
                    score += self.ISOLATED_PAWN_PENALTY if color else -self.ISOLATED_PAWN_PENALTY
                
                # Pawn chains check
                if file_idx + 1 in pawn_files[color]:
                    score += self.PAWN_CHAIN_BONUS if color else -self.PAWN_CHAIN_BONUS
        
        return score
    
    def _evaluate_bishop_pair(self, board: chess.Board) -> int:
        """Award bonus for having both bishops"""
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            bishop_count = len(board.pieces(chess.BISHOP, color))
            if bishop_count >= 2:
                score += self.BISHOP_PAIR_BONUS if color else -self.BISHOP_PAIR_BONUS
        return score
    
    def _evaluate_rook_placement(self, board: chess.Board) -> int:
        """Evaluate rook placement, especially on open files"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.ROOK:
                file_idx = chess.square_file(square)
                
                # Check if file is completely open
                file_is_open = True
                for rank in range(8):
                    check_square = chess.square(file_idx, rank)
                    check_piece = board.piece_at(check_square)
                    if check_piece is not None and check_piece.piece_type == chess.PAWN:
                        file_is_open = False
                        break
                
                if file_is_open:
                    score += self.ROOK_OPEN_FILE_BONUS if piece.color else -self.ROOK_OPEN_FILE_BONUS
                else:
                    # Check if file is semi-open (no friendly pawns)
                    file_is_semi_open = True
                    for rank in range(8):
                        check_square = chess.square(file_idx, rank)
                        check_piece = board.piece_at(check_square)
                        if (check_piece is not None and 
                            check_piece.piece_type == chess.PAWN and 
                            check_piece.color == piece.color):
                            file_is_semi_open = False
                            break
                    
                    if file_is_semi_open:
                        score += self.ROOK_SEMI_OPEN_FILE_BONUS if piece.color else -self.ROOK_SEMI_OPEN_FILE_BONUS
        
        return score
    
    def _evaluate_piece_mobility(self, board: chess.Board) -> int:
        """Evaluate piece mobility"""
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
        center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
        score = 0
        for square in center_squares:
            white_control = len(board.attackers(chess.WHITE, square))
            black_control = len(board.attackers(chess.BLACK, square))
            score += (white_control - black_control) * self.CENTER_CONTROL_BONUS
        return score

if __name__ == "__main__":
    # Example usage and testing
    board = chess.Board()
    evaluator = Evaluator()
    
    # Test initial position
    print(f"Initial position evaluation: {evaluator.evaluate(board)}")
    
    # Test after e4
    board.push_san("e4")
    print(f"After 1.e4: {evaluator.evaluate(board)}")
    
    # Test a position with a bishop pair
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"Position with bishop pair: {evaluator.evaluate(board)}")