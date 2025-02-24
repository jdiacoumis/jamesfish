import chess
from typing import Dict, Set, Tuple, List

class Evaluator:
    """
    Chess position evaluator that provides a static evaluation score in centipawns.
    
    This evaluator incorporates multiple evaluation components:
    - Material balance
    - Piece positioning (using piece-square tables)
    - King safety
    - Pawn structure (doubled, isolated, passed pawns)
    - Piece mobility
    - Center control
    - Development and castling
    - Game phase detection (for transitioning between middlegame and endgame)
    """
    
    # Piece values in centipawns
    PIECE_VALUES: Dict[chess.PieceType, int] = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Phase values for game stage detection
    PHASE_VALUES: Dict[chess.PieceType, int] = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
        chess.KING: 0
    }
    
    # Maximum phase value (full board)
    TOTAL_PHASE = 16 * 1 + 4 * 2 + 2 * 4  # 16 pawns + 4 rooks + 2 queens = 24
    
    # Bonuses and penalties
    CENTER_CONTROL_BONUS = 10
    EXTENDED_CENTER_CONTROL_BONUS = 5  # Bonus for controlling squares adjacent to the center
    MOBILITY_BONUS = 5
    DOUBLED_PAWN_PENALTY = -20
    ISOLATED_PAWN_PENALTY = -15
    PASSED_PAWN_BONUS = 20  # Bonus for passed pawns
    BISHOP_PAIR_BONUS = 30
    ROOK_OPEN_FILE_BONUS = 25
    ROOK_SEMI_OPEN_FILE_BONUS = 10
    ROOK_ON_SEVENTH_BONUS = 20  # Bonus for rook on 7th rank
    PAWN_CHAIN_BONUS = 10
    KNIGHT_OUTPOST_BONUS = 15  # Bonus for knight protected by pawn and not attackable by enemy pawns
    TEMPO_BONUS = 10  # Bonus for the side to move
    
    # Development bonuses/penalties
    UNDEVELOPED_PIECE_PENALTY = -10  # Penalty for undeveloped piece in opening
    CASTLING_BONUS = 40  # Bonus for having castled
    CASTLING_RIGHTS_BONUS = 10  # Bonus for each available castling right
    
    # King safety
    KING_SHIELD_BONUS = 10  # Bonus for each pawn near king
    KING_ATTACKER_PENALTY = -15  # Penalty for each enemy piece attacking king zone
    KING_OPEN_FILE_PENALTY = -25  # Penalty for king on open/semi-open file
    
    # Piece-square tables (midgame values)
    # Pawns: Encourage center control, slightly higher values for center pawns, penalize a/h pawns
    PAWN_MG_TABLE = [
        0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        15,  15,  25,  35,  35,  25,  15,  15,
        5,   5,   15,  25,  25,  15,  5,   5,
        0,   0,   5,   20,  20,  5,   0,   0,
        5,  -5,  -10,  0,   0,  -10, -5,   5,
        5,   10,  10, -20, -20,  10,  10,  5,
        0,   0,   0,   0,   0,   0,   0,   0
    ]
    
    # Knights: Emphasize center positioning, penalize edges
    KNIGHT_MG_TABLE = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,  0,   5,   5,   0,  -20, -40,
        -30,  0,   10,  15,  15,  10,  0,  -30,
        -30,  5,   15,  20,  20,  15,  5,  -30,
        -30,  0,   15,  20,  20,  15,  0,  -30,
        -30,  5,   10,  15,  15,  10,  5,  -30,
        -40, -20,  0,   5,   5,   0,  -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ]
    
    # Bishops: Encourage diagonals, center control, and fianchetto positions
    BISHOP_MG_TABLE = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,  0,   0,   0,   0,   0,   0,  -10,
        -10,  0,   10,  10,  10,  10,  0,  -10,
        -10,  5,   5,   10,  10,  5,   5,  -10,
        -10,  0,   10,  10,  10,  10,  0,  -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,  5,   0,   0,   0,   0,   5,  -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]
    
    # Rooks: Encourage 7th rank and open files, central positions in endgame
    ROOK_MG_TABLE = [
        0,   0,   0,   0,   0,   0,   0,   0,
        5,   10,  10,  10,  10,  10,  10,  5,
        -5,  0,   0,   0,   0,   0,   0,  -5,
        -5,  0,   0,   0,   0,   0,   0,  -5,
        -5,  0,   0,   0,   0,   0,   0,  -5,
        -5,  0,   0,   0,   0,   0,   0,  -5,
        -5,  0,   0,   0,   0,   0,   0,  -5,
        0,   0,   0,   5,   5,   0,   0,   0
    ]
    
    # Queens: Slight center preference, but mostly mobile
    QUEEN_MG_TABLE = [
        -20, -10, -10, -5,  -5,  -10, -10, -20,
        -10, 0,   0,   0,   0,   0,   0,   -10,
        -10, 0,   5,   5,   5,   5,   0,   -10,
        -5,  0,   5,   5,   5,   5,   0,   -5,
        0,   0,   5,   5,   5,   5,   0,   -5,
        -10, 5,   5,   5,   5,   5,   0,   -10,
        -10, 0,   5,   0,   0,   0,   0,   -10,
        -20, -10, -10, -5,  -5,  -10, -10, -20
    ]
    
    # Kings: Encourage castling in midgame, hiding behind pawns
    KING_MG_TABLE = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20,  20,  -10, -10, -10, -10, 20,  20,
        20,  30,  10,  0,   0,   10,  30,  20
    ]
    
    # Endgame piece-square tables
    # Pawns: Emphasize promotion potential
    PAWN_EG_TABLE = [
        0,   0,   0,   0,   0,   0,   0,   0,
        80,  80,  80,  80,  80,  80,  80,  80,
        60,  60,  60,  60,  60,  60,  60,  60,
        40,  40,  40,  40,  40,  40,  40,  40,
        20,  20,  20,  20,  20,  20,  20,  20,
        10,  10,  10,  10,  10,  10,  10,  10,
        10,  10,  10,  10,  10,  10,  10,  10,
        0,   0,   0,   0,   0,   0,   0,   0
    ]
    
    # Knights: Less valuable in open endgames, but still prefer center
    KNIGHT_EG_TABLE = [
        -40, -30, -20, -20, -20, -20, -30, -40,
        -30, -20, 0,   0,   0,   0,   -20, -30,
        -20, 0,   10,  10,  10,  10,  0,   -20,
        -20, 0,   10,  20,  20,  10,  0,   -20,
        -20, 0,   10,  20,  20,  10,  0,   -20,
        -20, 0,   10,  10,  10,  10,  0,   -20,
        -30, -20, 0,   0,   0,   0,   -20, -30,
        -40, -30, -20, -20, -20, -20, -30, -40
    ]
    
    # Bishops: Still valuable in endgames, good control across the board
    BISHOP_EG_TABLE = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0,   0,   0,   0,   0,   0,   -10,
        -10, 0,   10,  10,  10,  10,  0,   -10,
        -10, 0,   10,  20,  20,  10,  0,   -10,
        -10, 0,   10,  20,  20,  10,  0,   -10,
        -10, 0,   10,  10,  10,  10,  0,   -10,
        -10, 0,   0,   0,   0,   0,   0,   -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]
    
    # Rooks: Promote centralization and 7th rank in endgames
    ROOK_EG_TABLE = [
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
        15,  15,  15,  15,  15,  15,  15,  15,
        0,   0,   0,   0,   0,   0,   0,   0
    ]
    
    # Queens: More central in endgames for maximum mobility
    QUEEN_EG_TABLE = [
        -50, -30, -30, -10, -10, -30, -30, -50,
        -30, -20, -10, 0,   0,   -10, -20, -30,
        -30, -10, 20,  30,  30,  20,  -10, -30,
        -10, 0,   30,  40,  40,  30,  0,   -10,
        -10, 0,   30,  40,  40,  30,  0,   -10,
        -30, -10, 20,  30,  30,  20,  -10, -30,
        -30, -20, -10, 0,   0,   -10, -20, -30,
        -50, -30, -30, -10, -10, -30, -30, -50
    ]
    
    # Kings: Centralize in endgames, opposite of midgame
    KING_EG_TABLE = [
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0,   0,   -10, -20, -30,
        -30, -10, 20,  30,  30,  20,  -10, -30,
        -30, -10, 30,  40,  40,  30,  -10, -30,
        -30, -10, 30,  40,  40,  30,  -10, -30,
        -30, -10, 20,  30,  30,  20,  -10, -30,
        -30, -30, 0,   0,   0,   0,   -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    ]
    
    # Extended center squares - the 12 squares surrounding the 4 center squares
    EXTENDED_CENTER = {
        chess.C3, chess.D3, chess.E3, chess.F3, 
        chess.C4, chess.F4, 
        chess.C5, chess.F5, 
        chess.C6, chess.D6, chess.E6, chess.F6
    }
    
    # Main center squares
    CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
    
    def __init__(self):
        """Initialize the evaluator with pre-calculated data structures."""
        self._init_king_zone_squares()
        self._init_passed_pawn_masks()
        
    def _init_king_zone_squares(self):
        """Pre-calculate king zones for all possible king positions."""
        self.king_zones = {}
        for square in chess.SQUARES:
            zone = set()
            rank, file = chess.square_rank(square), chess.square_file(square)
            # Add squares around king
            for r in range(max(0, rank - 1), min(8, rank + 2)):
                for f in range(max(0, file - 1), min(8, file + 2)):
                    zone.add(chess.square(f, r))
            self.king_zones[square] = zone
            
    def _init_passed_pawn_masks(self):
        """Pre-calculate passed pawn masks for efficient detection."""
        # For each square, create a mask of squares that must be empty 
        # for a pawn on that square to be considered passed
        self.passed_pawn_masks = {
            chess.WHITE: {},
            chess.BLACK: {}
        }
        
        for square in chess.SQUARES:
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            
            # White passed pawn mask
            white_mask = set()
            for r in range(rank_idx + 1, 8):
                # Same file
                white_mask.add(chess.square(file_idx, r))
                # Adjacent files
                if file_idx > 0:
                    white_mask.add(chess.square(file_idx - 1, r))
                if file_idx < 7:
                    white_mask.add(chess.square(file_idx + 1, r))
            self.passed_pawn_masks[chess.WHITE][square] = white_mask
            
            # Black passed pawn mask
            black_mask = set()
            for r in range(rank_idx - 1, -1, -1):
                # Same file
                black_mask.add(chess.square(file_idx, r))
                # Adjacent files
                if file_idx > 0:
                    black_mask.add(chess.square(file_idx - 1, r))
                if file_idx < 7:
                    black_mask.add(chess.square(file_idx + 1, r))
            self.passed_pawn_masks[chess.BLACK][square] = black_mask
    
    def evaluate(self, board: chess.Board) -> int:
        """
        Evaluate the current position.
        
        Returns:
            int: Score in centipawns from the perspective of the side to move.
                 Positive values favor the side to move.
        """
        if board.is_checkmate():
            return -20000 if board.turn else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # Calculate game phase
        phase = self._calculate_game_phase(board)
        
        # Core evaluation components that apply to both midgame and endgame
        mg_score = 0
        eg_score = 0
        
        mg_score += self._evaluate_material(board)
        eg_score += self._evaluate_material(board)
        
        mg_score += self._evaluate_piece_positioning(board, phase)
        eg_score += self._evaluate_piece_positioning(board, phase)
        
        mg_score += self._evaluate_piece_mobility(board) * 1.0  # Full weight in midgame
        eg_score += self._evaluate_piece_mobility(board) * 1.2  # More important in endgame
        
        mg_score += self._evaluate_center_control(board) * 1.2  # More important in midgame
        eg_score += self._evaluate_center_control(board) * 0.8  # Less important in endgame
        
        mg_score += self._evaluate_pawn_structure(board, phase)
        eg_score += self._evaluate_pawn_structure(board, phase)
        
        mg_score += self._evaluate_king_safety(board) * 1.5  # Very important in midgame
        eg_score += self._evaluate_king_safety(board) * 0.3  # Less important in endgame
        
        mg_score += self._evaluate_bishop_pair(board) * 1.0  # Standard in midgame
        eg_score += self._evaluate_bishop_pair(board) * 1.2  # More important in endgame
        
        mg_score += self._evaluate_rook_placement(board) * 1.0
        eg_score += self._evaluate_rook_placement(board) * 1.2
        
        mg_score += self._evaluate_development(board) * 1.5  # Very important in opening/midgame
        eg_score += 0  # Not relevant in endgame
        
        # Tempo bonus
        mg_score += self.TEMPO_BONUS
        eg_score += self.TEMPO_BONUS * 2  # More important in endgame
        
        # Interpolate between midgame and endgame scores based on phase
        score = self._interpolate_score(mg_score, eg_score, phase)
        
        return score if board.turn else -score
    
    def _calculate_game_phase(self, board: chess.Board) -> float:
        """
        Calculate the game phase as a float between 0 (opening/middlegame) and 1 (endgame).
        
        Args:
            board: Current chess position
            
        Returns:
            float: Game phase between 0.0 (middlegame) and 1.0 (endgame)
        """
        phase = self.TOTAL_PHASE
        
        # Subtract the phase value for each piece still on the board
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            phase -= len(board.pieces(piece_type, chess.WHITE)) * self.PHASE_VALUES[piece_type]
            phase -= len(board.pieces(piece_type, chess.BLACK)) * self.PHASE_VALUES[piece_type]
        
        # Convert to a 0-1 scale, clamped to that range
        phase = 1.0 - (phase / self.TOTAL_PHASE)
        return max(0.0, min(1.0, phase))
    
    def _interpolate_score(self, mg_score: int, eg_score: int, phase: float) -> int:
        """
        Interpolate between middlegame and endgame scores based on game phase.
        
        Args:
            mg_score: Middlegame score
            eg_score: Endgame score
            phase: Game phase between 0.0 (middlegame) and 1.0 (endgame)
            
        Returns:
            int: Interpolated score
        """
        return int(mg_score * (1.0 - phase) + eg_score * phase)
    
    def _evaluate_material(self, board: chess.Board) -> int:
        """
        Calculate material balance.
        
        Args:
            board: Current chess position
            
        Returns:
            int: Material balance in centipawns
        """
        score = 0
        for piece_type in chess.PIECE_TYPES:
            # Skip kings when counting material
            if piece_type == chess.KING:
                continue
                
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            piece_value = self.PIECE_VALUES[piece_type]
            
            score += (white_count - black_count) * piece_value
            
        return score
    
    def _evaluate_piece_positioning(self, board: chess.Board, phase: float) -> int:
        """
        Evaluate piece positions using piece-square tables, interpolating between middlegame
        and endgame values based on game phase.
        
        Args:
            board: Current chess position
            phase: Game phase between 0.0 (middlegame) and 1.0 (endgame)
            
        Returns:
            int: Score for piece positioning
        """
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Get square index from white's perspective
            square_idx = square if piece.color else chess.square_mirror(square)
            
            piece_type = piece.piece_type
            mg_value = 0
            eg_value = 0
            
            # Get appropriate table values based on piece type
            if piece_type == chess.PAWN:
                mg_value = self.PAWN_MG_TABLE[square_idx]
                eg_value = self.PAWN_EG_TABLE[square_idx]
            elif piece_type == chess.KNIGHT:
                mg_value = self.KNIGHT_MG_TABLE[square_idx]
                eg_value = self.KNIGHT_EG_TABLE[square_idx]
            elif piece_type == chess.BISHOP:
                mg_value = self.BISHOP_MG_TABLE[square_idx]
                eg_value = self.BISHOP_EG_TABLE[square_idx]
            elif piece_type == chess.ROOK:
                mg_value = self.ROOK_MG_TABLE[square_idx]
                eg_value = self.ROOK_EG_TABLE[square_idx]
            elif piece_type == chess.QUEEN:
                mg_value = self.QUEEN_MG_TABLE[square_idx]
                eg_value = self.QUEEN_EG_TABLE[square_idx]
            elif piece_type == chess.KING:
                mg_value = self.KING_MG_TABLE[square_idx]
                eg_value = self.KING_EG_TABLE[square_idx]
            
            # Interpolate between midgame and endgame values
            value = int(mg_value * (1.0 - phase) + eg_value * phase)
            score += value if piece.color else -value
            
        return score
    
    def _evaluate_king_safety(self, board: chess.Board) -> int:
        """
        Evaluate king safety based on pawn shield, open files, and attackers.
        
        Args:
            board: Current chess position
            
        Returns:
            int: Score for king safety
        """
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
            
            # Count attackers targeting squares near the king
            attackers = 0
            attacker_weight = 0  # Weight by piece type - attackers with higher values are more dangerous
            for square in king_zone:
                for attacker_square in board.attackers(not color, square):
                    attacker_piece = board.piece_at(attacker_square)
                    if attacker_piece:
                        attackers += 1
                        attacker_weight += self.PIECE_VALUES[attacker_piece.piece_type] // 100
            
            # Check if king is on an open or semi-open file
            king_file = chess.square_file(king_square)
            king_file_open = True
            king_file_semi_open = True
            
            for rank in range(8):
                check_square = chess.square(king_file, rank)
                piece = board.piece_at(check_square)
                
                if piece and piece.piece_type == chess.PAWN:
                    king_file_open = False
                    if piece.color == color:
                        king_file_semi_open = False
            
            # Calculate safety score
            safety = (friendly_pawns * self.KING_SHIELD_BONUS + 
                      attackers * self.KING_ATTACKER_PENALTY * (1 + attacker_weight / 10))
                      
            # Add penalty for open files near the king
            if king_file_open:
                safety += self.KING_OPEN_FILE_PENALTY
            elif king_file_semi_open:
                safety += self.KING_OPEN_FILE_PENALTY // 2
            
            score += safety if color else -safety
            
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board, phase: float) -> int:
        """
        Evaluate pawn structure including chains, doubled, isolated, and passed pawns.
        
        Args:
            board: Current chess position
            phase: Game phase between 0.0 (middlegame) and 1.0 (endgame)
            
        Returns:
            int: Score for pawn structure
        """
        score = 0
        pawn_files = {chess.WHITE: [0] * 8, chess.BLACK: [0] * 8}
        pawn_ranks = {chess.WHITE: [0] * 8, chess.BLACK: [0] * 8}
        
        # First pass: record pawn files and check for doubles
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                
                pawn_files[piece.color][file_idx] += 1
                pawn_ranks[piece.color][file_idx] = max(pawn_ranks[piece.color][file_idx], 
                                                      rank_idx if piece.color else 7 - rank_idx)
                
                # Handle doubled pawns
                if pawn_files[piece.color][file_idx] > 1:
                    doubled_penalty = self.DOUBLED_PAWN_PENALTY
                    # Doubled pawns are worse in the endgame
                    if phase > 0.5:
                        doubled_penalty = int(doubled_penalty * 1.5)
                    score += doubled_penalty if piece.color else -doubled_penalty
                
                # Check for passed pawns
                is_passed = True
                for enemy_square in self.passed_pawn_masks[piece.color][square]:
                    enemy_piece = board.piece_at(enemy_square)
                    if enemy_piece and enemy_piece.piece_type == chess.PAWN and enemy_piece.color != piece.color:
                        is_passed = False
                        break
                
                if is_passed:
                    # Calculate bonus based on rank and game phase
                    passed_bonus = self.PASSED_PAWN_BONUS
                    
                    # Passed pawns are more valuable in endgame and when further advanced
                    rank_factor = rank_idx if piece.color else 7 - rank_idx
                    passed_bonus += passed_bonus * rank_factor // 6
                    
                    # Increase bonus in endgame
                    passed_bonus = int(passed_bonus * (1.0 + phase))
                    
                    score += passed_bonus if piece.color else -passed_bonus
        
        # Check for isolated pawns
        for color in [chess.WHITE, chess.BLACK]:
            for file_idx in range(8):
                if pawn_files[color][file_idx] > 0:
                    # Check if isolated
                    is_isolated = True
                    if file_idx > 0 and pawn_files[color][file_idx - 1] > 0:
                        is_isolated = False
                    if file_idx < 7 and pawn_files[color][file_idx + 1] > 0:
                        is_isolated = False
                    
                    if is_isolated:
                        isolated_penalty = self.ISOLATED_PAWN_PENALTY
                        # Isolated pawns are worse in the endgame
                        if phase > 0.5:
                            isolated_penalty = int(isolated_penalty * 1.5)
                        score += isolated_penalty if color else -isolated_penalty
                    
                    # Pawn chains check
                    if file_idx < 7 and pawn_files[color][file_idx + 1] > 0:
                        score += self.PAWN_CHAIN_BONUS if color else -self.PAWN_CHAIN_BONUS
        
        return score
    
    def _evaluate_bishop_pair(self, board: chess.Board) -> int:
        """
        Award bonus for having both bishops.
        
        Args:
            board: Current chess position
            
        Returns:
            int: Score for bishop pair bonus
        """
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            bishop_count = len(board.pieces(chess.BISHOP, color))
            if bishop_count >= 2:
                score += self.BISHOP_PAIR_BONUS if color else -self.BISHOP_PAIR_BONUS
        return score
    
    def _evaluate_rook_placement(self, board: chess.Board) -> int:
        """
        Evaluate rook placement, especially on open files and 7th rank.
        
        Args:
            board: Current chess position
            
        Returns:
            int: Score for rook placement
        """
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.ROOK:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                
                # Check if rook is on 7th rank (relative to color)
                if (piece.color == chess.WHITE and rank_idx == 6) or \
                   (piece.color == chess.BLACK and rank_idx == 1):
                    score += self.ROOK_ON_SEVENTH_BONUS if piece.color else -self.ROOK_ON_SEVENTH_BONUS
                
                # Check if file is completely open
                file_is_open = True
                for r in range(8):
                    check_square = chess.square(file_idx, r)
                    check_piece = board.piece_at(check_square)
                    if check_piece is not None and check_piece.piece_type == chess.PAWN:
                        file_is_open = False
                        break
                
                if file_is_open:
                    score += self.ROOK_OPEN_FILE_BONUS if piece.color else -self.ROOK_OPEN_FILE_BONUS
                else:
                    # Check if file is semi-open (no friendly pawns)
                    file_is_semi_open = True
                    for r in range(8):
                        check_square = chess.square(file_idx, r)
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
        """
        Evaluate piece mobility (number of legal moves per piece).
        
        Args:
            board: Current chess position
            
        Returns:
            int: Score for piece mobility
        """
        original_turn = board.turn
        
        # Save the current state of the castling rights
        white_kingside = board.has_kingside_castling_rights(chess.WHITE)
        white_queenside = board.has_queenside_castling_rights(chess.WHITE)
        black_kingside = board.has_kingside_castling_rights(chess.BLACK)
        black_queenside = board.has_queenside_castling_rights(chess.BLACK)
        
        # Create a temporary board copy for move generation
        board_copy = board.copy()
        
        # Calculate white's moves
        board_copy.turn = chess.WHITE
        white_moves = list(board_copy.legal_moves)
        
        # Calculate black's moves
        board_copy.turn = chess.BLACK
        black_moves = list(board_copy.legal_moves)
        
        # Restore original turn
        board.turn = original_turn
        
        # Count the difference in moves
        move_diff = len(white_moves) - len(black_moves)
        
        return move_diff * self.MOBILITY_BONUS
    
    def _evaluate_center_control(self, board: chess.Board) -> int:
        """
        Evaluate control of the center squares and extended center.
        
        Args:
            board: Current chess position
            
        Returns:
            int: Score for center control
        """
        score = 0
        
        # Evaluate control of main center squares
        for square in self.CENTER_SQUARES:
            white_control = len(board.attackers(chess.WHITE, square))
            black_control = len(board.attackers(chess.BLACK, square))
            score += (white_control - black_control) * self.CENTER_CONTROL_BONUS
        
        # Evaluate control of extended center squares
        for square in self.EXTENDED_CENTER:
            white_control = len(board.attackers(chess.WHITE, square))
            black_control = len(board.attackers(chess.BLACK, square))
            score += (white_control - black_control) * self.EXTENDED_CENTER_CONTROL_BONUS
        
        return score
    
    def _evaluate_development(self, board: chess.Board) -> int:
        """
        Evaluate piece development and castling status.
        Only relevant in opening/early middlegame.
        
        Args:
            board: Current chess position
            
        Returns:
            int: Score for development
        """
        # Don't evaluate development if too many pieces are off the board
        total_pieces = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            total_pieces += len(board.pieces(piece_type, chess.WHITE))
            total_pieces += len(board.pieces(piece_type, chess.BLACK))
        
        if total_pieces < 10:  # Arbitrary threshold
            return 0
        
        score = 0
        
        # Check for undeveloped pieces in the opening
        for color in [chess.WHITE, chess.BLACK]:
            # Knights and bishops on their original squares
            home_rank = 0 if color == chess.WHITE else 7
            
            # Check knights
            for file in [1, 6]:  # b and g files
                square = chess.square(file, home_rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.KNIGHT and piece.color == color:
                    score += self.UNDEVELOPED_PIECE_PENALTY if color else -self.UNDEVELOPED_PIECE_PENALTY
            
            # Check bishops
            for file in [2, 5]:  # c and f files
                square = chess.square(file, home_rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.BISHOP and piece.color == color:
                    score += self.UNDEVELOPED_PIECE_PENALTY if color else -self.UNDEVELOPED_PIECE_PENALTY
        
        # Check castling status
        for color in [chess.WHITE, chess.BLACK]:
            king_file = chess.square_file(board.king(color))
            
            # Check if already castled
            if color == chess.WHITE and king_file in [2, 6]:  # O-O-O or O-O
                score += self.CASTLING_BONUS
            elif color == chess.BLACK and king_file in [2, 6]:
                score -= self.CASTLING_BONUS
                
            # Award bonus for each available castling right
            if board.has_kingside_castling_rights(color):
                score += self.CASTLING_RIGHTS_BONUS if color else -self.CASTLING_RIGHTS_BONUS
            if board.has_queenside_castling_rights(color):
                score += self.CASTLING_RIGHTS_BONUS if color else -self.CASTLING_RIGHTS_BONUS
        
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
    
    # Test a common opening position
    board = chess.Board()
    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"]
    for move in moves:
        board.push_san(move)
    print(f"After 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6: {evaluator.evaluate(board)}")
    
    # Test a middlegame position
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5")
    print(f"Middlegame position: {evaluator.evaluate(board)}")
    
    # Test an endgame position
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    print(f"Endgame position: {evaluator.evaluate(board)}")
