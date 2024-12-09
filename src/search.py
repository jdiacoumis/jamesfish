import chess
from typing import Optional, Tuple, Dict, List
from .evaluation import Evaluator
import time

class SearchEngine:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.nodes_searched = 0
        self.start_time = 0
        self.max_time = min(30, 5 * (depth - 3))
        self.best_move_found = None
        self.transposition_table = {}
        
        # Move ordering tables
        self.history_table: Dict[tuple, int] = {}  # (from_square, to_square): score
        self.killer_moves: List[Optional[chess.Move]] = [None] * 128  # One killer move per ply
        self.pv_line: List[Optional[chess.Move]] = [None] * 64  # Store principal variation
        
        # Search & pruning parameters
        self.NULL_MOVE_R = 2 + (depth >= 6)  # More reduction at deeper depths
        self.ASPIRATION_WINDOW = 25
        self.FUTILITY_MARGIN = 50
        self.DELTA_MARGIN = 150  # For delta pruning in quiescence
        self.FULL_DEPTH_MOVES = 3
        self.REDUCTION_LIMIT = 3
        
    def clear_tables(self):
        """Clear move ordering tables between searches"""
        self.history_table.clear()
        self.killer_moves = [None] * 128
        self.pv_line = [None] * 64

    def should_stop_search(self) -> bool:
        """Check if we should stop searching based on time"""
        return time.time() - self.start_time > self.max_time

    def search(self, board: chess.Board, max_depth: int = 4, max_time: int = 30) -> Tuple[chess.Move, int]:
        """Main search function with aspiration windows"""
        self.nodes_searched = 0
        self.start_time = time.time()
        self.max_time = max_time
        self.best_move_found = None
        self.clear_tables()
        
        previous_score = 0
        for depth in range(1, max_depth + 1):
            try:
                # Use aspiration windows after depth 3
                if depth >= 4:
                    alpha = previous_score - self.ASPIRATION_WINDOW
                    beta = previous_score + self.ASPIRATION_WINDOW
                    score = self._search_with_aspiration(board, depth, alpha, beta)
                else:
                    # Full window for early depths
                    score = self.negamax(board, depth, -30000, 30000, 0, True)
                    
                if self.should_stop_search():
                    break
                    
                previous_score = score
                print(f"Depth {depth} completed. Best move: {self.best_move_found} Score: {score}")
            except TimeoutError:
                break
            
        if self.best_move_found is None:
            self.best_move_found = list(board.legal_moves)[0]
            
        return self.best_move_found, previous_score

    def _search_with_aspiration(self, board: chess.Board, depth: int, alpha: int, beta: int) -> int:
        """Search with aspiration windows, widening if score falls outside window"""
        while True:
            score = self.negamax(board, depth, alpha, beta, 0, True)
            
            # Score within window, return it
            if alpha < score < beta:
                return score
                
            # Score outside window, widen and retry
            if score <= alpha:
                alpha = max(-30000, alpha - self.ASPIRATION_WINDOW)
            else:
                beta = min(30000, beta + self.ASPIRATION_WINDOW)

    def static_null_move_pruning(self, board: chess.Board, beta: float, depth: int) -> bool:
        """Static null move pruning (reverse futility pruning)"""
        if depth < 3 or board.is_check():
            return False
        margin = 120 * depth
        if self.evaluator.evaluate(board) - margin >= beta:
            return True
        return False

    def reverse_futility_pruning(self, board: chess.Board, alpha: float, depth: int) -> bool:
        """Reverse futility pruning"""
        if depth < 3 or board.is_check():
            return False
        margin = 120 * depth
        if self.evaluator.evaluate(board) + margin <= alpha:
            return True
        return False

    def get_move_score(self, board: chess.Board, move: chess.Move, ply: int) -> int:
        """Fast move scoring for move ordering"""
        # PV move (highest priority)
        if ply == 0 and self.pv_line[0] == move:
            return 30000
        
        # Hash move from transposition table
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            stored_depth, _, stored_move = self.transposition_table[board_hash]
            if stored_move == move:
                return 25000
        
        # Captures
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                return 20000 + (victim.piece_type * 6 - attacker.piece_type)
            return 20000
            
        # Killer move
        if self.killer_moves[ply] == move:
            return 10000
            
        # History score
        return self.history_table.get((move.from_square, move.to_square), 0)

    def has_non_pawn_material(self, board: chess.Board, color: bool) -> bool:
        """Check if side has any pieces besides pawns and king"""
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            if any(board.pieces(piece_type, color)):
                return True
        return False

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int, is_pv: bool = False) -> float:
        """Negamax algorithm with PVS and all pruning techniques"""
        self.nodes_searched += 1
        
        if self.should_stop_search():
            raise TimeoutError
            
        # Quick checks for draws
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0
            
        alpha_original = alpha
        in_check = board.is_check()
        
        # Base cases
        if depth <= 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)
        
        # Retrieve from transposition table
        board_hash = board.fen()
        if not is_pv and board_hash in self.transposition_table:
            stored_depth, stored_value, _ = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_value
                
        # Static evaluation for pruning
        static_eval = self.evaluator.evaluate(board)
        
        # Initialize do_futility before the if block
        do_futility = False

        # Various pruning techniques
        if not is_pv:
            # Static null move pruning
            if self.static_null_move_pruning(board, beta, depth):
                return beta
                
            # Reverse futility pruning
            if self.reverse_futility_pruning(board, alpha, depth):
                return alpha
                
            # Null move pruning
            if (depth >= 3 and not in_check and 
                self.has_non_pawn_material(board, board.turn)):
                board.push(chess.Move.null())
                value = -self.negamax(board, depth - 1 - self.NULL_MOVE_R, -beta, -beta + 1, ply + 1)
                board.pop()
                
                if value >= beta:
                    return beta
            
            # Futility pruning preparation
            futility_base = static_eval + self.FUTILITY_MARGIN
            do_futility = (
                depth <= 3 and 
                not in_check and 
                abs(beta) < 9000  # Not near mate scores
            )
                
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self.get_move_score(board, m, ply), reverse=True)
        
        best_move = None
        best_value = float('-inf')
        moves_searched = 0
        
        # Principal Variation Search
        for move in moves:
            moves_searched += 1
            
            # Futility pruning
            if (do_futility and moves_searched > 1 and 
                not board.is_capture(move) and not board.gives_check(move)):
                if futility_base <= alpha:
                    continue
            
            board.push(move)
            
            # PVS logic
            if moves_searched == 1:
                value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, is_pv)
            else:
                # Late move reductions
                if (depth >= self.REDUCTION_LIMIT and 
                    moves_searched > self.FULL_DEPTH_MOVES and 
                    not in_check and 
                    not board.is_capture(move)):
                    reduction = 1 + (moves_searched > 6) + (depth >= 6)  # More aggressive at depth 6+
                    value = -self.negamax(board, depth - 1 - reduction, -(alpha + 1), -alpha, ply + 1, False)
                else:
                    value = alpha + 1
                
                # PVS re-search if promising
                if value > alpha:
                    value = -self.negamax(board, depth - 1, -(alpha + 1), -alpha, ply + 1, False)
                    if value > alpha and value < beta:
                        value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, False)
                
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
                
                if value > alpha:
                    alpha = value
                    if ply == 0:
                        self.best_move_found = move
                        self.pv_line[0] = move
                    
            if alpha >= beta:
                # Update killer moves and history for quiet moves
                if not board.is_capture(move):
                    self.killer_moves[ply] = move
                    self.history_table[(move.from_square, move.to_square)] = (
                        self.history_table.get((move.from_square, move.to_square), 0) + depth * depth
                    )
                break
        
        # Store in transposition table
        self.transposition_table[board_hash] = (depth, best_value, best_move)
        
        return best_value

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Quiescence search with delta pruning"""
        self.nodes_searched += 1
        
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
            
        # Delta pruning
        if stand_pat < alpha - self.DELTA_MARGIN:
            return alpha
            
        alpha = max(alpha, stand_pat)
        
        # Only look at captures and checks
        captures = [move for move in board.legal_moves if board.is_capture(move) or board.gives_check(move)]
        captures.sort(key=lambda m: self.get_move_score(board, m, 0), reverse=True)
        
        for move in captures:
            # Delta pruning at move level
            if not board.gives_check(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    if stand_pat + self.evaluator.PIECE_VALUES[victim.piece_type] + 200 < alpha:
                        continue
            
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
                
        return alpha