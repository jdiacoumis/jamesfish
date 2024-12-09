import chess
from typing import Optional, Tuple
from .evaluation import Evaluator
import time

class SearchEngine:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.nodes_searched = 0
        self.start_time = 0
        self.max_time = 0
        self.best_move_found = None
        self.transposition_table = {}
        
        # Pruning parameters
        self.NULL_MOVE_R = 2  # Reduction factor for null move pruning
        self.FUTILITY_MARGIN = 100  # Margin in centipawns
        self.LMR_DEPTH = 3  # Minimum depth for late move reduction
        self.LMR_MOVES = 4  # Number of moves before late move reduction
        
    def should_stop_search(self) -> bool:
        return time.time() - self.start_time > self.max_time

    def search(self, board: chess.Board, max_depth: int = 4, max_time: int = 30) -> Tuple[chess.Move, int]:
        self.nodes_searched = 0
        self.start_time = time.time()
        self.max_time = max_time
        self.best_move_found = None
        
        alpha = float('-inf')
        beta = float('inf')
        
        for current_depth in range(1, max_depth + 1):
            try:
                score = self.negamax(board, current_depth, alpha, beta, True)
                print(f"Depth {current_depth} completed. Best move: {self.best_move_found} Score: {score}")
            except TimeoutError:
                break
                
        if self.best_move_found is None:
            self.best_move_found = list(board.legal_moves)[0]
            
        return self.best_move_found, self.evaluator.evaluate(board)

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_root: bool = False) -> float:
        """Negamax algorithm with various pruning techniques"""
        self.nodes_searched += 1
        
        if self.should_stop_search():
            raise TimeoutError
            
        # Check transposition table
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            stored_depth, stored_value = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_value
                
        original_alpha = alpha
        in_check = board.is_check()
        
        # Base cases
        if depth <= 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)
            
        # Null move pruning
        # Skip if in check or at low depth to avoid horizon effect
        if depth >= 3 and not in_check and self.can_do_null_move(board):
            board.push(chess.Move.null())
            null_value = -self.negamax(board, depth - 1 - self.NULL_MOVE_R, -beta, -beta + 1)
            board.pop()
            
            if null_value >= beta:
                return beta
                
        # Futility pruning preparation
        futility_margin = self.FUTILITY_MARGIN * depth
        do_futility = (
            depth <= 2 and 
            not in_check and 
            abs(beta) < 9000  # Not near mate scores
        )
        
        if do_futility:
            static_eval = self.evaluator.evaluate(board)
            if static_eval - futility_margin >= beta:
                return static_eval
        
        # Move ordering
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self._mvv_lva_score(board, move), reverse=True)
        
        best_value = float('-inf')
        best_move = None
        moves_searched = 0
        
        for move in moves:
            moves_searched += 1
            
            # Late Move Reduction (LMR)
            # Reduce search depth for later moves that are unlikely to be good
            if (depth >= self.LMR_DEPTH and
                moves_searched >= self.LMR_MOVES and
                not in_check and
                not board.is_capture(move)):
                reduction = 1
                
                # Do reduced depth search
                board.push(move)
                value = -self.negamax(board, depth - 1 - reduction, -beta, -alpha)
                
                # If the reduced search beats alpha, do a full-depth search
                if value > alpha:
                    value = -self.negamax(board, depth - 1, -beta, -alpha)
                board.pop()
            else:
                # Normal search for early moves, captures, and checks
                board.push(move)
                value = -self.negamax(board, depth - 1, -beta, -alpha)
                board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
                if is_root:
                    self.best_move_found = move
                    
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
                
        # Store in transposition table
        self.transposition_table[board_hash] = (depth, best_value)
        
        return best_value

    def can_do_null_move(self, board: chess.Board) -> bool:
        """Determine if we can do a null move in this position"""
        # Don't do null move if we don't have major pieces
        # This prevents horizon effect in endgames
        major_pieces = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            major_pieces += len(board.pieces(piece_type, board.turn))
        return major_pieces > 0

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 4) -> float:
        """Quiescence search with delta pruning"""
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        
        # Delta pruning
        DELTA_MARGIN = 200  # Big enough to catch most tactical sequences
        if stand_pat < alpha - DELTA_MARGIN:
            return alpha
            
        alpha = max(alpha, stand_pat)
        
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
                
            # Delta pruning at move level
            if not self.is_capture_worth_searching(board, move, alpha):
                continue
                
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
                
        return alpha

    def is_capture_worth_searching(self, board: chess.Board, move: chess.Move, alpha: float) -> bool:
        """Determine if a capture is worth searching in quiescence search"""
        victim_value = 0
        victim_piece = board.piece_at(move.to_square)
        if victim_piece:
            victim_value = self.evaluator.PIECE_VALUES[victim_piece.piece_type]
            
        # If capturing a piece doesn't bring us close to alpha, skip it
        if self.evaluator.evaluate(board) + victim_value + 200 < alpha:
            return False
            
        return True

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        if not board.is_capture(move):
            return 0
            
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        
        if attacker is None or victim is None:
            return 0
            
        attacker_value = self.evaluator.PIECE_VALUES[attacker.piece_type]
        victim_value = self.evaluator.PIECE_VALUES[victim.piece_type]
        
        return victim_value - (attacker_value / 100)