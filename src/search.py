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
        
        # More aggressive pruning parameters
        self.NULL_MOVE_R = 3  # Increased from 2
        self.FUTILITY_MARGIN = 80  # Decreased from 100
        self.LMR_DEPTH = 2  # Decreased from 3
        self.LMR_MOVES = 3  # Decreased from 4
        
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

    def static_null_move_pruning(self, board: chess.Board, beta: float, depth: int) -> bool:
        """Static null move pruning (also known as reverse futility pruning)"""
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

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_root: bool = False) -> float:
        """Negamax algorithm with enhanced pruning techniques"""
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
        
        # Static evaluation for pruning
        static_eval = self.evaluator.evaluate(board)
        
        # Base cases
        if depth <= 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)
            
        # Static null move pruning
        if self.static_null_move_pruning(board, beta, depth):
            return beta
            
        # Reverse futility pruning
        if self.reverse_futility_pruning(board, alpha, depth):
            return alpha
            
        # Null move pruning
        if depth >= 3 and not in_check and self.can_do_null_move(board):
            board.push(chess.Move.null())
            null_value = -self.negamax(board, depth - 1 - self.NULL_MOVE_R, -beta, -beta + 1)
            board.pop()
            
            if null_value >= beta:
                return beta
                
        # Futility pruning preparation
        futility_base = static_eval + self.FUTILITY_MARGIN
        do_futility = (
            depth <= 2 and 
            not in_check and 
            abs(beta) < 9000  # Not near mate scores
        )
        
        if do_futility and futility_base <= alpha:
            return futility_base
        
        # Move ordering
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self._mvv_lva_score(board, move), reverse=True)
        
        best_value = float('-inf')
        best_move = None
        moves_searched = 0
        
        for move in moves:
            moves_searched += 1
            
            # Extended futility pruning
            if (do_futility and moves_searched > 1 and 
                not board.is_capture(move) and 
                not board.gives_check(move)):
                if futility_base <= alpha:
                    continue
                    
            # Late Move Reduction (LMR)
            if (depth >= self.LMR_DEPTH and
                moves_searched >= self.LMR_MOVES and
                not in_check and
                not board.is_capture(move)):
                
                # More aggressive reduction for later moves
                reduction = 1 + (moves_searched > 6)
                
                board.push(move)
                value = -self.negamax(board, depth - 1 - reduction, -beta, -alpha)
                
                # Re-search if it might raise alpha
                if value > alpha:
                    value = -self.negamax(board, depth - 1, -beta, -alpha)
                board.pop()
            else:
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
        """More aggressive null move conditions"""
        if board.is_check():
            return False
            
        # Require fewer major pieces for null move
        major_pieces = 0
        for piece_type in [chess.QUEEN, chess.ROOK]:  # Only count queens and rooks
            major_pieces += len(board.pieces(piece_type, board.turn))
        return major_pieces > 0

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 4) -> float:
        """Enhanced quiescence search with more aggressive delta pruning"""
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        
        # More aggressive delta pruning
        DELTA_MARGIN = 150  # Reduced from 200
        if stand_pat < alpha - DELTA_MARGIN:
            return alpha
            
        alpha = max(alpha, stand_pat)
        
        for move in board.legal_moves:
            if not board.is_capture(move) and not board.gives_check(move):
                continue
                
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
        """More aggressive capture pruning"""
        victim_value = 0
        victim_piece = board.piece_at(move.to_square)
        if victim_piece:
            victim_value = self.evaluator.PIECE_VALUES[victim_piece.piece_type]
            
        if self.evaluator.evaluate(board) + victim_value + 150 < alpha:  # Reduced margin
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