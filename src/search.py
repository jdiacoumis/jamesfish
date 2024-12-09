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
        self.transposition_table = {}  # Simple transposition table

    def should_stop_search(self) -> bool:
        """Check if we should stop searching based on time"""
        return time.time() - self.start_time > self.max_time

    def search(self, board: chess.Board, max_depth: int = 4, max_time: int = 30) -> Tuple[chess.Move, int]:
        """
        Search for the best move using negamax with alpha-beta pruning.
        
        Args:
            board: Current position
            max_depth: Maximum search depth
            max_time: Maximum search time in seconds
            
        Returns:
            Tuple of (best_move, evaluation)
        """
        self.nodes_searched = 0
        self.start_time = time.time()
        self.max_time = max_time
        self.best_move_found = None
        
        alpha = float('-inf')
        beta = float('inf')
        
        # Iterative deepening
        for current_depth in range(1, max_depth + 1):
            try:
                score = self.negamax(board, current_depth, alpha, beta, True)
                print(f"Depth {current_depth} completed. Best move: {self.best_move_found} Score: {score}")
            except TimeoutError:
                break
                
        if self.best_move_found is None:
            # If we haven't found any move (shouldn't happen), pick the first legal move
            self.best_move_found = list(board.legal_moves)[0]
            
        return self.best_move_found, self.evaluator.evaluate(board)

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, is_root: bool = False) -> float:
        """
        Negamax algorithm with alpha-beta pruning.
        
        Args:
            board: Current position
            depth: Remaining depth to search
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_root: Whether this is the root node
            
        Returns:
            Position evaluation score
        """
        self.nodes_searched += 1
        
        if self.should_stop_search():
            raise TimeoutError
            
        # Check transposition table
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            stored_depth, stored_value = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_value

        # Base case: if we've reached maximum depth, do quiescence search
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)
            
        if board.is_game_over():
            if board.is_checkmate():
                return float('-inf')  # Lost for side to move
            return 0.0  # Draw
            
        best_value = float('-inf')
        best_move = None
        
        # Move ordering: try captures first
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self._mvv_lva_score(board, move), reverse=True)
        
        for move in moves:
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

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float, depth: int = 4) -> float:
        """
        Quiescence search to handle tactical positions.
        Only looks at captures to reach a "quiet" position.
        """
        if depth == 0:
            return self.evaluator.evaluate(board)
            
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        # Look at capture moves only
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
                
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """
        Most Valuable Victim - Least Valuable Attacker (MVV-LVA) score for move ordering.
        Higher scores for capturing valuable pieces with less valuable pieces.
        """
        if not board.is_capture(move):
            return 0
            
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        
        if attacker is None or victim is None:
            return 0
            
        # Use piece values from evaluator
        attacker_value = self.evaluator.PIECE_VALUES[attacker.piece_type]
        victim_value = self.evaluator.PIECE_VALUES[victim.piece_type]
        
        return victim_value - (attacker_value / 100)  # Divide attacker value to break ties