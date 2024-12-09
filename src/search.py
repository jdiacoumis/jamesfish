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
        
        # Move ordering tables
        self.history_table: Dict[tuple, int] = {}  # (from_square, to_square): score
        self.killer_moves: List[List[Optional[chess.Move]]] = [[None, None] for _ in range(128)]  # Two killer moves per ply
        self.pv_table: List[List[Optional[chess.Move]]] = [[None for _ in range(64)] for _ in range(64)]
        
        # Pruning parameters
        self.NULL_MOVE_R = 3
        self.FUTILITY_MARGIN = 80
        self.LMR_DEPTH = 2
        self.LMR_MOVES = 3

    def clear_tables(self):
        """Clear move ordering tables between searches"""
        self.history_table.clear()
        self.killer_moves = [[None, None] for _ in range(128)]
        self.pv_table = [[None for _ in range(64)] for _ in range(64)]
        
    def search(self, board: chess.Board, max_depth: int = 4, max_time: int = 30) -> Tuple[chess.Move, int]:
        self.nodes_searched = 0
        self.start_time = time.time()
        self.max_time = max_time
        self.best_move_found = None
        self.clear_tables()
        
        alpha = float('-inf')
        beta = float('inf')
        
        for current_depth in range(1, max_depth + 1):
            try:
                score = self.negamax(board, current_depth, alpha, beta, 0, True)
                print(f"Depth {current_depth} completed. Best move: {self.best_move_found} Score: {score}")
            except TimeoutError:
                break
                
        if self.best_move_found is None:
            self.best_move_found = list(board.legal_moves)[0]
            
        return self.best_move_found, self.evaluator.evaluate(board)

    def get_move_score(self, board: chess.Board, move: chess.Move, ply: int) -> int:
        """
        Score a move for move ordering.
        Higher scores will be searched first.
        """
        score = 0
        from_square, to_square = move.from_square, move.to_square
        
        # 1. PV move (highest priority)
        if self.pv_table[0][ply] == move:
            return 20000
            
        # 2. Hash move from transposition table
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            stored_move = self.transposition_table[board_hash][2]  # Assuming we store move as third element
            if stored_move == move:
                return 19000
        
        # 3. Captures (MVV-LVA)
        if board.is_capture(move):
            attacker = board.piece_at(from_square)
            victim = board.piece_at(to_square)
            if attacker and victim:
                score = 10000 + (victim.piece_type * 10 - attacker.piece_type)
            return score
            
        # 4. Killer moves
        if move in self.killer_moves[ply]:
            return 9000 + (1000 if move == self.killer_moves[ply][0] else 0)
            
        # 5. History score
        history_key = (from_square, to_square)
        return self.history_table.get(history_key, 0)

    def order_moves(self, board: chess.Board, moves: List[chess.Move], ply: int) -> List[chess.Move]:
        """Sort moves based on their predicted strength"""
        move_scores = [(move, self.get_move_score(board, move, ply)) for move in moves]
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in move_scores]

    def update_history_score(self, move: chess.Move, depth: int):
        """Update history heuristic scores"""
        history_key = (move.from_square, move.to_square)
        self.history_table[history_key] = self.history_table.get(history_key, 0) + depth * depth

    def update_killer_moves(self, move: chess.Move, ply: int):
        """Update killer moves for the current ply"""
        if move != self.killer_moves[ply][0]:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move

    def store_pv_move(self, move: chess.Move, ply: int, depth: int):
        """Store principal variation move"""
        self.pv_table[ply][depth] = move
        
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

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int, is_root: bool = False) -> float:
        self.nodes_searched += 1
        
        if self.should_stop_search():
            raise TimeoutError
            
        # PV node handling
        is_pv = beta > alpha + 1
            
        # Transposition table lookup
        board_hash = board.fen()
        if not is_root and board_hash in self.transposition_table:
            stored_depth, stored_value, stored_move = self.transposition_table[board_hash]
            if stored_depth >= depth:
                return stored_value
                
        original_alpha = alpha
        in_check = board.is_check()
        
        if depth <= 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)
            
        # Static null move pruning
        if self.static_null_move_pruning(board, beta, depth):
            return beta
            
        # Reverse futility pruning
        if self.reverse_futility_pruning(board, alpha, depth):
            return alpha
            
        # Null move pruning
        if not is_pv and depth >= 3 and not in_check and self.can_do_null_move(board):
            board.push(chess.Move.null())
            null_value = -self.negamax(board, depth - 1 - self.NULL_MOVE_R, -beta, -beta + 1, ply + 1)
            board.pop()
            
            if null_value >= beta:
                return beta
        
        moves = list(board.legal_moves)
        moves = self.order_moves(board, moves, ply)
        
        best_value = float('-inf')
        best_move = None
        
        for move_index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            
            # Late Move Reduction
            do_lmr = (
                depth >= self.LMR_DEPTH and
                move_index >= self.LMR_MOVES and
                not in_check and
                not is_capture and
                not gives_check and
                not is_pv
            )
            
            board.push(move)
            
            # Decide search depth
            if do_lmr:
                reduction = 1 + (move_index > 6)
                value = -self.negamax(board, depth - 1 - reduction, -beta, -alpha, ply + 1)
                if value > alpha:  # Re-search if promising
                    value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
                
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
                if is_root:
                    self.best_move_found = move
                    
            alpha = max(alpha, value)
            
            if alpha >= beta:
                # Update killer moves and history for quiet moves
                if not is_capture:
                    self.update_killer_moves(move, ply)
                    self.update_history_score(move, depth)
                break
                
        # Store PV move
        if best_move:
            self.store_pv_move(best_move, ply, depth)
            
        # Store in transposition table
        self.transposition_table[board_hash] = (depth, best_value, best_move)
        
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