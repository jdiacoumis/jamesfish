import chess
from typing import Optional, Tuple, Dict, List
from .evaluation import Evaluator
import time

class SearchEngine:
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.nodes_searched = 0
        self.start_time = 0
        self.max_time = 0
        self.best_move_found = None
        
        # Static parameters
        self.ASPIRATION_WINDOW = 25  # Narrower window for better pruning
        self.FUTILITY_MARGIN = 50
        self.FULL_DEPTH_MOVES = 3
        
        # Move ordering tables
        self.history_table: Dict[tuple, int] = {}
        self.killer_moves: List[Optional[chess.Move]] = [None] * 128
        self.pv_line: List[Optional[chess.Move]] = [None] * 64
        self.transposition_table = {}
        
    def clear_tables(self):
        """Clear move ordering tables between searches"""
        self.history_table.clear()
        self.killer_moves = [None] * 128
        self.pv_line = [None] * 64
        self.transposition_table.clear()

    def should_stop_search(self) -> bool:
        return time.time() - self.start_time > self.max_time

    def search(self, board: chess.Board, max_depth: int = 4, max_time: int = 30) -> Tuple[chess.Move, int]:
        self.nodes_searched = 0
        self.start_time = time.time()
        self.max_time = max_time
        self.best_move_found = None
        self.clear_tables()
        
        previous_score = 0
        for depth in range(1, max_depth + 1):
            try:
                if depth >= 4:
                    alpha = previous_score - self.ASPIRATION_WINDOW
                    beta = previous_score + self.ASPIRATION_WINDOW
                    score = self._search_with_aspiration(board, depth, alpha, beta)
                else:
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
        while True:
            score = self.negamax(board, depth, alpha, beta, 0, True)
            
            if alpha < score < beta:
                return score
                
            if score <= alpha:
                alpha = max(-30000, alpha - self.ASPIRATION_WINDOW)
            else:
                beta = min(30000, beta + self.ASPIRATION_WINDOW)

    def get_move_score(self, board: chess.Board, move: chess.Move, ply: int) -> int:
        """Enhanced move scoring"""
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                return 20000 + (victim.piece_type * 6 - attacker.piece_type)
            return 15000
            
        # Killer moves (only for non-captures)
        if self.killer_moves[ply] == move:
            return 10000
            
        # History score (only for non-captures)
        return self.history_table.get((move.from_square, move.to_square), 0)

    def static_null_move_pruning(self, board: chess.Board, beta: float, depth: int) -> bool:
        """Static null move pruning with depth-dependent margin"""
        if depth < 3 or board.is_check():
            return False
        margin = 100 + 50 * depth
        if self.evaluator.evaluate(board) - margin >= beta:
            return True
        return False

    def can_do_null_move(self, board: chess.Board, depth: int) -> bool:
        """Enhanced null move conditions"""
        if board.is_check():
            return False
        # More aggressive at deeper depths
        if depth >= 6:
            return any(board.pieces(pt, board.turn) 
                      for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT])
        # More conservative at lower depths
        return len(board.pieces(chess.QUEEN, board.turn)) > 0 or len(board.pieces(chess.ROOK, board.turn)) > 0

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int, is_pv: bool = False) -> float:
        """Enhanced negamax with dynamic depth parameters"""
        self.nodes_searched += 1
        
        if self.should_stop_search():
            raise TimeoutError
            
        # Quick draws
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0
            
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
        
        # Dynamic null move R based on depth
        null_move_r = 2 + (depth >= 6)
        
        #Initialise futility
        do_futility = False

        # Various pruning techniques
        if not is_pv:
            # Static null move pruning
            if self.static_null_move_pruning(board, beta, depth):
                return beta
                
            # Null move pruning with dynamic R
            if (depth >= 3 and not in_check and 
                self.can_do_null_move(board, depth)):
                board.push(chess.Move.null())
                value = -self.negamax(board, depth - 1 - null_move_r, -beta, -beta + 1, ply + 1)
                board.pop()
                
                if value >= beta:
                    return beta
            
            # Dynamic futility pruning
            do_futility = (
                depth <= 3 and  # Increased from 2
                not in_check and 
                abs(beta) < 9000
            )
            if do_futility:
                futility_base = static_eval + self.FUTILITY_MARGIN * depth
                if futility_base <= alpha:
                    return futility_base
                
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: self.get_move_score(board, m, ply), reverse=True)
        
        best_move = None
        best_value = float('-inf')
        moves_searched = 0
        
        for move in moves:
            moves_searched += 1
            
            # Dynamic late move pruning
            if (do_futility and moves_searched > self.FULL_DEPTH_MOVES and 
                not board.is_capture(move) and not board.gives_check(move)):
                continue
            
            board.push(move)
            
            # Dynamic LMR
            if moves_searched == 1:
                value = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, is_pv)
            else:
                # More aggressive reductions at deeper depths
                if (depth >= 2 and moves_searched > self.FULL_DEPTH_MOVES and 
                    not in_check and not board.is_capture(move)):
                    reduction = 1 + (moves_searched > 6) + (depth >= 6)
                    value = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, False)
                else:
                    value = alpha + 1
                
                if value > alpha:
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
                if not board.is_capture(move):
                    self.killer_moves[ply] = move
                    # More aggressive history scoring
                    self.history_table[(move.from_square, move.to_square)] = (
                        self.history_table.get((move.from_square, move.to_square), 0) + depth * depth * 2
                    )
                break
        
        # Store in transposition table
        self.transposition_table[board_hash] = (depth, best_value, best_move)
        
        return best_value

    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Enhanced quiescence search"""
        self.nodes_searched += 1
        
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
            
        alpha = max(alpha, stand_pat)
        
        # Only look at captures and checks
        moves = [move for move in board.legal_moves if board.is_capture(move) or board.gives_check(move)]
        moves.sort(key=lambda m: self.get_move_score(board, m, 0), reverse=True)
        
        for move in moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            alpha = max(alpha, score)
                
        return alpha