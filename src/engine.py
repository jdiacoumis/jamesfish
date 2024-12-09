import chess
from typing import Optional, Tuple
from .evaluation import Evaluator
from .search import SearchEngine

class JamesFish:
    def __init__(self):
        self.board = chess.Board()
        self.evaluator = Evaluator()
        self.search_engine = SearchEngine(self.evaluator)
    
    def make_move(self, move: Optional[str] = None) -> str:
        """Make a move on the board"""
        if move:
            move_obj = chess.Move.from_uci(move)
            if move_obj in self.board.legal_moves:
                self.board.push(move_obj)
                return move
            raise ValueError(f"Illegal move: {move}")
        
        # Use search engine to find best move
        best_move, eval_score = self.search_engine.search(
            self.board,
            max_depth=4,  # Adjust these parameters as needed
            max_time=30   # 30 seconds max per move
        )
        
        self.board.push(best_move)
        return best_move.uci()
    
    def get_board_state(self) -> str:
        """Return current board state in FEN notation"""
        return self.board.fen()
    
    def get_position_evaluation(self) -> int:
        """Get static evaluation of current position"""
        return self.evaluator.evaluate(self.board)

if __name__ == "__main__":
    # Test the engine
    engine = JamesFish()
    
    # Print initial position
    print(f"Initial position: {engine.get_board_state()}")
    print(f"Initial evaluation: {engine.get_position_evaluation()}")
    
    # Make a move and show results
    move = engine.make_move()
    print(f"Engine played: {move}")
    print(f"New evaluation: {engine.get_position_evaluation()}")