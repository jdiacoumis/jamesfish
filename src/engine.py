import chess
import random
from typing import Optional
from .evaluation import Evaluator

class JamesFish:
    def __init__(self):
        self.board = chess.Board()
        self.evaluator = Evaluator()
    
    def make_move(self, move: Optional[str] = None) -> str:
        if move:
            move_obj = chess.Move.from_uci(move)
            if move_obj in self.board.legal_moves:
                self.board.push(move_obj)
                return move
            raise ValueError(f"Illegal move: {move}")
        
        # For now, make the move that leads to the best evaluated position
        best_move = None
        best_eval = float('-inf') if self.board.turn else float('inf')
        
        for move in self.board.legal_moves:
            self.board.push(move)
            eval = self.evaluator.evaluate(self.board)
            self.board.pop()
            
            if self.board.turn:  # White to move
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:  # Black to move
                if eval < best_eval:
                    best_eval = eval
                    best_move = move
        
        if best_move is None:
            raise ValueError("No legal moves available")
            
        self.board.push(best_move)
        return best_move.uci()
    
    def get_board_state(self) -> str:
        return self.board.fen()
    
    def get_position_evaluation(self) -> int:
        return self.evaluator.evaluate(self.board)

if __name__ == "__main__":
    # Example usage showing evaluation
    engine = JamesFish()
    print(f"Initial position evaluation: {engine.get_position_evaluation()}")
    move = engine.make_move()
    print(f"Made move: {move}")
    print(f"New position evaluation: {engine.get_position_evaluation()}")