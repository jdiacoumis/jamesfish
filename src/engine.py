import chess
import random
from typing import Optional

class JamesFish:
    def __init__(self):
        self.board = chess.Board()
    
    def make_move(self, move: Optional[str] = None) -> str:
        """
        Make a move on the board. If a move is provided, make that move.
        Otherwise, generate a legal move.
        
        Args:
            move: Optional UCI format move string (e.g., 'e2e4')
            
        Returns:
            str: The move made in UCI format
        """
        if move:
            move_obj = chess.Move.from_uci(move)
            if move_obj in self.board.legal_moves:
                self.board.push(move_obj)
                return move
            raise ValueError(f"Illegal move: {move}")
        
        # For now, just make a random legal move
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        chosen_move = random.choice(legal_moves)
        self.board.push(chosen_move)
        return chosen_move.uci()
    
    def get_board_state(self) -> str:
        """Return the current board state as a FEN string."""
        return self.board.fen()

if __name__ == "__main__":
    # Simple example usage
    engine = JamesFish()
    print(f"Initial position: {engine.get_board_state()}")
    move = engine.make_move()
    print(f"Made move: {move}")
    print(f"New position: {engine.get_board_state()}")