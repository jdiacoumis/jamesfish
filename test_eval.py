
import sys
sys.path.append('C:\\Users\\jdiacoumis\\OneDrive - Quantium\\Documents\\GitHub\\jamesfish')
from src.evaluation import Evaluator
import chess

def main():
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

if __name__ == "__main__":
    main()
