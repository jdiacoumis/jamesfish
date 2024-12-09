import chess
import time
from .engine import JamesFish
from typing import Dict, List

def run_depth_benchmark(starting_fen: str = chess.STARTING_FEN, 
                       max_test_depth: int = 6,
                       positions: int = 3) -> Dict:
    """
    Run benchmark across different depths and collect performance metrics.
    
    Args:
        starting_fen: Starting position in FEN notation
        max_test_depth: Maximum depth to test
        positions: Number of positions to test from the starting position
    """
    engine = JamesFish()
    results = {}
    
    # Use starting position and a few positions after some moves
    test_positions = []
    board = chess.Board(starting_fen)
    test_positions.append(board.fen())
    
    # Generate a few more positions by making some moves
    for _ in range(positions - 1):
        if len(list(board.legal_moves)) > 0:
            move = list(board.legal_moves)[0]  # Just take first legal move
            board.push(move)
            test_positions.append(board.fen())
    
    # Test each depth
    for depth in range(1, max_test_depth + 1):
        depth_results = {
            'total_time': 0.0,
            'total_nodes': 0,
            'positions_tested': 0
        }
        
        print(f"\nTesting Depth {depth}...")
        
        for pos_num, fen in enumerate(test_positions, 1):
            board = chess.Board(fen)
            engine.board = board
            
            start_time = time.time()
            nodes_before = engine.search_engine.nodes_searched
            
            # Set a reasonable timeout for each search
            try:
                move, eval_score = engine.search_engine.search(
                    board,
                    max_depth=depth,
                    max_time=300  # 5 minutes max per position
                )
                
                end_time = time.time()
                nodes_searched = engine.search_engine.nodes_searched - nodes_before
                
                depth_results['total_time'] += end_time - start_time
                depth_results['total_nodes'] += nodes_searched
                depth_results['positions_tested'] += 1
                
                print(f"Position {pos_num}: Time: {end_time - start_time:.2f}s, "
                      f"Nodes: {nodes_searched}, "
                      f"NPS: {int(nodes_searched/(end_time - start_time))}")
                
            except TimeoutError:
                print(f"Depth {depth} timed out on position {pos_num}")
                break
        
        if depth_results['positions_tested'] > 0:
            results[depth] = {
                'avg_time': depth_results['total_time'] / depth_results['positions_tested'],
                'avg_nodes': depth_results['total_nodes'] / depth_results['positions_tested'],
                'avg_nps': int(depth_results['total_nodes'] / depth_results['total_time']
                             if depth_results['total_time'] > 0 else 0),
                'positions_completed': depth_results['positions_tested']
            }
            
            print(f"\nDepth {depth} Summary:")
            print(f"Average Time: {results[depth]['avg_time']:.2f} seconds")
            print(f"Average Nodes: {int(results[depth]['avg_nodes'])}")
            print(f"Nodes per Second: {results[depth]['avg_nps']}")
    
    return results

if __name__ == "__main__":
    print("Starting Chess Engine Benchmark...")
    results = run_depth_benchmark(
        max_test_depth=6,  # Test depths 1-6
        positions=3        # Test 3 different positions
    )