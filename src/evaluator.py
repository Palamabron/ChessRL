#!/usr/bin/env python

import os
import torch
import numpy as np
from alpha_net import ChessNet
from chess_board import board as c_board
import encoder_decoder as ed
import copy
import torch.multiprocessing as mp
from MCTS_chess import UCT_search, do_decode_n_move_pieces
import pickle
import types
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set multiprocessing start method to spawn
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Create necessary directories
os.makedirs("./evaluator_data/", exist_ok=True)
os.makedirs("./evaluation_results/", exist_ok=True)

def save_as_pickle(filename, data):
    completeName = os.path.join("./evaluator_data/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

# Monkey patch the move_piece method to handle None values
def patched_move_piece(self, initial_position, final_position, promoted_piece="queen"):
    # Handle None promoted_piece
    if promoted_piece is None:
        promoted_piece = "queen"  # Default to queen if None
    
    promoted_piece = promoted_piece[:1].lower()
    if promoted_piece == "k":
        promoted_piece = "n"
    
    if self.player == 0:
        promoted = False
        i, j = initial_position
        piece = self.current_board[i,j]
        self.current_board[i,j] = " "
        i, j = final_position
        if piece == "R" and initial_position == (7,0):
            self.R1_move_count += 1
        if piece == "R" and initial_position == (7,7):
            self.R2_move_count += 1
        if piece == "K":
            self.K_move_count += 1
        x, y = initial_position
        if piece == "P":
            if abs(x-i) > 1:
                self.en_passant = j; self.en_passant_move = self.move_count
            if abs(y-j) == 1 and self.current_board[i,j] == " ": # En passant capture
                self.current_board[i+1,j] = " "
            if i == 0 and promoted_piece in ["r","b","n","q"]:
                self.current_board[i,j] = promoted_piece.upper()
                promoted = True
        if promoted == False:
            self.current_board[i,j] = piece
        self.player = 1
        self.move_count += 1

    elif self.player == 1:
        promoted = False
        i, j = initial_position
        piece = self.current_board[i,j]
        self.current_board[i,j] = " "
        i, j = final_position
        if piece == "r" and initial_position == (0,0):
            self.r1_move_count += 1
        if piece == "r" and initial_position == (0,7):
            self.r2_move_count += 1
        if piece == "k":
            self.k_move_count += 1
        x, y = initial_position
        if piece == "p":
            if abs(x-i) > 1:
                self.en_passant = j; self.en_passant_move = self.move_count
            if abs(y-j) == 1 and self.current_board[i,j] == " ": # En passant capture
                self.current_board[i-1,j] = " "
            if i == 7 and promoted_piece in ["r","b","n","q"]:
                self.current_board[i,j] = promoted_piece
                promoted = True
        if promoted == False:
            self.current_board[i,j] = piece
        self.player = 0
        self.move_count += 1

    else:
        print("Invalid move: ",initial_position,final_position,promoted_piece)


class Arena:
    def __init__(self, model1, model2, model1_name, model2_name, mcts_simulations=800, max_moves=100):
        self.model1 = model1  # First model
        self.model2 = model2  # Second model
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.mcts_simulations = mcts_simulations
        self.max_moves = max_moves
        
    def play_game(self, game_id, model1_plays_white=None):
        """Play a single game between the two models."""
        if model1_plays_white is None:
            # Randomly decide which model plays white
            model1_plays_white = np.random.choice([True, False])
            
        white_model = self.model1 if model1_plays_white else self.model2
        black_model = self.model2 if model1_plays_white else self.model1
        
        white_name = self.model1_name if model1_plays_white else self.model2_name
        black_name = self.model2_name if model1_plays_white else self.model1_name
        
        print(f"Game {game_id}: {white_name} (White) vs {black_name} (Black)")
        
        # Initialize the board
        current_board = c_board()
        
        # Apply the monkey patch to the current board instance
        current_board.move_piece = types.MethodType(patched_move_piece, current_board)
        
        checkmate = False
        states = []
        moves_history = []
        
        try:
            while checkmate == False and current_board.move_count <= self.max_moves:
                # Check for draws by repetition
                draw_counter = 0
                for s in states:
                    if np.array_equal(current_board.current_board, s):
                        draw_counter += 1
                if draw_counter == 3:
                    print(f"Game {game_id}: Draw by repetition after {current_board.move_count} moves")
                    return 0.5, current_board.move_count, moves_history  # Draw
                
                # Save current state
                states.append(copy.deepcopy(current_board.current_board))
                
                # Apply monkey patch to the original UCT_search function's internal board copies
                original_deepcopy = copy.deepcopy
                
                def patched_deepcopy(obj, *args, **kwargs):
                    result = original_deepcopy(obj, *args, **kwargs)
                    if isinstance(result, c_board):
                        result.move_piece = types.MethodType(patched_move_piece, result)
                    return result
                
                # Temporarily replace copy.deepcopy with our patched version
                copy.deepcopy = patched_deepcopy
                
                try:
                    # Choose which model to use based on current player
                    current_model = white_model if current_board.player == 0 else black_model
                    
                    # Run MCTS search
                    best_move, _ = UCT_search(current_board, self.mcts_simulations, current_model)
                    
                    # Record the move
                    i_pos, f_pos, prom = ed.decode_action(current_board, best_move)
                    moves_history.append((current_board.player, i_pos[0], f_pos[0], prom[0]))
                    
                    # Apply the selected move
                    current_board = do_decode_n_move_pieces(current_board, best_move)
                finally:
                    # Restore the original deepcopy function
                    copy.deepcopy = original_deepcopy
                
                # Status update every 10 moves
                if current_board.move_count % 10 == 0:
                    print(f"Game {game_id} - Move {current_board.move_count}")
                
                # Check for checkmate
                if current_board.check_status() == True and current_board.in_check_possible_moves() == []:
                    if current_board.player == 0:  # black wins
                        winner = black_name
                        result = 0 if model1_plays_white else 1
                    elif current_board.player == 1:  # white wins
                        winner = white_name
                        result = 1 if model1_plays_white else 0
                    checkmate = True
                    print(f"Game {game_id}: {winner} wins by checkmate after {current_board.move_count} moves")
                    return result, current_board.move_count, moves_history
            
            # If we reach the maximum moves without a conclusion, it's a draw
            print(f"Game {game_id}: Draw by move limit ({self.max_moves})")
            return 0.5, current_board.move_count, moves_history
            
        except Exception as e:
            print(f"Error in Game {game_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, []
    
    def evaluate(self, num_games=100, cpu_id=0):
        """Play multiple games and return the win rate for model1."""
        results = []
        game_lengths = []
        all_moves = []
        
        for i in range(num_games):
            # Alternate which model plays white
            model1_plays_white = (i % 2 == 0)
            result, moves, history = self.play_game(i + cpu_id * num_games, model1_plays_white)
            
            if result is not None:
                results.append(result)
                game_lengths.append(moves)
                all_moves.append(history)
        
        # Calculate win rate for model1
        model1_wins = results.count(1)
        model1_losses = results.count(0)
        draws = results.count(0.5)
        
        win_rate = model1_wins / len(results) if results else 0
        draw_rate = draws / len(results) if results else 0
        
        print(f"Process {cpu_id} - Results:")
        print(f"{self.model1_name}: {model1_wins} wins, {model1_losses} losses, {draws} draws")
        print(f"Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}")
        print(f"Average game length: {np.mean(game_lengths):.1f} moves")
        
        return {
            "model1_name": self.model1_name,
            "model2_name": self.model2_name,
            "model1_wins": model1_wins,
            "model1_losses": model1_losses,
            "draws": draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "game_lengths": game_lengths,
            "all_moves": all_moves
        }

def evaluate_model_pair(model1_path, model2_path, model1_name, model2_name, games_per_worker, mcts_sims, cpu_id):
    """Evaluate a pair of models on one CPU."""
    # Load models
    model1 = ChessNet()
    model2 = ChessNet()
    
    cuda = torch.cuda.is_available()
    if cuda:
        model1.cuda()
        model2.cuda()
    
    model1.eval()
    model2.eval()
    
    # Load checkpoints
    checkpoint1 = torch.load(model1_path, map_location="cuda" if cuda else "cpu")
    checkpoint2 = torch.load(model2_path, map_location="cuda" if cuda else "cpu")
    
    model1.load_state_dict(checkpoint1['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    
    # Create arena and run evaluation
    arena = Arena(model1, model2, model1_name, model2_name, mcts_sims)
    results = arena.evaluate(games_per_worker, cpu_id)
    
    return results

def run_evaluation(args):
    """Run a full evaluation with multiple processes."""
    # Prepare model paths and names
    model_paths = []
    model_names = []
    
    # Find all model files and sort them
    for file in sorted(os.listdir("./model_data/")):
        if file.endswith(".pth.tar"):
            model_paths.append(os.path.join("./model_data/", file))
            model_names.append(file.replace(".pth.tar", ""))
    
    if len(model_paths) < 2:
        print("Need at least 2 models to evaluate.")
        return
    
    print(f"Found {len(model_paths)} models: {', '.join(model_names)}")
    
    # If specific models are provided, use only those
    if args.model1 and args.model2:
        model1_path = os.path.join("./model_data/", args.model1 if args.model1.endswith(".pth.tar") else args.model1 + ".pth.tar")
        model2_path = os.path.join("./model_data/", args.model2 if args.model2.endswith(".pth.tar") else args.model2 + ".pth.tar")
        
        if not os.path.exists(model1_path) or not os.path.exists(model2_path):
            print(f"One or both specified models not found.")
            return
            
        model1_name = args.model1.replace(".pth.tar", "")
        model2_name = args.model2.replace(".pth.tar", "")
        
        model_pairs = [(model1_path, model2_path, model1_name, model2_name)]
    else:
        # Compare consecutive models
        model_pairs = []
        for i in range(len(model_paths) - 1):
            model_pairs.append((model_paths[i], model_paths[i+1], model_names[i], model_names[i+1]))
    
    # Determine number of processes
    num_processes = min(args.workers, os.cpu_count() or 1)
    print(f"Using {num_processes} workers for evaluation")
    
    # Run evaluations
    all_results = []
    for model1_path, model2_path, model1_name, model2_name in model_pairs:
        print(f"\nEvaluating: {model1_name} vs {model2_name}")
        print(f"Playing {args.games} games ({args.games_per_worker} per worker) with {args.mcts_sims} MCTS simulations")
        
        processes = []
        manager = mp.Manager()
        result_list = manager.list()
        
        for i in range(num_processes):
            p = mp.Process(target=lambda: result_list.append(
                evaluate_model_pair(model1_path, model2_path, model1_name, model2_name, 
                                   args.games_per_worker, args.mcts_sims, i)
            ))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # Aggregate results
        model1_wins = sum(r["model1_wins"] for r in result_list)
        model1_losses = sum(r["model1_losses"] for r in result_list)
        draws = sum(r["draws"] for r in result_list)
        total_games = model1_wins + model1_losses + draws
        
        if total_games > 0:
            win_rate = model1_wins / total_games
            draw_rate = draws / total_games
            
            # Combine game lengths
            game_lengths = []
            for r in result_list:
                game_lengths.extend(r["game_lengths"])
            
            # Combine all moves
            all_moves = []
            for r in result_list:
                all_moves.extend(r["all_moves"])
            
            print("\nFinal Results:")
            print(f"{model1_name}: {model1_wins} wins, {model1_losses} losses, {draws} draws")
            print(f"Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}")
            print(f"Average game length: {np.mean(game_lengths):.1f} moves")
            
            # Save results
            result_summary = {
                "model1_name": model1_name,
                "model2_name": model2_name,
                "model1_wins": model1_wins,
                "model1_losses": model1_losses,
                "draws": draws,
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "game_lengths": game_lengths,
                "all_moves": all_moves,
                "timestamp": time.strftime("%Y%m%d-%H%M%S")
            }
            
            all_results.append(result_summary)
            
            # Save result to file
            result_filename = f"eval_{model1_name}_vs_{model2_name}_{time.strftime('%Y%m%d-%H%M%S')}.pkl"
            save_as_pickle(result_filename, result_summary)
    
    # Generate plots
    if all_results:
        plot_evaluation_results(all_results)

def plot_evaluation_results(results):
    """Generate plots from evaluation results."""
    os.makedirs("./evaluation_results/plots/", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Win rates plot
    model_pairs = [f"{r['model1_name']} vs\n{r['model2_name']}" for r in results]
    win_rates = [r["win_rate"] for r in results]
    draw_rates = [r["draw_rate"] for r in results]
    loss_rates = [1 - r["win_rate"] - r["draw_rate"] for r in results]
    
    axes[0].bar(model_pairs, win_rates, label="Wins", color="green")
    axes[0].bar(model_pairs, draw_rates, bottom=win_rates, label="Draws", color="gray")
    axes[0].bar(model_pairs, loss_rates, bottom=[w+d for w, d in zip(win_rates, draw_rates)], label="Losses", color="red")
    
    axes[0].set_ylabel("Percentage")
    axes[0].set_title("Match Results")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Game lengths plot
    for i, r in enumerate(results):
        axes[1].boxplot(r["game_lengths"], positions=[i+1], widths=0.6)
    
    axes[1].set_xticks(range(1, len(results) + 1))
    axes[1].set_xticklabels(model_pairs)
    axes[1].set_ylabel("Number of Moves")
    axes[1].set_title("Game Lengths")
    
    plt.tight_layout()
    plt.savefig(f"./evaluation_results/plots/evaluation_summary_{time.strftime('%Y%m%d-%H%M%S')}.png")
    
    # Win progression over time
    if len(results) > 1:
        plt.figure(figsize=(10, 6))
        for i, r in enumerate(results):
            if i > 0:  # Skip the first model as it doesn't have a predecessor
                plt.plot(i, r["win_rate"], 'go', markersize=10)
        
        plt.xticks(range(1, len(results)))
        plt.xlabel("Model Iteration")
        plt.ylabel("Win Rate Against Previous Model")
        plt.title("Training Progress: Win Rate Against Previous Model")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"./evaluation_results/plots/progress_{time.strftime('%Y%m%d-%H%M%S')}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero Chess Models')
    parser.add_argument('--games', type=int, default=100, help='Total number of games to play')
    parser.add_argument('--games_per_worker', type=int, default=10, help='Games per worker')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--mcts_sims', type=int, default=800, help='Number of MCTS simulations per move')
    parser.add_argument('--model1', type=str, help='First model to evaluate (optional)')
    parser.add_argument('--model2', type=str, help='Second model to evaluate (optional)')
    
    args = parser.parse_args()
    
    # Override games_per_worker if total games is specified
    if args.games < args.games_per_worker * args.workers:
        args.games_per_worker = max(1, args.games // args.workers)
    
    run_evaluation(args)