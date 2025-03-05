"""
Main training script for the chess reinforcement learning model
"""

import argparse
import os
import random
import numpy as np
import torch
import time

from ChessRL.config import Config
from ChessRL.network import ChessNetwork
from ChessRL.trainer import Trainer
from ChessRL.self_play import SelfPlay

# Add after device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Can improve speed if input sizes don't change


def print_gpu_info():
    """Print GPU information if available"""
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Set environment variables for better GPU performance
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def main():
    parser = argparse.ArgumentParser(description='Train a chess reinforcement learning model')
    parser.add_argument('--iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--load', type=str, default=None, help='Load a saved model')
    parser.add_argument('--self-play-games', type=int, default=100, help='Number of self-play games per iteration')
    parser.add_argument('--mcts-simulations', type=int, default=None, help='Number of MCTS simulations per move')
    parser.add_argument('--parallel', action='store_true', help='Use parallel self-play')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize components
    config = Config()
    
    # Override config with command line arguments
    if args.self_play_games:
        config.num_self_play_games = args.self_play_games
    if args.mcts_simulations:
        config.num_simulations = args.mcts_simulations
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print_gpu_info()
    
    # Initialize the network with optimizations for speed
    network = ChessNetwork(config)
    network.to(device)
    
    # Initialize trainer and self-play
    trainer = Trainer(network, config)
    self_play = SelfPlay(network, config)
    
    # Load a saved model if specified
    if args.load:
        print(f"Loading model from {args.load}")
        trainer.load_model(args.load)
    
    print("Starting training...")
    total_start_time = time.time()
    
    for iteration in range(1, args.iterations + 1):
        iteration_start_time = time.time()
        print(f"\nTraining Iteration {iteration}")
        
        # Self-play to generate data
        sp_start_time = time.time()
        print(f"Executing {config.num_self_play_games} self-play games...")
        
        if args.parallel:
            # Use parallel self-play
            game_examples = self_play.execute_parallel_self_play(config.num_self_play_games)
        else:
            # Sequential self-play
            game_examples = []
            for i in range(config.num_self_play_games):
                iteration_start_self_play_time = time.time() 
                if i % 10 == 0:
                    print(f"Self-play game {i}/{config.num_self_play_games}")
                game_data = self_play.execute_episode()
                game_examples.extend(game_data)
                print(f"Self play game {i} time: {time.time() - iteration_start_self_play_time} seconds")
        
        sp_end_time = time.time()
        sp_duration = sp_end_time - sp_start_time
        
        num_positions = len(game_examples)
        positions_per_sec = num_positions / sp_duration if sp_duration > 0 else 0
        print(f"Generated {num_positions} examples from self-play in {sp_duration:.1f}s ({positions_per_sec:.1f} positions/s)")
        
        # Train network with examples
        train_start_time = time.time()
        print("Training network with generated examples...")
        trainer.train(game_examples)
        train_end_time = time.time()
        
        # Save the model periodically
        if iteration % 5 == 0 or iteration == args.iterations:
            trainer.save_model(f"chess_model_iter_{iteration}.pt")
        
        # Calculate and print timing information
        iter_duration = time.time() - iteration_start_time
        train_duration = train_end_time - train_start_time
        total_duration = time.time() - total_start_time
        
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"Iteration {iteration} completed in {iter_duration:.1f}s "
              f"(Self-play: {sp_duration:.1f}s, Training: {train_duration:.1f}s)")
        print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    
    # Save final model
    trainer.save_model("chess_model_final.pt")
    print("Training complete!")


if __name__ == "__main__":
    main()