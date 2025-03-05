"""
Fixed self-play implementation with correct CUDA device handling
"""

import numpy as np
import chess
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import copy
import time
import threading
import queue

from ChessRL.mcts import MCTS
from ChessRL.encoding import encode_board, move_to_index

# Global model for multiprocessing
global_network = None
global_config = None
global_device = None
global_mcts = None

def initialize_worker(network_state_dict, config_dict, device_str):
    """Initialize the worker process with the neural network and config"""
    global global_network
    global global_config
    global global_device
    global global_mcts
    
    # Import inside function to prevent circular imports
    from ChessRL.network import ChessNetwork
    from ChessRL.config import Config
    from ChessRL.mcts import MCTS
    
    # Create a new config object
    global_config = Config()
    # Update with the provided config
    for key, value in config_dict.items():
        if hasattr(global_config, key):
            setattr(global_config, key, value)
    
    # Set environment variables for better CPU performance
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads
    
    # Create a new network and load the state dict
    global_device = torch.device(device_str)
    global_network = ChessNetwork(global_config)
    global_network.to(global_device)
    global_network.load_state_dict(network_state_dict)
    global_network.eval()  # Set to evaluation mode
    
    # Pre-create MCTS for each worker
    global_mcts = MCTS(global_network, global_config)
    
    # Fix: Only set CUDA device if using a CUDA device with a specific index
    if global_device.type == 'cuda' and hasattr(global_device, 'index'):
        torch.cuda.set_device(global_device.index)
        # Enable TensorFloat32 for faster computation
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def run_self_play_game(game_idx):
    """
    Run a single self-play game in a separate process
    
    Args:
        game_idx (int): Index of the game for logging
    
    Returns:
        list: Game history with states, policies, and values
    """
    global global_network
    global global_config
    global global_device
    global global_mcts
    
    # Play a game
    board = chess.Board()
    game_history = []
    
    # Pre-allocate memory for states
    move_count = 0
    
    while not board.is_game_over():
        # Temperature parameter for move selection
        temp = global_config.temperature
        if move_count >= global_config.temperature_threshold:
            temp = 0.1  # Almost deterministic selection
        
        # Run optimized MCTS
        root = global_mcts.search(board)
        
        # Select move based on visit counts and temperature
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if temp == 0 or len(actions) == 1:
            # Choose the move with highest visit count
            best_idx = np.argmax(visit_counts)
            action = actions[best_idx]
        else:
            # Apply temperature and sample
            visit_count_distribution = visit_counts ** (1 / temp)
            sum_visits = np.sum(visit_count_distribution)
            if sum_visits > 0:
                visit_count_distribution = visit_count_distribution / sum_visits
                action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                action = actions[action_idx]
            else:
                # Fallback to highest count if numerical issues
                best_idx = np.argmax(visit_counts)
                action = actions[best_idx]
        
        # Store the current state, MCTS policy, and turn
        encoded_board = encode_board(board)
        
        # Efficiently create policy vector
        policy = np.zeros(4672, dtype=np.float32)  # Size of policy vector
        for i, a in enumerate(actions):
            try:
                move_idx = move_to_index(a)
                if 0 <= move_idx < 4672:
                    policy[move_idx] = visit_counts[i]
            except Exception:
                continue
                
        # Normalize policy
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback to uniform policy
            indices = np.random.choice(4672, len(actions), replace=False)
            policy[indices] = 1.0 / len(actions)
        
        game_history.append({
            'board': encoded_board.cpu().numpy() if global_device.type == 'cuda' else encoded_board.numpy(),
            'policy': policy,
            'turn': board.turn
        })
        
        # Execute the chosen move
        board.push(action)
        move_count += 1
    
    # Game result
    result = board.result()
    winner = None
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    
    # Add game result to all stored states
    for state in game_history:
        if winner is None:  # Draw
            state['value'] = 0
        else:
            # Win: 1, Loss: -1 (from perspective of the player who made the move)
            state['value'] = 1 if state['turn'] == winner else -1
    
    return game_history

class SelfPlay:
    """
    Optimized self-play for generating training data
    """
    def __init__(self, network, config):
        """
        Initialize the self-play module
        
        Args:
            network (ChessNetwork): The neural network for position evaluation
            config (Config): Configuration parameters
        """
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device
        
        # Create optimized MCTS
        self.mcts = MCTS(network, config)
        
        # Determine optimal number of processes based on hardware
        if self.device.type == 'cuda':
            # For GPU, use more threads since they share the GPU
            self.num_processes = min(os.cpu_count(), 6)
        else:
            # For CPU, allocate fewer processes to avoid oversubscription
            self.num_processes = max(1, os.cpu_count() // 2)
            
        print(f"Using {self.num_processes} processes for self-play")
        
        # Enable CUDA optimizations if available
        if self.device.type == 'cuda':
            # Enable TensorFloat32 for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
    
    def execute_episode(self):
        """
        Play a full game of self-play with optimized efficiency
        
        Returns:
            list: Game history with states, policies, and values
        """
        # Pre-allocate memory and warm up GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        board = chess.Board()
        game_history = []
        move_count = 0
        
        # Set thread optimizations
        torch.set_num_threads(4)  # Limit threads to reduce overhead
        
        print("Playing self-play game...")
        game_start_time = time.time()
        
        while not board.is_game_over():
            move_start_time = time.time()
            
            # Temperature parameter for move selection
            temp = self.config.temperature
            if move_count >= self.config.temperature_threshold:
                temp = 0.1  # Almost deterministic selection
            
            # Run MCTS with timing
            mcts_start = time.time()
            root = self.mcts.search(board)
            mcts_time = time.time() - mcts_start
            
            # Select move based on visit counts and temperature
            policy_start = time.time()
            visit_counts = np.array([child.visit_count for child in root.children.values()])
            actions = list(root.children.keys())
            
            if temp == 0 or len(actions) == 1:
                best_idx = np.argmax(visit_counts)
                action = actions[best_idx]
            else:
                # Apply temperature and sample
                visit_count_distribution = visit_counts ** (1 / temp)
                visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
                action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                action = actions[action_idx]
            
            # Store the current state, MCTS policy, and turn
            policy = np.zeros(4672, dtype=np.float32)  # Size of policy vector
            
            # Use vectorized operations for efficiency
            for i, a in enumerate(actions):
                try:
                    move_idx = move_to_index(a)
                    if 0 <= move_idx < 4672:
                        policy[move_idx] = visit_counts[i]
                except Exception:
                    continue
            
            # Normalize policy
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            
            # Store state information
            with torch.no_grad():
                encoded_board = encode_board(board, self.device)
            
            game_history.append({
                'board': encoded_board,
                'policy': policy,
                'turn': board.turn
            })
            
            # Execute the chosen move
            board.push(action)
            move_count += 1
            
            move_time = time.time() - move_start_time
            if move_count % 10 == 0:
                print(f"Move {move_count}: {move_time:.3f}s (MCTS: {mcts_time:.3f}s)", flush=True)
        
        # Game result
        result = board.result()
        winner = None
        if result == "1-0":
            winner = chess.WHITE
        elif result == "0-1":
            winner = chess.BLACK
            
        game_time = time.time() - game_start_time
        print(f"Game completed in {game_time:.1f}s ({move_count} moves, {game_time/move_count:.2f}s per move)", flush=True)
        print(f"Game result: {result}", flush=True)
        
        # Add game result to all stored states
        for state in game_history:
            if winner is None:  # Draw
                state['value'] = 0
            else:
                # Win: 1, Loss: -1 (from perspective of the player who made the move)
                state['value'] = 1 if state['turn'] == winner else -1
        
        return game_history
    
    def execute_parallel_self_play(self, num_games):
        """
        Execute multiple self-play games in parallel with optimized resource usage
        
        Args:
            num_games (int): Number of games to play
            
        Returns:
            list: Combined game history from all games
        """
        # Prepare the network state dict (CPU version for sharing)
        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        
        # Prepare config as a dictionary
        config_dict = {
            'num_simulations': self.config.num_simulations,
            'c_puct': self.config.c_puct,
            'dirichlet_alpha': self.config.dirichlet_alpha,
            'dirichlet_noise_factor': self.config.dirichlet_noise_factor,
            'temperature': self.config.temperature,
            'temperature_threshold': self.config.temperature_threshold,
            'n_features': self.config.n_features,
            'n_residual_blocks': self.config.n_residual_blocks,
            'batch_mcts_size': self.config.batch_mcts_size * 2  # Increase batch size for better efficiency
        }
        
        # Clear GPU memory before parallel processing
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        all_game_history = []
        start_time = time.time()
        
        # Fix: Pass the correct device string for CUDA with index
        device_str = str(self.device)
        
        # For multi-GPU, make sure the device string has a proper index
        if self.device.type == 'cuda' and not hasattr(self.device, 'index'):
            device_str = 'cuda:0'  # Default to first GPU
        
        # Use different parallel strategies based on device
        if self.device.type == 'cuda':
            # For GPU, use multiprocessing with CPU-only workers
            # This avoids CUDA issues in subprocess
            cpu_config_dict = config_dict.copy()
            device_str = 'cpu'  # Force CPU for worker processes
            
            multiprocessing.set_start_method('spawn', force=True)
            
            try:
                # Create a process pool with CPU workers
                with ProcessPoolExecutor(max_workers=self.num_processes, 
                                      initializer=initialize_worker,
                                      initargs=(network_state_dict, cpu_config_dict, device_str)) as executor:
                    
                    # Submit all games at once
                    game_futures = [executor.submit(run_self_play_game, i) for i in range(num_games)]
                    
                    # Collect results as they complete
                    completed = 0
                    for future in game_futures:
                        try:
                            game_data = future.result()
                            all_game_history.extend(game_data)
                            completed += 1
                            
                            # Report progress periodically
                            if completed % max(1, num_games // 10) == 0:
                                elapsed = time.time() - start_time
                                games_per_hour = (completed / elapsed) * 3600
                                print(f"Completed {completed}/{num_games} games "
                                    f"({games_per_hour:.1f} games/hour)")
                        except Exception as e:
                            print(f"Error in game: {e}")
            except Exception as e:
                print(f"Error in parallel execution: {e}")
                # Fallback to single-process execution
                print("Falling back to sequential execution")
                all_game_history = []
                for i in range(num_games):
                    print(f"Running game {i+1}/{num_games}")
                    game_data = self.execute_episode()
                    all_game_history.extend(game_data)
        else:
            # For CPU-only, use thread pool
            # Each thread gets a deep copy of the MCTS object
            mcts_copies = [copy.deepcopy(self.mcts) for _ in range(self.num_processes)]
            
            def thread_self_play(game_idx):
                # Deterministically assign a thread-local MCTS
                thread_id = game_idx % len(mcts_copies)
                mcts = mcts_copies[thread_id]
                
                # Play game using the thread's MCTS instance
                board = chess.Board()
                game_history = []
                
                while not board.is_game_over():
                    # Run MCTS
                    root = mcts.search(board)
                    
                    # Select move (same logic as before)
                    temp = self.config.temperature
                    if len(game_history) >= self.config.temperature_threshold:
                        temp = 0.1
                        
                    visit_counts = np.array([child.visit_count for child in root.children.values()])
                    actions = list(root.children.keys())
                    
                    if temp == 0 or len(actions) == 1:
                        best_idx = np.argmax(visit_counts)
                        action = actions[best_idx]
                    else:
                        visit_count_distribution = visit_counts ** (1 / temp)
                        visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
                        action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                        action = actions[action_idx]
                    
                    # Store state info
                    encoded_board = encode_board(board)
                    policy = np.zeros(4672)
                    
                    for i, a in enumerate(actions):
                        try:
                            move_idx = move_to_index(a)
                            if 0 <= move_idx < 4672:
                                policy[move_idx] = visit_counts[i]
                        except Exception:
                            continue
                            
                    policy_sum = np.sum(policy)
                    if policy_sum > 0:
                        policy = policy / policy_sum
                    
                    game_history.append({
                        'board': encoded_board,
                        'policy': policy,
                        'turn': board.turn
                    })
                    
                    # Make the move
                    board.push(action)
                
                # Process game result
                result = board.result()
                winner = None
                if result == "1-0":
                    winner = chess.WHITE
                elif result == "0-1":
                    winner = chess.BLACK
                
                # Add values
                for state in game_history:
                    if winner is None:
                        state['value'] = 0
                    else:
                        state['value'] = 1 if state['turn'] == winner else -1
                
                return game_history
            
            # Use ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                game_futures = [executor.submit(thread_self_play, i) for i in range(num_games)]
                
                # Collect results
                completed = 0
                for future in game_futures:
                    game_data = future.result()
                    all_game_history.extend(game_data)
                    completed += 1
                    
                    if completed % max(1, num_games // 10) == 0:
                        elapsed = time.time() - start_time
                        games_per_hour = (completed / elapsed) * 3600
                        print(f"Completed {completed}/{num_games} games "
                              f"({games_per_hour:.1f} games/hour)")
        
        # Convert NumPy arrays to torch tensors where needed
        for state in all_game_history:
            if isinstance(state['board'], np.ndarray):
                state['board'] = torch.FloatTensor(state['board']).to(self.device)
        
        return all_game_history