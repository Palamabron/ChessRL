#!/usr/bin/env python

import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from alpha_net import ChessNet
from chess_board import board as c_board
import encoder_decoder as ed
import copy
import datetime
import types
import argparse
import time
import gc

# Enable GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set multiprocessing start method to spawn
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Create necessary directories
os.makedirs("./model_data/", exist_ok=True)
os.makedirs("./datasets/iter0/", exist_ok=True)
os.makedirs("./evaluator_data/", exist_ok=True)

# Import required functions from MCTS_chess
from MCTS_chess import (
    UCT_search, do_decode_n_move_pieces, get_policy, UCTNode, DummyNode
)

# Custom Dataset class for proper batch processing
class ChessDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        print(f"Initialized dataset with {len(samples)} samples")
        
        # Validate a few samples to catch potential issues early
        for i in range(min(5, len(samples))):
            try:
                state, policy, value = samples[i]
                state_shape = state.shape
                policy_shape = policy.shape
                print(f"Sample {i} shapes - state: {state_shape}, policy: {policy_shape}, value: {value}")
            except Exception as e:
                print(f"Warning: Issue with sample {i}: {e}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            state, policy, value = self.samples[idx]
            
            # Ensure state has correct shape and type
            if state.ndim != 3:
                raise ValueError(f"Expected 3D state array, got shape {state.shape}")
                
            # Convert board state to the format expected by the network
            state = state.transpose(2, 0, 1).astype(np.float32)
            policy = policy.astype(np.float32)
            value = float(value)
            
            return state, policy, value
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a dummy sample to avoid crashing
            dummy_state = np.zeros((22, 8, 8), dtype=np.float32)
            dummy_policy = np.zeros(4672, dtype=np.float32)
            dummy_policy[0] = 1.0  # Put all probability on first action
            dummy_value = 0.0
            return dummy_state, dummy_policy, dummy_value

# Custom AlphaLoss class (adapted from original)
class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, value_pred, value_target, policy_pred, policy_target):
        value_error = (value_pred - value_target) ** 2
        policy_error = torch.sum((-policy_target * 
                                (1e-6 + policy_pred).log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error

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

def save_as_pickle(filename, data, iteration=0):
    directory = f"./datasets/iter{iteration}/"
    os.makedirs(directory, exist_ok=True)
    completeName = os.path.join(directory, filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

# Optimized MCTS self-play function
def optimized_MCTS_self_play(net, num_games, cpu_id, iteration, mcts_simulations=800):
    print(f"Starting optimized MCTS self-play process {cpu_id}")
    
    # Explicitly set to evaluation mode and optimize for inference
    net.eval()
    
    # Use torch.inference_mode() to reduce memory usage and increase speed
    with torch.inference_mode():
        for idxx in range(num_games):
            current_board = c_board()
            
            # Apply the monkey patch to the current board instance
            current_board.move_piece = types.MethodType(patched_move_piece, current_board)
            
            checkmate = False
            dataset = []
            states = []
            value = 0
            
            try:
                print(f"Process {cpu_id}, Game {idxx} started")
                
                while checkmate == False and current_board.move_count <= 100:  # Full game length
                    # Check for draws by repetition
                    draw_counter = 0
                    for s in states:
                        if np.array_equal(current_board.current_board, s):
                            draw_counter += 1
                    if draw_counter == 3:
                        break
                    
                    # Save current state
                    states.append(copy.deepcopy(current_board.current_board))
                    board_state = copy.deepcopy(ed.encode_board(current_board))
                    
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
                        # Run MCTS with increased simulations
                        best_move, root = UCT_search(current_board, mcts_simulations, net)
                    finally:
                        # Restore the original deepcopy function
                        copy.deepcopy = original_deepcopy
                    
                    # Apply the selected move
                    current_board = do_decode_n_move_pieces(current_board, best_move)
                    
                    # Get policy from MCTS
                    policy = get_policy(root)
                    dataset.append([board_state, policy])
                    
                    # Status update
                    if current_board.move_count % 10 == 0:
                        print(f"Process {cpu_id}, Game {idxx}, Move {current_board.move_count}")
                    
                    # Check for checkmate
                    if current_board.check_status() == True and current_board.in_check_possible_moves() == []:
                        if current_board.player == 0:  # black wins
                            value = -1
                        elif current_board.player == 1:  # white wins
                            value = 1
                        checkmate = True
                        print(f"Process {cpu_id}, Game {idxx}: Checkmate! Value: {value}")
                
                # Format data for neural network training
                dataset_p = []
                for idx, data in enumerate(dataset):
                    s, p = data
                    if idx == 0:
                        dataset_p.append([s, p, 0])
                    else:
                        dataset_p.append([s, p, value])
                
                # Save the data with iteration info
                iteration_folder = iteration  # Distribute across iteration folders for better organization
                save_as_pickle(f"dataset_cpu{cpu_id}_{idxx}", dataset_p, iteration_folder)
                print(f"Process {cpu_id}, Game {idxx} completed and saved")
                
            except Exception as e:
                print(f"Error in Process {cpu_id}, Game {idxx}: {e}")
                import traceback
                traceback.print_exc()

# Simplified training function
def simple_train(net, datasets, epoch_start=0, epoch_stop=20, batch_size=64):
    print(f"Starting simplified training (no mixed precision)")
    torch.manual_seed(0)
    
    # Move net to CUDA if available
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated, "
              f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB reserved")
    
    net.train()
    alpha_loss = AlphaLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create DataLoader with minimal settings
    train_dataset = ChessDataset(datasets)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No workers to avoid multiprocessing issues
        pin_memory=cuda
    )
    
    print(f"DataLoader created with batch size {batch_size} (no workers)")
    
    # Training loop
    losses_per_epoch = []
    
    for epoch in range(epoch_start, epoch_stop):
        total_loss = 0.0
        batch_count = 0
        start_time = time.time()
        
        print(f"Starting epoch {epoch+1}/{epoch_stop}")
        
        for batch_idx, (states, policies, values) in enumerate(train_loader):
            try:
                if cuda:
                    states = states.cuda()
                    policies = policies.cuda()
                    values = values.cuda()
                
                # Simple training step without mixed precision
                optimizer.zero_grad()
                policy_preds, value_preds = net(states)
                loss = alpha_loss(value_preds.view(-1), values, policy_preds, policies)
                loss.backward()
                optimizer.step()
                
                if cuda:
                    # Explicit sync to detect issues early
                    torch.cuda.synchronize()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Print progress every 5 batches
                if batch_idx % 5 == 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = batch_count * batch_size / elapsed if elapsed > 0 else 0
                    print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, Speed: {samples_per_sec:.1f} samples/sec')
                    
                    if cuda:
                        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue
            
        # Average loss for this epoch
        if batch_count > 0:
            epoch_loss = total_loss / batch_count
            losses_per_epoch.append(epoch_loss)
            print(f'Epoch {epoch+1} completed, Average Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.1f}s')
        else:
            print(f'Epoch {epoch+1} failed, no batches processed')
    
    print(f'Training completed with {len(losses_per_epoch)} epochs')
    return net

# Simplified run_training function
def simple_run_training(net, iteration, epochs=20, past_iterations=5, batch_size=64):
    # Gather data from previous iterations
    all_samples = []
    data_found = False
    
    # Calculate which iterations to include
    start_iter = max(0, iteration - past_iterations)  # Don't go below 0
    end_iter = iteration  # Current iteration
    
    print(f"Loading training data from iterations {start_iter} to {end_iter}")
    
    # Try to load data from the specified iterations
    for iter_num in range(start_iter, end_iter + 1):
        data_path = f"./datasets/iter{iter_num}/"
        if os.path.exists(data_path):
            files = os.listdir(data_path)
            if files:
                print(f"Loading files from iteration {iter_num}")
                
                for file in files:
                    filename = os.path.join(data_path, file)
                    if os.path.isfile(filename):
                        try:
                            with open(filename, 'rb') as fo:
                                dataset = pickle.load(fo, encoding='bytes')
                                if dataset and len(dataset) > 0:
                                    # Very simple validation - just check length
                                    valid_samples = []
                                    for sample in dataset:
                                        if len(sample) == 3:
                                            valid_samples.append(sample)
                                    
                                    if valid_samples:
                                        all_samples.extend(valid_samples)
                                        print(f"Loaded {len(valid_samples)} samples from {filename}")
                                        data_found = True
                        except Exception as e:
                            print(f"Error loading {filename}: {e}")
    
    if not data_found or len(all_samples) == 0:
        print("No valid training data found.")
        return False
    
    print(f"Total training samples: {len(all_samples)}")
    
    # Implement dynamic batch size based on dataset size
    adjusted_batch_size = batch_size
    if len(all_samples) < batch_size * 10:  # If we have less than 10 batches
        adjusted_batch_size = max(16, len(all_samples) // 10)  # Ensure at least 10 batches, minimum size 16
        print(f"Warning: Small dataset detected. Reducing batch size to {adjusted_batch_size}")
    
    # Try training with simple configuration
    try:
        print(f"Starting training with batch size {adjusted_batch_size}")
        # Use the simplified training function
        simple_train(net, all_samples, 0, epochs, batch_size=adjusted_batch_size)
        return True
    except Exception as e:
        print(f"Training failed: {e}")
        # Try with smaller batch size as a last resort
        try:
            print(f"Retrying with reduced batch size {adjusted_batch_size // 2}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            simple_train(net, all_samples, 0, epochs, batch_size=adjusted_batch_size // 2)
            return True
        except Exception as e:
            print(f"Retry training failed: {e}")
            return False

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fast AlphaZero Chess Pipeline')
    parser.add_argument('--iterations', type=int, default=30, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=8, help='Number of games per worker')
    parser.add_argument('--workers', type=int, default=12, help='Number of worker processes')
    parser.add_argument('--mcts_sims', type=int, default=300, help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs per iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--past_iterations', type=int, default=3, help='Number of past iterations to include in training data')
    parser.add_argument('--safest_mode', action='store_true', help='Use the safest, most stable training settings')
    parser.add_argument('--skip_self_play', action='store_true', help='Skip self-play and only run training')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only run self-play')
    args = parser.parse_args()
    
    # Apply safest mode settings if requested
    if args.safest_mode:
        args.batch_size = min(32, args.batch_size)
        args.past_iterations = min(2, args.past_iterations)
        print("Using safest mode settings: reduced batch size and past iterations")
    
    print("Starting Optimized AlphaZero Chess Pipeline")
    
    # Check if CUDA is available
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"Using device: {device}")
    
    if cuda:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Print initial GPU memory info
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated, "
              f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB reserved")
    
    # Calculate optimal number of workers based on CPU count
    num_processes = min(args.workers, os.cpu_count() or 1)
    
    for iteration in range(args.iterations):
        print(f"\n===== Starting iteration {iteration} =====")
        
        # Select model to use
        if iteration == 0:
            net_to_play = "current_net.pth.tar"
        else:
            net_to_play = f"current_net_trained_iter{iteration-1}.pth.tar"
        
        # Save name for the next model
        save_as = f"current_net_trained_iter{iteration}.pth.tar"
        
        # Check if the model file exists
        current_net_filename = os.path.join("./model_data/", net_to_play)
        if not os.path.exists(current_net_filename):
            print(f"Error: Model file {current_net_filename} not found.")
            break
        
        # ===== PHASE 1: MCTS Self-Play =====
        if not args.skip_self_play:
            print(f"Running optimized MCTS self-play with model: {net_to_play}")
            print(f"Using {num_processes} workers, {args.games} games per worker, {args.mcts_sims} MCTS simulations per move")
            
            # Load model
            net = ChessNet()
            if cuda:
                net.cuda()
            net.share_memory()
            checkpoint = torch.load(current_net_filename, map_location="cuda" if cuda else "cpu")
            net.load_state_dict(checkpoint['state_dict'])
            
            # Run self-play processes
            processes = []
            for i in range(num_processes):
                p = mp.Process(target=optimized_MCTS_self_play, args=(net, args.games, i, iteration, args.mcts_sims))
                p.start()
                processes.append(p)
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
                
            # Clean up to free memory
            del net
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
                print(f"GPU memory after self-play: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated, "
                      f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB reserved")
        else:
            print("Skipping self-play phase as requested")
        
        # ===== PHASE 2: Neural Network Training =====
        if not args.skip_training:
            print(f"Training neural network, will save as: {save_as}")
            print(f"Training for {args.epochs} epochs with batch size {args.batch_size}")
            print(f"Using data from the past {args.past_iterations} iterations")
            
            # Load model for training
            train_net = ChessNet()
            if cuda:
                train_net.cuda()
            
            checkpoint = torch.load(current_net_filename, map_location="cuda" if cuda else "cpu")
            train_net.load_state_dict(checkpoint['state_dict'])
            
            # Run training with simplified function
            training_success = simple_run_training(
                train_net, 
                iteration, 
                epochs=args.epochs, 
                past_iterations=args.past_iterations,
                batch_size=args.batch_size
            )
            
            if training_success:
                # Save results
                torch.save({'state_dict': train_net.state_dict()}, 
                          os.path.join("./model_data/", save_as))
                print(f"Training completed and model saved as {save_as}")
            else:
                # Just copy the previous model as a fallback
                import shutil
                shutil.copy2(current_net_filename, os.path.join("./model_data/", save_as))
                print(f"Training failed, copied previous model to {save_as}")
            
            # Clean up to free memory
            del train_net
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
                print(f"GPU memory after training: {torch.cuda.memory_allocated() / 1024**2:.1f} MB allocated, "
                     f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB reserved")
        else:
            print("Skipping training phase as requested")
        
        print(f"Completed iteration {iteration}")
    
    print("Optimized pipeline completed!")