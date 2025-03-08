#!/usr/bin/env python

import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from alpha_net import ChessNet, train
import datetime
import time

# Modified MCTS functions to run on CPU
def run_mcts_cpu(net, num_games, cpu_id):
    from chess_board import board as c_board
    import encoder_decoder as ed
    import copy
    
    # Import required functions but keep them local to avoid conflicts
    def UCT_search(game_state, num_reads, net):
        from MCTS_chess import UCTNode, DummyNode
        import math
        
        root = UCTNode(game_state, move=None, parent=DummyNode())
        for i in range(num_reads):
            leaf = root.select_leaf()
            encoded_s = ed.encode_board(leaf.game)
            encoded_s = encoded_s.transpose(2,0,1)
            # Use CPU tensor
            encoded_s = torch.from_numpy(encoded_s).float()
            child_priors, value_estimate = net(encoded_s)
            child_priors = child_priors.detach().numpy().reshape(-1)
            value_estimate = value_estimate.item()
            if leaf.game.check_status() == True and leaf.game.in_check_possible_moves() == []:
                leaf.backup(value_estimate)
                continue
            leaf.expand(child_priors)
            leaf.backup(value_estimate)
        return np.argmax(root.child_number_visits), root
    
    def do_decode_n_move_pieces(board, move):
        i_pos, f_pos, prom = ed.decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.move_piece(i, f, p)
            a, b = i
            c, d = f
            if board.current_board[c, d] in ["K", "k"] and abs(d-b) == 2:
                if a == 7 and d-b > 0:
                    board.player = 0
                    board.move_piece((7, 7), (7, 5), None)
                if a == 7 and d-b < 0:
                    board.player = 0
                    board.move_piece((7, 0), (7, 3), None)
                if a == 0 and d-b > 0:
                    board.player = 1
                    board.move_piece((0, 7), (0, 5), None)
                if a == 0 and d-b < 0:
                    board.player = 1
                    board.move_piece((0, 0), (0, 3), None)
        return board
    
    def get_policy(root):
        policy = np.zeros([4672], dtype=np.float32)
        for idx in np.where(root.child_number_visits != 0)[0]:
            policy[idx] = root.child_number_visits[idx] / root.child_number_visits.sum()
        return policy
    
    def save_as_pickle(filename, data):
        os.makedirs("./datasets/iter0/", exist_ok=True)
        completeName = os.path.join("./datasets/iter0/", filename)
        with open(completeName, 'wb') as output:
            pickle.dump(data, output)
    
    # Main self-play loop
    for idxx in range(num_games):
        current_board = c_board()
        checkmate = False
        dataset = []
        states = []
        value = 0
        
        try:
            print(f"CPU {cpu_id}, starting game {idxx}")
            while checkmate == False and current_board.move_count <= 50:  # Reduced max moves for testing
                draw_counter = 0
                for s in states:
                    if np.array_equal(current_board.current_board, s):
                        draw_counter += 1
                if draw_counter == 3:  # draw by repetition
                    break
                
                states.append(copy.deepcopy(current_board.current_board))
                board_state = copy.deepcopy(ed.encode_board(current_board))
                
                # Reduced MCTS iterations for CPU
                best_move, root = UCT_search(current_board, 100, net)  
                current_board = do_decode_n_move_pieces(current_board, best_move)
                policy = get_policy(root)
                dataset.append([board_state, policy])
                
                if current_board.move_count % 5 == 0:  # Print progress less frequently
                    print(f"CPU {cpu_id}, Game {idxx}, Move {current_board.move_count}")
                
                if current_board.check_status() == True and current_board.in_check_possible_moves() == []:
                    if current_board.player == 0:  # black wins
                        value = -1
                    elif current_board.player == 1:  # white wins
                        value = 1
                    checkmate = True
            
            # Create dataset
            dataset_p = []
            for idx, data in enumerate(dataset):
                s, p = data
                if idx == 0:
                    dataset_p.append([s, p, 0])
                else:
                    dataset_p.append([s, p, value])
            
            # Save dataset
            save_as_pickle(f"dataset_cpu{cpu_id}_{idxx}", dataset_p)
            print(f"CPU {cpu_id}, Game {idxx} completed")
            
        except Exception as e:
            print(f"Error in CPU {cpu_id}, Game {idxx}: {e}")


if __name__ == "__main__":
    print("Starting CPU-only AlphaZero pipeline")
    
    # Force CPU mode
    device = "cpu"
    print(f"Using device: {device}")
    
    # Ensure multiprocessing works correctly
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create directories
    os.makedirs("./model_data/", exist_ok=True)
    os.makedirs("./datasets/iter0/", exist_ok=True)
    
    # Check for initial model
    model_path = "./model_data/current_net.pth.tar"
    if not os.path.exists(model_path):
        print("Initial model not found. Run init_model.py first.")
        exit(1)
    
    # Load model
    net = ChessNet()
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()  # Set to evaluation mode
    
    # Determine number of processes
    num_processes = min(4, os.cpu_count() or 1)  # Limit to 4 processes for CPU
    games_per_process = 1  # Start small
    
    print(f"Starting self-play with {num_processes} processes, {games_per_process} games per process")
    print("This may take a while on CPU...")
    
    # Run self-play processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=run_mcts_cpu, args=(net, games_per_process, i))
        p.start()
        processes.append(p)
    
    # Wait for processes to finish
    for p in processes:
        p.join()
    
    print("Self-play completed. Gathering training data...")
    
    # Collect data for training
    datasets = []
    data_path = "./datasets/iter0/"
    for file in os.listdir(data_path):
        try:
            filename = os.path.join(data_path, file)
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                datasets.extend(data)
                print(f"Loaded {len(data)} samples from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not datasets:
        print("No training data found. Exiting.")
        exit(1)
    
    # Convert to numpy array
    datasets = np.array(datasets)
    print(f"Total training samples: {len(datasets)}")
    
    # Train the network
    print("Starting training...")
    train_net = ChessNet()
    checkpoint = torch.load(model_path, map_location=device)
    train_net.load_state_dict(checkpoint['state_dict'])
    train_net.train()  # Set to training mode
    
    # Run training with reduced epochs for CPU
    train(train_net, datasets, 0, 5)
    
    # Save trained model
    save_path = "./model_data/current_net_trained_cpu.pth.tar"
    torch.save({'state_dict': train_net.state_dict()}, save_path)
    print(f"Training completed. Model saved to {save_path}")