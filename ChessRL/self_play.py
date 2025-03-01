"""
Self-play implementation for the chess reinforcement learning model
"""

import numpy as np
import chess

from chess_rl.mcts import MCTS
from chess_rl.encoding import encode_board, move_to_index

class SelfPlay:
    """
    Self-play for generating training data
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
        self.mcts = MCTS(network, config)
    
    def execute_episode(self):
        """
        Play a full game of self-play and return game data for training
        
        Returns:
            list: Game history with states, policies, and values
        """
        board = chess.Board()
        game_history = []
        
        while not board.is_game_over():
            # Temperature parameter for move selection
            temp = self.config.temperature
            if len(game_history) >= self.config.temperature_threshold:
                temp = 0.1  # Almost deterministic selection
            
            # Run MCTS
            root = self.mcts.search(board)
            
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
                visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
                action_idx = np.random.choice(len(actions), p=visit_count_distribution)
                action = actions[action_idx]
            
            # Store the current state, MCTS policy, and turn
            encoded_board = encode_board(board)
            policy = np.zeros(4672)  # Size of policy vector
            for i, a in enumerate(actions):
                policy[move_to_index(a)] = visit_counts[i]
            policy = policy / np.sum(policy)  # Normalize
            
            game_history.append({
                'board': encoded_board,
                'policy': policy,
                'turn': board.turn
            })
            
            # Execute the chosen move
            board.push(action)
        
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
