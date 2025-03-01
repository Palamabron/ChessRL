"""
Monte Carlo Tree Search implementation for the chess reinforcement learning model
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import chess

from chess_rl.encoding import encode_board, move_to_index

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search
    """
    def __init__(self, prior=0, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.board = None
    
    def expanded(self):
        """Check if the node has been expanded"""
        return len(self.children) > 0
    
    def value(self):
        """Get the average value of the node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """
    Monte Carlo Tree Search algorithm
    """
    def __init__(self, network, config):
        """
        Initialize the MCTS algorithm
        
        Args:
            network (ChessNetwork): The neural network for position evaluation
            config (Config): Configuration parameters
        """
        self.network = network
        self.config = config
    
    def search(self, board, is_training=True):
        """
        Perform a Monte Carlo Tree Search from the given board position
        
        Args:
            board (chess.Board): The starting board position
            is_training (bool): Whether this is during training (adds noise for exploration)
            
        Returns:
            MCTSNode: The root node of the search tree
        """
        root = MCTSNode()
        root.board = board.copy()
        
        # Expand the root node immediately
        self._expand_node(root, board)
        
        if is_training:
            # Add Dirichlet noise to root node for exploration in training
            self._add_dirichlet_noise(root)
        
        for _ in range(self.config.num_simulations):
            node = root
            scratch_board = board.copy()
            search_path = [node]
            
            # Selection phase - Select path through tree
            while node.expanded() and not scratch_board.is_game_over():
                action, node = self._select_child(node)
                scratch_board.push(action)
                search_path.append(node)
            
            # Expansion and evaluation phase
            value = 0
            if not scratch_board.is_game_over() and not node.expanded():
                value = self._expand_node(node, scratch_board)
            else:
                # Game result: 1 for win, 0 for draw, -1 for loss
                result = scratch_board.result()
                if result == "1-0":
                    value = 1 if scratch_board.turn == chess.BLACK else -1
                elif result == "0-1":
                    value = 1 if scratch_board.turn == chess.WHITE else -1
                else:  # Draw
                    value = 0
            
            # Backpropagation phase
            self._backpropagate(search_path, value, scratch_board.turn)
        
        return root
    
    def _select_child(self, node):
        """
        Select the child with the highest UCB score
        
        Args:
            node (MCTSNode): The parent node
            
        Returns:
            tuple: (selected action, selected child node)
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Exploration parameter
        c_puct = self.config.c_puct
        
        # Total visit count for parent
        parent_visits = node.visit_count
        
        for action, child in node.children.items():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(parent_visits) / (1 + child_visits)
            ucb_score = child.value() + c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand_node(self, node, board):
        """
        Expand the node and return the value
        
        Args:
            node (MCTSNode): The node to expand
            board (chess.Board): The current board state
            
        Returns:
            float: The value estimate from the neural network
        """
        if board.is_game_over():
            # Game result: 1 for win, 0 for draw, -1 for loss
            result = board.result()
            if result == "1-0":
                return 1 if board.turn == chess.WHITE else -1
            elif result == "0-1":
                return 1 if board.turn == chess.BLACK else -1
            else:  # Draw
                return 0
        
        # Predict policy and value using the neural network
        encoded_board = encode_board(board)
        encoded_board = encoded_board.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            policy_logits, value_pred = self.network(encoded_board)
        
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value_pred.item()
        
        # Create children for all legal moves
        for move in board.legal_moves:
            move_idx = move_to_index(move)
            child = MCTSNode(prior=policy[move_idx], parent=node)
            child.board = board.copy()
            child.board.push(move)
            node.children[move] = child
        
        return value
    
    def _backpropagate(self, search_path, value, player):
        """
        Backpropagate the value through the search path
        
        Args:
            search_path (list): List of nodes in the search path
            value (float): The value to backpropagate
            player (bool): The player who made the last move
        """
        for node in search_path:
            node.visit_count += 1
            # Value is from the perspective of the current player
            if node.board.turn == player:
                node.value_sum += value
            else:
                node.value_sum -= value  # Negate value for the opponent
    
    def _add_dirichlet_noise(self, node):
        """
        Add Dirichlet noise to the prior probabilities in the root node
        
        Args:
            node (MCTSNode): The root node
        """
        if not node.expanded():
            return
        
        # Generate noise
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(node.children))
        
        # Add noise to each child's prior probability
        for i, (action, child) in enumerate(node.children.items()):
            child.prior = (1 - self.config.dirichlet_noise_factor) * child.prior + self.config.dirichlet_noise_factor * noise[i]
