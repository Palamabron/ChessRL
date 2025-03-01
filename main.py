import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math
import time
import os
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Configuration
class Config:
    # Neural Network
    n_residual_blocks = 10
    n_features = 128
    
    # Training
    batch_size = 256
    epochs = 10
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # MCTS
    num_simulations = 100
    c_puct = 1.0
    dirichlet_alpha = 0.3
    dirichlet_noise_factor = 0.25
    
    # Self-play
    num_self_play_games = 100
    temperature = 1.0
    temperature_threshold = 15  # Move number after which temperature is set to ~0
    
    # Memory
    replay_buffer_size = 10000
    
    # Save/Load
    model_dir = 'models'


# Encode chess board state into input features for the neural network
def encode_board(board):
    # 12 pieces (6 white, 6 black) + additional features
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Piece placement and types
    piece_idx = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
                 "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = chess.square_rank(square), chess.square_file(square)
            planes[piece_idx[piece.symbol()]][rank][file] = 1
    
    # Return as a PyTorch tensor
    return torch.FloatTensor(planes)

# Neural Network Architecture (similar to AlphaZero)
class ChessNetwork(nn.Module):
    def __init__(self, config):
        super(ChessNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(12, config.n_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.n_features)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.n_features) for _ in range(config.n_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(config.n_features, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)  # 4672 possible moves (64x73 - invalid)
        
        # Value head
        self.value_conv = nn.Conv2d(config.n_features, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        # Common layers
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_features)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_features)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

# Move encoding/decoding
def move_to_index(move):
    """Convert a chess.Move to an index for the policy vector."""
    # This is a simplified mapping - real implementation needs to handle all legal moves
    from_square = move.from_square
    to_square = move.to_square
    
    # Handle promotions
    if move.promotion:
        # For simplicity, we'll map promotions to specific indices
        # In a full implementation, you would need a more sophisticated mapping
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }
        return 64*64 + from_square*64 + to_square + promotion_offset[move.promotion]
    
    return from_square*64 + to_square

def index_to_move(index, board):
    """Convert an index from the policy vector to a chess.Move."""
    # If it's a promotion move
    if index >= 64*64:
        prom_index = index - 64*64
        from_square = prom_index // 64 // 4
        to_square = (prom_index // 4) % 64
        promotion_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][prom_index % 4]
        return chess.Move(from_square, to_square, promotion_piece)
    
    # Regular move
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

# MCTS Node
class MCTSNode:
    def __init__(self, prior=0, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.board = None
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

# Monte Carlo Tree Search
class MCTS:
    def __init__(self, network, config):
        self.network = network
        self.config = config
    
    def search(self, board, is_training=True):
        root = MCTSNode()
        root.board = board.copy()
        
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
            value = self._expand_node(node, scratch_board)
            
            # Backpropagation phase
            self._backpropagate(search_path, value, scratch_board.turn)
        
        return root
    
    def _select_child(self, node):
        """Select the child with the highest UCB score."""
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
        """Expand the node and return the value."""
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
        """Backpropagate the value through the search path."""
        for node in search_path:
            node.visit_count += 1
            # Value is from the perspective of the current player
            if node.board.turn == player:
                node.value_sum += value
            else:
                node.value_sum -= value  # Negate value for the opponent
    
    def _add_dirichlet_noise(self, node):
        """Add Dirichlet noise to the prior probabilities in the root node."""
        if not node.expanded() or node.board is None:
            return
        
        # Generate noise
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(node.children))
        
        # Add noise to each child's prior probability
        for i, (action, child) in enumerate(node.children.items()):
            child.prior = (1 - self.config.dirichlet_noise_factor) * child.prior + self.config.dirichlet_noise_factor * noise[i]

# Self-play for data generation
class SelfPlay:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.mcts = MCTS(network, config)
    
    def execute_episode(self):
        """Play a full game of self-play and return game data for training."""
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
            
            if temp == 0:
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

# Training Data Management
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.extend(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

class ChessDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return example['board'], torch.FloatTensor(example['policy']), torch.FloatTensor([example['value']])

# Training
class Trainer:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
    
    def train(self, examples):
        """Update the network using examples from self-play."""
        # Add examples to replay buffer
        self.replay_buffer.add(examples)
        
        if len(self.replay_buffer) < self.config.batch_size:
            print(f"Not enough examples for training. Current: {len(self.replay_buffer)}, Required: {self.config.batch_size}")
            return
        
        # Sample batches and train
        for _ in range(self.config.epochs):
            batches = self._create_batches()
            policy_losses = []
            value_losses = []
            total_losses = []
            
            for batch in batches:
                states, target_policies, target_values = batch
                
                # Forward pass
                out_policies, out_values = self.network(states)
                
                # Calculate loss
                policy_loss = F.cross_entropy(out_policies, target_policies)
                value_loss = F.mse_loss(out_values, target_values)
                total_loss = policy_loss + value_loss
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Keep track of losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(total_loss.item())
            
            # Log training progress
            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_total_loss = np.mean(total_losses)
            
            print(f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
    
    def _create_batches(self):
        """Create batches for training."""
        samples = self.replay_buffer.sample(self.config.batch_size)
        dataset = ChessDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        return dataloader
    
    def save_model(self, filename):
        """Save the model to disk."""
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.model_dir, filename))
    
    def load_model(self, filename):
        """Load the model from disk."""
        checkpoint = torch.load(os.path.join(self.config.model_dir, filename))
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Main training loop
def main():
    config = Config()
    network = ChessNetwork(config)
    trainer = Trainer(network, config)
    self_play = SelfPlay(network, config)
    
    print("Starting training...")
    
    for iteration in range(1, 51):  # 50 training iterations
        print(f"\nTraining Iteration {iteration}")
        
        # Self-play to generate data
        print(f"Executing {config.num_self_play_games} self-play games...")
        game_examples = []
        
        for i in range(config.num_self_play_games):
            if i % 10 == 0:
                print(f"Self-play game {i}/{config.num_self_play_games}")
            game_data = self_play.execute_episode()
            game_examples.extend(game_data)
        
        print(f"Generated {len(game_examples)} examples from self-play.")
        
        # Train network with examples
        print("Training network with generated examples...")
        trainer.train(game_examples)
        
        # Save the model periodically
        if iteration % 5 == 0:
            trainer.save_model(f"chess_model_iter_{iteration}.pt")
    
    # Save final model
    trainer.save_model("chess_model_final.pt")
    print("Training complete!")

if __name__ == "__main__":
    main()
