"""
Configuration parameters for the chess reinforcement learning model
"""

import os

class Config:
    """Configuration class for the chess RL system"""
    
    def __init__(self):
        # Paths
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Neural Network
        self.n_residual_blocks = 10
        self.n_features = 128
        
        # Training
        self.batch_size = 256
        self.epochs = 10
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # MCTS
        self.num_simulations = 100
        self.c_puct = 1.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_noise_factor = 0.25
        
        # Self-play
        self.num_self_play_games = 100
        self.temperature = 1.0
        self.temperature_threshold = 15  # Move number after which temperature is set to ~0
        
        # Memory
        self.replay_buffer_size = 10000
        
        # GUI
        self.board_size = 600
        self.piece_theme = 'default'
