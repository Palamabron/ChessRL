"""
Optimized configuration parameters for the chess reinforcement learning model
"""

import os
import multiprocessing
import torch

class Config:
    """Configuration class for the chess RL system with optimized performance settings"""
    
    def __init__(self):
        # Paths
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Neural Network
        self.n_residual_blocks = 3
        self.n_features = 128
        
        # Training
        self.batch_size = 512  # Increased for better GPU utilization
        self.epochs = 10
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        # MCTS
        self.num_simulations = 16  # Increased for better performance
        self.c_puct = 1.0
        self.dirichlet_alpha = 0.3
        self.dirichlet_noise_factor = 0.25
        
        # Self-play
        self.num_self_play_games = 50
        self.temperature = 1.0
        self.temperature_threshold = 15
        
        # Parallelization - optimized for better resource utilization
        self.num_workers = max(1, min(multiprocessing.cpu_count(), 4))
        
        # For CPU-only machines, use fewer workers to avoid thread contention
        if not torch.cuda.is_available():
            self.num_workers = max(1, multiprocessing.cpu_count() // 3)
        
        # Batch processing for MCTS
        self.batch_mcts_size = 32 if torch.cuda.is_available() else 8
        
        # Memory management
        self.replay_buffer_size = 20000
        
        # Performance optimizations
        self.pin_memory = torch.cuda.is_available()  # Faster data transfer to GPU
        self.num_data_workers = min(2, multiprocessing.cpu_count() // 2)
        self.use_mixed_precision = torch.cuda.is_available()  # Use FP16 for speed
        
        # GUI
        self.board_size = 600
        self.piece_theme = 'default'