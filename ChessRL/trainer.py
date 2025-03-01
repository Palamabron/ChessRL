"""
Trainer for the chess reinforcement learning model
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from chess_rl.replay_buffer import ReplayBuffer, ChessDataset

class Trainer:
    """
    Trainer for the chess reinforcement learning model
    """
    def __init__(self, network, config):
        """
        Initialize the trainer
        
        Args:
            network (ChessNetwork): The neural network to train
            config (Config): Configuration parameters
        """
        self.network = network
        self.config = config
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
    
    def train(self, examples):
        """
        Update the network using examples from self-play
        
        Args:
            examples (list): List of experience dictionaries from self-play
        """
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
        """
        Create batches for training
        
        Returns:
            DataLoader: Batches of training data
        """
        samples = self.replay_buffer.sample(self.config.batch_size)
        dataset = ChessDataset(samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        return dataloader
    
    def save_model(self, filename):
        """
        Save the model to disk
        
        Args:
            filename (str): Filename to save the model to
        """
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.model_dir, filename))
    
    def load_model(self, filename):
        """
        Load the model from disk
        
        Args:
            filename (str): Filename to load the model from
        """
        checkpoint = torch.load(os.path.join(self.config.model_dir, filename))
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
