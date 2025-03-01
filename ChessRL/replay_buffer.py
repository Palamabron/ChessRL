"""
Replay buffer and dataset for training the chess model
"""

from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset

class ReplayBuffer:
    """
    Buffer for storing and sampling experience
    """
    def __init__(self, max_size):
        """
        Initialize the replay buffer
        
        Args:
            max_size (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        """
        Add experience to the buffer
        
        Args:
            experience (list): List of experience dictionaries
        """
        self.buffer.extend(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: Sampled experiences
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        """Get the current size of the buffer"""
        return len(self.buffer)


class ChessDataset(Dataset):
    """
    Dataset for training the chess model
    """
    def __init__(self, examples):
        """
        Initialize the dataset
        
        Args:
            examples (list): List of experience dictionaries
        """
        self.examples = examples
    
    def __len__(self):
        """Get the size of the dataset"""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index of the item
            
        Returns:
            tuple: (board, policy, value)
        """
        example = self.examples[idx]
        return example['board'], torch.FloatTensor(example['policy']), torch.FloatTensor([example['value']])
