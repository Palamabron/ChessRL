"""
Neural network architecture for the chess reinforcement learning model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block for the chess network
    """
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

class ChessNetwork(nn.Module):
    """
    Neural Network Architecture (similar to AlphaZero)
    """
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
