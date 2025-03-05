"""
Optimized neural network architecture for the chess reinforcement learning model
with better GPU utilization and memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Optimized residual block for the chess network
    """
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_features)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_features)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)  # Inplace ReLU for memory efficiency
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out, inplace=True)
        return out

class ChessNetwork(nn.Module):
    """
    Optimized Neural Network Architecture
    """
    def __init__(self, config):
        super(ChessNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(12, config.n_features, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(config.n_features)
        
        # Residual blocks in a sequential container for performance
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(config.n_features) for _ in range(config.n_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(config.n_features, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4672)  # 4672 possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(config.n_features, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        # JIT compilation for faster inference (if not training)
        self.is_compiled = False
    
    def _initialize_weights(self):
        """Initialize weights with optimized method"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def compile_for_inference(self):
        """Use TorchScript to compile the model for faster inference"""
        if not self.is_compiled:
            try:
                # Only compile when in eval mode
                is_training = self.training
                self.eval()
                
                # Create example input
                example_input = torch.zeros(1, 12, 8, 8, device=next(self.parameters()).device)
                
                # Trace the model
                self.forward = torch.jit.trace(self.forward, example_input)
                self.is_compiled = True
                
                # Restore training mode if it was on
                if is_training:
                    self.train()
                
                print("Model successfully compiled for faster inference")
            except Exception as e:
                print(f"Could not compile model: {e}")
    
    def forward(self, x):
        # Common layers with memory-efficient operations
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        
        # Residual tower (sequential is more efficient)
        x = self.res_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        policy = policy.reshape(-1, 2 * 8 * 8)  # reshape is faster than view for non-contiguous tensors
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        value = value.reshape(-1, 8 * 8)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value