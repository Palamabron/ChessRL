#!/usr/bin/env python

import os
import torch
from alpha_net import ChessNet

# Initialize a new network
net = ChessNet()
print("Initialized a new network")

# Create the model_data directory if it doesn't exist
os.makedirs("./model_data/", exist_ok=True)

# Save the initialized network
torch.save(
    {'state_dict': net.state_dict()}, 
    os.path.join("./model_data/", "current_net.pth.tar")
)
print("Saved initial model to ./model_data/current_net.pth.tar")