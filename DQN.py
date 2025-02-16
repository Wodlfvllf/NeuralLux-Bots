import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os 

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr=0.001):
        super(DeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)  # Single output layer for Q-values

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for all actions
