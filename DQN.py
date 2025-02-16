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

        # Separate layers for action Q-values and coordinate predictions
        self.q_values = nn.Linear(64, n_actions)
        self.x_coordinate = nn.Linear(64, 24)  # Predict X-coordinate
        self.y_coordinate = nn.Linear(64, 24)  # Predict Y-coordinate

        self.softmax = nn.Softmax(dim=-1)  # Softmax for coordinate selection
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        q_values = self.q_values(x)  # Discrete action Q-values
        x_coordinates = self.softmax(self.x_coordinate(x))  # Normalized X selection
        y_coordinates = self.softmax(self.y_coordinate(x))  # Normalized Y selection

        return q_values, x_coordinates, y_coordinates

