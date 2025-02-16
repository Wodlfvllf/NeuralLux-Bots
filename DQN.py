import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os 

class SharedReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(max_size, dtype=np.int32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool)

    def store_transition(self, state, next_state, action, reward, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (
            self.state_memory[batch],
            self.next_state_memory[batch],
            self.action_memory[batch],
            self.reward_memory[batch],
            self.terminal_memory[batch],
        )


class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, lr=0.001):
        super(DeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 64)

        # Q-values for 5 discrete actions (0-4 move, 5=sap)
        self.q_values = nn.Linear(64, n_actions)

        # Output for (x, y) coordinates (only for sap action)
        self.x_coordinate = nn.Linear(64, 4)  # Predict X-coordinate in 4x4 region
        self.y_coordinate = nn.Linear(64, 4)  # Predict Y-coordinate in 4x4 region

        self.softmax = nn.Softmax(dim=-1)  # Softmax for coordinate selection
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        q_values = self.q_values(x)  # Get Q-values for all actions
        x_coordinates = self.softmax(self.x_coordinate(x))  # Normalize X output
        y_coordinates = self.softmax(self.y_coordinate(x))  # Normalize Y output

        return q_values, x_coordinates, y_coordinates


