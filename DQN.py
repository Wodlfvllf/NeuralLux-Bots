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


class MultiAgentDQN:
    def __init__(self, n_agents, input_dims, n_actions, lr=0.001, gamma=0.99, tau=0.005, target_update_freq=5):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau  # Soft update factor
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.policy_agents = [DeepQNetwork(input_dims, n_actions, lr) for _ in range(n_agents)]
        self.target_agents = [DeepQNetwork(input_dims, n_actions, lr) for _ in range(n_agents)]

        # Sync target networks initially
        for target_agent, policy_agent in zip(self.target_agents, self.policy_agents):
            target_agent.load_state_dict(policy_agent.state_dict())

    def _update_target_network(self):
        """Polyak averaging to update target network"""
        for target_agent, policy_agent in zip(self.target_agents, self.policy_agents):
            for target_param, policy_param in zip(target_agent.parameters(), policy_agent.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def train_agents(self):
        if self.replay_buffer.mem_ctr < self.batch_size:
            return

        for policy_agent, target_agent in zip(self.policy_agents, self.target_agents):
            state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size)

            q_values = policy_agent(state_batch)
            q_val_curr = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            q_values_next = target_agent(next_state_batch).max(1)[0].detach()  # Use target network
            q_target = reward_batch + self.gamma * q_values_next
            loss = nn.MSELoss()(q_target, q_val_curr)

            policy_agent.optimizer.zero_grad()
            loss.backward()
            policy_agent.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()
