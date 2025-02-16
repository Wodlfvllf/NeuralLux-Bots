import numpy as np
from DQN import MultiAgentDQN

class Agent():
    def __init__(self, player: str, env_cfg):
        """Step 7: Store experiences in the replay buffer"""
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        # Initialize Multi-Agent DQN
        n_agents = self.env_cfg["max_units"]  # Number of agents
        input_dims = (self.env_cfg["max_units"] * 3) + 12  # Input size
        n_actions = 6  # Number of actions

        self.multi_agent_dqn = MultiAgentDQN(n_agents=n_agents, input_dims=(input_dims,), n_actions=n_actions)
        
        # Tracking previous state information
        self.prev_unit_positions = None
        self.prev_unit_energies = None
        self.prev_map_energy = None
        self.prev_relic_positions = None
        self.prev_actions = None

    def compute_reward(self, unit_positions, prev_unit_positions, unit_energies, prev_unit_energies, relic_positions):
        """Computes a reward based on movement, energy efficiency, and relic exploration."""
        reward = np.zeros(self.env_cfg["max_units"])  # Initialize rewards for all units

        for unit_id in range(self.env_cfg["max_units"]):
            curr_pos = unit_positions[unit_id]
            prev_pos = prev_unit_positions[unit_id]
            curr_energy = unit_energies[unit_id]
            prev_energy = prev_unit_energies[unit_id]

            # Movement Reward: Encourage movement
            if np.array_equal(curr_pos, prev_pos):
                reward[unit_id] -= 1  # Penalize staying in the same position
            else:
                reward[unit_id] += 2  # Reward movement
            
            # Energy Management: Encourage efficient energy use
            if curr_energy < prev_energy:
                reward[unit_id] -= 1  # Penalize unnecessary energy loss
            
            # Relic Exploration Reward
            if len(relic_positions) > 0:
                closest_relic = min(relic_positions, key=lambda r: np.linalg.norm(r - curr_pos))
                distance = np.linalg.norm(closest_relic - curr_pos)
                reward[unit_id] += max(0, 10 - distance)  # Closer to relic = higher reward

        return reward

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Step 7: Selects actions and stores experiences in replay buffer."""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energies = np.array(obs["units"]["energy"][self.team_id])
        map_energy = np.array(obs["map_features"]["energy"])
        relic_positions = np.array(obs["relic_nodes"])
        
        # Initialize previous states on the first step
        if self.prev_unit_positions is None:
            self.prev_unit_positions = unit_positions.copy()
            self.prev_unit_energies = unit_energies.copy()
            self.prev_map_energy = map_energy.copy()
            self.prev_relic_positions = relic_positions.copy()
            self.prev_actions = np.zeros(self.env_cfg["max_units"], dtype=int)  # Initialize actions
            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)  # No actions on first step
        
        # Compute rewards
        rewards = self.compute_reward(unit_positions, self.prev_unit_positions, unit_energies, self.prev_unit_energies, relic_positions)
        
        # State Representation
        unit_representation = np.concatenate((unit_positions.flatten(), unit_energies.flatten()))
        state_representation = np.concatenate((unit_representation, map_energy.flatten(), relic_positions.flatten()))
        
        # Select actions using Multi-Agent DQN
        actions, x_coords, y_coords = self.multi_agent_dqn.choose_actions([state_representation] * self.env_cfg["max_units"])
        
        # Store experience in replay buffer
        available_unit_ids = np.where(unit_mask)[0]
        num_available_units = len(available_unit_ids)
        self.multi_agent_dqn.store_experience(
            [state_representation] * num_available_units,  # Current states
            [state_representation] * num_available_units,  # Next states
            list(self.prev_actions[:num_available_units]),  # Previous actions
            list(rewards[:num_available_units]),  # Rewards
            [False] * num_available_units,  # Done flags
            num_available_units
        )
        
        formatted_actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, unit_id in enumerate(np.where(unit_mask)[0]):
            formatted_actions[unit_id] = [actions[i], x_coords[i], y_coords[i]]
        
        # Update previous states
        self.prev_unit_positions = unit_positions.copy()
        self.prev_unit_energies = unit_energies.copy()
        self.prev_map_energy = map_energy.copy()
        self.prev_relic_positions = relic_positions.copy()
        self.prev_actions = actions.copy()
        
        return formatted_actions