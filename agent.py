import numpy as np

class Agent():
    def __init__(self, player: str, env_cfg):
        """Step 4: Basic Reward Function"""
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        # Tracking previous state information
        self.prev_unit_positions = None
        self.prev_unit_energies = None
        self.prev_map_energy = None

    def compute_reward(self, unit_positions, prev_unit_positions, unit_energies, prev_unit_energies):
        """Computes a simple reward based on movement and energy efficiency."""
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

        return reward

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Step 4: Selects actions and computes rewards."""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energies = np.array(obs["units"]["energy"][self.team_id])
        map_energy = np.array(obs["map_features"]["energy"])
        
        # Initialize previous states on the first step
        if self.prev_unit_positions is None:
            self.prev_unit_positions = unit_positions.copy()
            self.prev_unit_energies = unit_energies.copy()
            self.prev_map_energy = map_energy.copy()
            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)  # No actions on first step
        
        # Compute rewards
        rewards = self.compute_reward(unit_positions, self.prev_unit_positions, unit_energies, self.prev_unit_energies)
        
        # State Representation
        unit_representation = np.concatenate((unit_positions.flatten(), unit_energies.flatten()))
        state_representation = np.concatenate((unit_representation, map_energy.flatten()))
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(self.env_cfg["max_units"]):
            if unit_mask[unit_id]:
                x, y = unit_positions[unit_id]
                move_direction = np.random.choice([0, 1, 2, 3, 4, 5])  # Random movement selection
                actions[unit_id] = [move_direction, x, y]
        
        # Update previous states
        self.prev_unit_positions = unit_positions.copy()
        self.prev_unit_energies = unit_energies.copy()
        self.prev_map_energy = map_energy.copy()
        
        return actions