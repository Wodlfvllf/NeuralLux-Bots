import numpy as np

class Agent():
    def __init__(self, player: str, env_cfg):
        """Step 1: Basic Agent Initialization"""
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Step 1: Randomly selects actions for each unit."""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        num_units = np.sum(unit_mask)  # Count active units
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(self.env_cfg["max_units"]):
            if unit_mask[unit_id]:
                actions[unit_id] = [np.random.randint(0, 6), 0, 0]  # Random movement action
        
        return actions
