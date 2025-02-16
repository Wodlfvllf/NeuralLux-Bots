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
        """Step 2: Selects movement actions based on unit positions."""
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(self.env_cfg["max_units"]):
            if unit_mask[unit_id]:
                x, y = unit_positions[unit_id]
                move_direction = np.random.choice([0, 1, 2, 3, 4, 5])  # Random movement selection
                actions[unit_id] = [move_direction, x, y]  # Store action with position info
        
        return actions

