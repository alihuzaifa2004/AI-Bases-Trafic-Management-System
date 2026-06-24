import torch
import torch.nn as nn
import random
from collections import deque
from config import Config

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNSignalController:
    def __init__(self):
        self.state_dim = Config.RL_STATE_SPACE
        self.action_dim = Config.RL_ACTION_SPACE
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.1  # Low exploration for live production test simulation
        
        self.model = DQNetwork(self.state_dim, self.action_dim).to(Config.DEVICE)
        try:
            self.model.load_state_dict(torch.load(Config.DQN_MODEL_PATH, map_location=Config.DEVICE))
            self.model.eval()
        except Exception:
            pass

    def select_action(self, state):
        """Chooses the next optimal lane phase to assign green time."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).view(1, -1).to(Config.DEVICE)
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(torch.argmax(q_values).item())

    def calculate_adaptive_time(self, density_score):
        """Translates current vehicle counts into concrete signal delay phases."""
        base_time = Config.MIN_GREEN_TIME
        added_time = int(density_score * 2.5)  # Scale green time based on count
        return min(Config.MAX_GREEN_TIME, base_time + added_time)