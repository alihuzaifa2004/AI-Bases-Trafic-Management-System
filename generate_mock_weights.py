import torch
import os
from traffic_predictor import TrafficLSTM

# Instantiate a fresh model with random weights
model = TrafficLSTM()

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save its clean state dictionary to the correct path
torch.save(model.state_dict(), 'models/traffic_lstm.pth')
print("Successfully generated a clean mock 'models/traffic_lstm.pth' file!")