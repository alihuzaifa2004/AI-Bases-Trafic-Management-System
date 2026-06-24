import torch
import torch.nn as nn
import numpy as np
import logging
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, Config.PREDICTION_STEPS)

    def forward(self, x):
        # Forward pass through PyTorch LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Extract the hidden state of the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Map to final output prediction steps
        out = self.fc(last_time_step)
        return out

class PredictorEngine:
    def __init__(self):
        # This will now find the class correctly!
        self.model = TrafficLSTM().to(Config.DEVICE)
        self.history = []
        
        # Load weights with legacy unpickler support for PyTorch 2.6+
        try:
            logger.info("Attempting to load LSTM weights...")
            checkpoint = torch.load(
                str(Config.LSTM_MODEL_PATH), 
                map_location=Config.DEVICE, 
                weights_only=False
            )
            
            # Handle both raw state_dicts and full serialized objects
            if isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            elif hasattr(checkpoint, 'state_dict'):
                self.model.load_state_dict(checkpoint.state_dict())
            else:
                self.model = checkpoint
                
            self.model.eval()
            logger.info(f"Successfully loaded model weights from {Config.LSTM_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"Could not load weights ({e}). Running model with random initialization.")
            self.model.eval()

    def append_history(self, count):
        self.history.append(count)
        if len(self.history) > Config.SEQUENCE_LENGTH:
            self.history.pop(0)

    def predict_next(self):
        if len(self.history) < Config.SEQUENCE_LENGTH:
            # Fallback estimation until time buffer sequences fill
            return float(np.mean(self.history)) if self.history else 0.0
            
        seq = np.array(self.history[-Config.SEQUENCE_LENGTH:], dtype=np.float32).reshape(1, -1, 1)
        tensor_seq = torch.from_numpy(seq).to(Config.DEVICE)
        
        with torch.no_grad():
            prediction_tensor = self.model(tensor_seq)
            predictions = prediction_tensor.cpu().numpy().flatten()
        
        # Clip negative predictions to 0.0
        predictions = np.clip(predictions, 0.0, None)
        
        if len(predictions) == 1:
            return float(predictions[0])
        return predictions.tolist()