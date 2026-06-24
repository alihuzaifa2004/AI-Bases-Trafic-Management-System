import os
import torch

class Config:
    # Hardware & Latency
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LATENCY_LIMIT_SEC = 3.0
    
    # Dataset and YOLO Settings
    DATASET_DIR = "Vehicle_Detection_Image_Dataset"
    YOLO_MODEL_PATH = r"E:\University All Semester Material\Spring 7th semester\AI theory\ITMS-Project\runs\detect\train-2\weights\best.pt"  # Nano variant selected for edge hardware optimization
    CONFIDENCE_THRESHOLD = 0.45
    
    # Class IDs based on standard COCO (or your custom data.yaml map)
    # Emergency classes prioritized implicitly
    EMERGENCY_CLASSES = ["ambulance", "fire truck", "police car"]
    VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"] + EMERGENCY_CLASSES
    
    # LSTM Forecasting Hyperparameters
    SEQUENCE_LENGTH = 12  # Look back 12 steps (e.g., last 1 hour if 5-min intervals)
    PREDICTION_STEPS = 1  # Predict next interval
    INPUT_SIZE = 1        # Total vehicle count
    HIDDEN_SIZE = 64
    LSTM_MODEL_PATH = "models/traffic_lstm.pth"
    
    # Reinforcement Learning Settings
    NUM_LANES = 4
    RL_STATE_SPACE = NUM_LANES * 2  # Count and waiting times per lane
    RL_ACTION_SPACE = NUM_LANES     # Choose which lane gets the Green Phase
    DQN_MODEL_PATH = "models/dqn_signal_controller.pth"
    
    # Signal Durations (Seconds)
    MIN_GREEN_TIME = 15
    MAX_GREEN_TIME = 60
    YELLOW_TIME = 4

os.makedirs("models", exist_ok=True)