from ultralytics import YOLO
from config import Config

# Load the ultra-lightweight pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt") 

if __name__ == "__main__":
    # Train the model optimized for CPU execution
    model.train(
        data=f"{Config.DATASET_DIR}/data.yaml", 
        epochs=3,          # Reduced from 50 to 3 for fast execution on CPU
        imgsz=320,         # Reduced from 640 to 320 to scale down computation by 4x
        device='cpu',      # Explicitly forces CPU usage, bypassing CUDA checks
        workers=0,         # '0' prevents multi-threading overhead/errors on Windows CPUs
        batch=4            # Small batch size to avoid running out of system RAM
    )