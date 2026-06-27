#importing libraries
import time
import numpy as np
from traffic_detector import TrafficDetector
from traffic_predictor import PredictorEngine
from signal_controller import DQNSignalController
from config import Config
#pipelines main class
class ITMSPipeline:
    def __init__(self):
        self.detector = TrafficDetector()
        self.predictor = PredictorEngine()
        self.controller = DQNSignalController()
        
        # State tracking arrays for structural simulation (4 intersections/lanes)
        self.lane_vehicle_counts = [0, 0, 0, 0]
        self.lane_waiting_times = [0.0, 0.0, 0.0, 0.0]
        self.current_active_lane = 0

    def execution_step(self, frame_lanes, sumo_counts=None):
        """
        Executes a single processing iteration loop for an active junction network.
        frame_lanes: List of frames captured concurrently from intersection cameras.
        sumo_counts: List of actual vehicle counts fetched directly from SUMO (optional).
        """
        start_time = time.time()
        emergency_override_triggered = False

        # 1. Computer Vision Processing
        for i, frame in enumerate(frame_lanes):
            if frame is None: continue
            metrics = self.detector.process_frame(frame)
            
            # Use SUMO counts if available; fallback to YOLO counts if standalone
            if sumo_counts is not None:
                self.lane_vehicle_counts[i] = sumo_counts[i]
            else:
                self.lane_vehicle_counts[i] = metrics["total_count"]
            
            # Keep safety critical triggers alive via vision array check
            if metrics["emergency_trigger"]:
                emergency_override_triggered = True
                self.current_active_lane = i 

        # 2. Time-series historical accumulation
        aggregated_current_flow = sum(self.lane_vehicle_counts)
        self.predictor.append_history(aggregated_current_flow)
        forecasted_flow = self.predictor.predict_next()

        # 3. Control Decision Optimization Logic
        if emergency_override_triggered:
            allocated_green_time = Config.MAX_GREEN_TIME
            decision_source = "EMERGENCY_OVERRIDE_SYSTEM"
        else:
            # Structure state space vector input matching DQNetwork criteria
            state = np.array(self.lane_vehicle_counts + self.lane_waiting_times, dtype=np.float32)
            self.current_active_lane = self.controller.select_action(state)
            allocated_green_time = self.controller.calculate_adaptive_time(
                self.lane_vehicle_counts[self.current_active_lane]
            )
            decision_source = "RL_DQN_OPTIMIZER"

        # Simulate update to waiting metrics across unselected paths
        for idx in range(Config.NUM_LANES):
            if idx != self.current_active_lane:
                self.lane_waiting_times[idx] += allocated_green_time
            else:
                self.lane_waiting_times[idx] = 0.0  # Reset on green phase clear

        processing_latency = time.time() - start_time
        latency_compliant = processing_latency < Config.LATENCY_LIMIT_SEC

        return {
            "active_lane": self.current_active_lane,
            "green_duration_seconds": allocated_green_time,
            "predicted_next_volume": forecasted_flow,
            "decision_engine": decision_source,
            "processing_latency_ms": processing_latency * 1000,
            "latency_compliant": latency_compliant
        }