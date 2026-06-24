import os
import torch
import sys

# 1. PERMANENT ENVIRONMENTAL FIXES
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
torch.classes.__path__ = []

import streamlit as st
import cv2
import numpy as np
import time
from pipeline_manager import ITMSPipeline
from config import Config

# Try loading TraCI for SUMO synchronization
try:
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    SUMO_AVAILABLE = False

# 2. CONFIGURE ENGINE PRESENTATION PAGE
st.set_page_config(layout="wide", page_title="AI-Enabled ITMS Console")
st.title("🎛️ Intelligent Traffic Management System (ITMS) Edge Control Dashboard")

# Singleton instantiation initialization to keep model weights cached in memory
@st.cache_resource
def load_pipeline():
    return ITMSPipeline()

pipeline = load_pipeline()

# 3. SIDEBAR CONTROLS & SUMO MANAGEMENT
st.sidebar.header("🕹️ Simulation & Core Control Panel")
sumo_mode = st.sidebar.checkbox("🔌 Enable SUMO TraCI Synchronization", value=False, 
                                 disabled=not SUMO_AVAILABLE,
                                 help="Connects the dashboard telemetry to a running SUMO environment.")

if not SUMO_AVAILABLE:
    st.sidebar.warning("⚠️ SUMO_HOME environment variable or 'traci' module not detected. Running in camera standalone mode.")

run_system = st.sidebar.toggle("🚀 Activate Live ITMS Pipeline Loop", value=False)

# 4. STRUCTURAL LAYOUT SETUP
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🌐 Live Intersection Camera Monitor Array")
    lane_grid = st.columns(2)
    viewports = []
    for index in range(Config.NUM_LANES):
        with lane_grid[index % 2]:
            viewports.append(st.empty())

with col2:
    st.subheader("📊 System Telemetry & Automation Execution Outputs")
    st.markdown("### ⚠️ Manual Dispatch Operations Overrides")
    manual_override = st.button("🚨 TRIGGER EMERGENCY CENTRAL LOCK (PHASE 1 GREEN)")
    
    # Set up real-time metric value placeholders
    metric_lane = st.empty()
    metric_duration = st.empty()
    metric_volume = st.empty()
    metric_latency = st.empty()
    metric_engine = st.empty()
    
    st.subheader("📈 Current Density Profile per Direction")
    chart_placeholder = st.empty()

# 5. LIVE RUNTIME ORCHESTRATION LOOP
video_path = r"Vehicle_Detection_Image_Dataset/sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# This loop continues running seamlessly if the sidebar toggle switch is flipped 'On'
while run_system:
    ret, real_frame = cap.read()
    
    # Loop back to the beginning if the video file hits the final frame segment
    if not ret or real_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, real_frame = cap.read()
        
    if not ret or real_frame is None:
        real_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(real_frame, "STREAM RECOVERY FAULT", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    real_frame = cv2.resize(real_frame, (320, 240))
    
    # --- SUMO TRANSLATION REGISTRY STEP ---
    sumo_counts = None
    if sumo_mode and SUMO_AVAILABLE:
        try:
            # Check if we can step or initialize connection cleanly on port 8813
            try:
                traci.getConnection("default")
            except traci.exceptions.FatalTraCIError:
                traci.init(port=8813)
                
            traci.simulationStep()
            
            # Map structural edge junction labels setup in your SUMO network files
            sumo_lanes = ["edge_N_0", "edge_E_0", "edge_S_0", "edge_W_0"]
            sumo_counts = []
            for lane_id in sumo_lanes:
                try:
                    sumo_counts.append(traci.lane.getLastStepVehicleNumber(lane_id))
                except traci.exceptions.TraCIException:
                    sumo_counts.append(0)
        except Exception:
            st.sidebar.error("❌ TraCI Connection Lost. Ensure python sumo_simulation.py is active.")
            sumo_mode = False

    # Run video matrices through our trained YOLOv8 model architecture
    processed_data = pipeline.detector.process_frame(real_frame)
    annotated_feed = processed_data["annotated_frame"]
    
    # Compute cross-model execution variables passing SUMO counts if connected
    results = pipeline.execution_step([real_frame] * Config.NUM_LANES, sumo_counts=sumo_counts)
    
    # If SUMO is connected, send our calculated optimal traffic light changes back into the simulator
    if sumo_mode and SUMO_AVAILABLE:
        try:
            phase_string_map = {0: "GGggrrrrGGggrrrr", 1: "rrrrGGggrrrrGGgg"}
            target_phase = phase_string_map.get(results["active_lane"], "rrrrrrrrrrrrrrrr")
            traci.trafficlight.setRedYellowGreenState("junction_center", target_phase)
        except Exception:
            pass

    # Administrative overrides intercept parameters
    if manual_override:
        results["active_lane"] = 0
        results["green_duration_seconds"] = Config.MAX_GREEN_TIME
        results["decision_engine"] = "MANUAL_ADMIN_OVERRIDE"

    # 6. DYNAMICALLY REWRITE UI CONTAINER SLOTS
    for idx in range(Config.NUM_LANES):
        viewports[idx].image(annotated_feed, channels="BGR", caption=f"AI Monitor Tracking - Corridor {idx+1}")
        
    metric_lane.metric(label="Active Intersection Lane Phase", value=f"Lane {results['active_lane'] + 1}")
    metric_duration.metric(label="Calculated Adaptive Green Duration", value=f"{results['green_duration_seconds']} Seconds")
    metric_volume.metric(label="Predicted System Volume (Next Interval)", value=f"{round(results['predicted_next_volume'], 1)} Vehicles")
    
    latency = results['processing_latency_ms']
    status_color = "green" if results['latency_compliant'] else "red"
    metric_latency.markdown(f"**Pipeline Latency Execution Metric:** :{status_color}[{round(latency, 2)} ms]")
    metric_engine.markdown(f"**Decision Architecture Context Engine:** `{results['decision_engine']}`")
    
    # Refresh the visualization distribution chart
    chart_placeholder.bar_chart(pipeline.lane_vehicle_counts)
    
    # Sleep threshold to stabilize frame cadence loops (~20 FPS matching inference speed)
    time.sleep(0.05)

cap.release()

# Static Display fallback block when system execution tracking loop is switched off
if not run_system:
    st.info("ℹ️ System Standby. Flip the 'Activate Live ITMS Pipeline Loop' sidebar toggle switch to begin monitoring operations.")