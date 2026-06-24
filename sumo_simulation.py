import os
import sys
import time
import random
from config import Config
from pipeline_manager import ITMSPipeline

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

class SumoITMSManager:
    def __init__(self, use_gui=True):
        self.pipeline = ITMSPipeline()
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.config_path = "simulation/intersection.sumocfg"
        self.tl_id = "junction_center" 
        self.phase_map = {}
        
        # Coordinates to place the floating counters near the crosswalks
        self.counter_positions = {
            "North_Counter": (-8, 25),
            "East_Counter": (25, 8),
            "South_Counter": (8, -25),
            "West_Counter": (-25, -8)
        }

    def init_ui_counters(self):
        """Spawns background text blocks that act as our UI display overhead boards."""
        for counter_id, pos in self.counter_positions.items():
            try:
                # Add a point of interest that displays text cleanly above the lanes
                traci.poi.add(counter_id, pos[0], pos[1], (0,0,0,255), poiType="3", layer=100)
            except Exception:
                pass

    def update_ui_counters(self, current_timer, active_lane):
        """Updates the numeric counters dynamically based on what the AI chooses."""
        for i, (counter_id, _) in enumerate(self.counter_positions.items()):
            try:
                if i == active_lane:
                    # Show the remaining green time left
                    traci.poi.setType(counter_id, str(int(current_timer)))
                else:
                    # Display 0 or waiting queue count
                    traci.poi.setType(counter_id, "0")
            except Exception:
                pass

    def build_dynamic_phases(self):
        try:
            current_state = traci.trafficlight.getRedYellowGreenState(self.tl_id)
            num_links = len(current_state)
            half_links = num_links // 2
            remaining_links = num_links - half_links
            
            self.phase_map = {
                0: ("G" * half_links) + ("r" * remaining_links),
                1: ("r" * half_links) + ("G" * remaining_links),
                2: ("y" * half_links) + ("r" * remaining_links),
                3: ("r" * half_links) + ("y" * remaining_links)
            }
        except Exception:
            self.phase_map = {0: "GGggrrrrGGggrrrr", 1: "rrrrGGggrrrrGGgg"}

    def fetch_lane_densities(self):
        lane_ids = ["edge_N_0_0", "edge_E_0_0", "edge_S_0_0", "edge_W_0_0"]
        counts = []
        for lane in lane_ids:
            try:
                counts.append(traci.lane.getLastStepVehicleNumber(lane))
            except Exception:
                counts.append(0)
        return counts

    def run_simulation(self):
        try: traci.close()
        except Exception: pass  
            
        sumo_cmd = [self.sumo_binary, "-c", self.config_path, "--start"]
        traci.start(sumo_cmd, port=8813)
        
        self.build_dynamic_phases()
        self.init_ui_counters()
        
        step = 0
        green_timer_remaining = 0
        optimal_lane = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            try:
                traci.simulationStep()  
            except Exception:
                break
            
            if green_timer_remaining <= 0:
                simulated_counts = self.fetch_lane_densities()
                self.pipeline.lane_vehicle_counts = simulated_counts
                
                state = simulated_counts + [0, 0, 0, 0] 
                optimal_lane = self.pipeline.controller.select_action(state)
                allocated_time = self.pipeline.controller.calculate_adaptive_time(simulated_counts[optimal_lane])

                target_string_code = self.phase_map.get(optimal_lane, list(self.phase_map.values())[0])
                traci.trafficlight.setRedYellowGreenState(self.tl_id, target_string_code)
                
                green_timer_remaining = allocated_time
            else:
                green_timer_remaining -= 1
            
            # Keep text overlay counting down every second frame execution
            self.update_ui_counters(green_timer_remaining, optimal_lane)
            
            step += 1
            time.sleep(0.05)

        try: traci.close()
        except Exception: pass

if __name__ == "__main__":
    sim_manager = SumoITMSManager(use_gui=True)
    sim_manager.run_simulation()