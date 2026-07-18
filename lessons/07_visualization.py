"""
This file introduces the passive viewer in MuJoCo.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 07_visualization.py
   - On Linux/Windows: uv run 07_visualization.py
"""
import time
import mujoco
import mujoco.viewer
import numpy as np

def main():
    # Load the environment from Lesson 6 (Actuation + PD control)
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L6.xml")
    data = mujoco.MjData(model)
    
    print("Starting Lesson 7: Visualization...")
    
    # Launch the passive viewer in a separate thread
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Run until the user closes the viewer window
        while viewer.is_running():
            step_start = time.time()
            
            # --- 1. OBSERVE & REWARD ---
            mujoco.mj_forward(model, data)
            
            box_pos_3d = data.sensor("box_position").data
            target_pos_3d = data.site("target").xpos
            distance = np.linalg.norm(box_pos_3d - target_pos_3d)
            reward = -distance
            
            # --- 2. CONTROL (PD Agent) ---
            box_pos_z = box_pos_3d[2]
            target_pos_z = target_pos_3d[2]
            box_vel_z = data.sensor("box_velocity").data[2]
            
            p_term = 50.0 * (target_pos_z - box_pos_z)
            d_term = 5.0 * (0.0 - box_vel_z)
            
            data.ctrl = p_term + d_term
            
            # --- 3. STEP PHYSICS ---
            mujoco.mj_step(model, data)
            
            # --- 4. RENDER SYNC ---
            # Tell the viewer to grab the latest physics state for rendering
            viewer.sync()
            
            # --- 5. REAL-TIME PACING ---
            # Throttle the loop so 1 simulated second roughly equals 1 real second
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()