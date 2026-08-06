"""
This script loads the Franka Emika Panda robot from the MuJoCo Menagerie using the robot_descriptions library and visualizes it in the passive viewer.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 10_menagerie.py
   - On Linux/Windows: uv run 10_menagerie.py
"""
import time
import mujoco
import mujoco.viewer

# Dynamically import the path to the Menagerie Franka Emika Panda MJCF file
from robot_descriptions.panda_mj_description import MJCF_PATH

def main():
    print(f"Loading robot from: {MJCF_PATH}")
    
    # Load the professional model directly from the package
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    
    mujoco.mj_forward(model, data)
    
    print("Starting Lesson 10: Visual Interrogation...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # We are not applying any control yet. 
            # The robot will simply collapse under gravity.
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()