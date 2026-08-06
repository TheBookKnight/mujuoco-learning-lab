"""
This script demonstrates Joint Space Action Control and End-Effector Tracking for the Panda arm.
It shows how to command joint targets using `data.ctrl` and track the resulting 3D position 
of the end-effector (hand) using forward kinematics.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 12_panda_control.py
   - On Linux/Windows: uv run 12_panda_control.py
"""
import time
import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions.panda_mj_description import MJCF_PATH

def main():
    # Load the pre-built Menagerie model
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    
    # 1. Define a virtual target in 3D world space (where we want the hand to reach)
    # This target is at X = 0.2m, Y = 0.3m, Z = 0.6m relative to the robot base
    target_pos = np.array([0.2, 0.3, 0.6])
    
    print("\n" + "="*50)
    print("Starting Lesson 12: Panda Arm Joint Control & Tracking")
    print("="*50)
    print(f"Control Array Size: {len(data.ctrl)}")
    print(f"Target Point: {target_pos}\n")

    # Launch passive viewer on Apple Silicon Mac
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while viewer.is_running():
            step_start = time.time()
            
            # 2. GENERATE TIME-VARYING JOINT COMMANDS (Our "Agent" action)
            # We will use sine waves to command the joints to swing smoothly.
            # We command the first 4 rotational joints of the Panda arm.
            t = data.time
            
            # Set joint targets (in radians) within the physical ranges
            data.ctrl[0] = 1.0 * np.sin(t)          # Joint 1 swing
            data.ctrl[1] = 0.5 * np.cos(t)          # Joint 2 nod
            data.ctrl[2] = 0.8 * np.sin(t * 0.5)    # Joint 3 twist
            data.ctrl[3] = -1.5 + 0.5 * np.sin(t)   # Joint 4 elbow bend
            
            # Keep fingers stationary (Actuator 7 controls gripper)
            data.ctrl[7] = 0.0 
            
            # 3. STEP PHYSICS
            # This advances positions and calculates forward kinematics
            mujoco.mj_step(model, data)
            
            # Synchronize state so sensors and global frames are consistent
            mujoco.mj_forward(model, data)
            
            # 4. TRACKING: Extract the hand's global coordinate
            # We use named body access to isolate the hand frame pos
            hand_pos = data.body("hand").xpos.copy()
            
            # 5. REWARD CALCULATION
            distance = np.linalg.norm(hand_pos - target_pos)
            tracking_reward = -distance
            
            # Optional energy penalty (penalizing high joint speeds for safety)
            joint_velocities = data.qvel[:7].copy()
            energy_penalty = -0.01 * np.sum(np.square(joint_velocities))
            
            total_reward = tracking_reward + energy_penalty
            
            # Log results periodically
            if int(t * 100) % 50 == 0:
                print(f"Time: {t:.2f}s | Hand: {hand_pos} | Dist: {distance:.3f}m | Reward: {total_reward:.3f}")
                
            viewer.sync()
            
            # Real-time throttling
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()