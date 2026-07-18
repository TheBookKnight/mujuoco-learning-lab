"""
This file introduces PD Control. This introduces damping into our controller, 
allowing it to settle into a stable position without oscillating around the target.  

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script: uv run 06_pd_control.py
"""
import mujoco
import numpy as np

def main():
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L6.xml")
    data = mujoco.MjData(model)
    
    print("Starting Lesson 6 simulation...")
    
    for i in range(500):
        mujoco.mj_forward(model, data)
        
        # 1. OBSERVE: Get position and velocity (Index 2 for Z-axis)
        box_pos_z = data.sensor("box_position").data[2]
        box_vel_z = data.sensor("box_velocity").data[2]
        target_pos_z = data.site("target").xpos[2]
        
        # 2. REWARD (Using full 3D vectors for the distance calculation)
        box_pos_3d = data.sensor("box_position").data
        target_pos_3d = data.site("target").xpos
        distance = np.linalg.norm(box_pos_3d - target_pos_3d)
        reward = -distance
        
        # 3. ACTION: Proportional-Derivative (PD) Controller
        # P-term: Push towards the target
        z_error = target_pos_z - box_pos_z
        kp = 50.0 
        p_term = kp * z_error
        
        # D-term: Oppose current velocity
        # Error velocity = target_velocity (0) - current_velocity
        kd = 5.0
        d_term = kd * (0.0 - box_vel_z)
        
        # Combine them and apply to the actuator
        data.ctrl = p_term + d_term
        
        mujoco.mj_step(model, data)
        
        if i % 50 == 0:
            print(f"Step {i:03d} | Box Z: {box_pos_z:.4f}m | Vel Z: {box_vel_z:.4f}m/s | Ctrl: {data.ctrl[0]:.4f}N")

if __name__ == "__main__":
    main()