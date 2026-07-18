"""
This file introduces the concept of "actuation" in MuJoCo.

We now have an actuator (a virtual motor) connected to the box. This actuator can apply a linear force to the box, counteracting gravity. We implement a simple Proportional (P) controller that calculates how much force to apply based on the error between the box's current position and the target position.

* Note #1: The reward is still the negative distance to the target. The actuator allows us to actively control the box to minimize this distance.
* Note #2: The box still oscillates around the target position. This is because the simple P controller does not have "damping" or "integral" terms, which would help it settle into a stable position. This oscillation is a common challenge in robotics and control theory, and it is something that we will explore in future lessons.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script: uv run 05_actuation.py
"""
import mujoco
import numpy as np

def main():
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L5.xml")
    data = mujoco.MjData(model)
    
    print("Starting Lesson 5 simulation...")
    
    # Run longer to watch the controller stabilize the box
    for i in range(500):
        # 1. OBSERVE: Get the synchronized state
        mujoco.mj_forward(model, data)
        box_pos = data.sensor("box_position").data
        target_pos = data.site("target").xpos
        
        # 2. REWARD: Calculate dense reward
        distance = np.linalg.norm(box_pos - target_pos)
        reward = -distance
        
        # 3. ACTION: Simple Proportional (P) Controller (Our "Agent")
        # The error is how far we are from the target Z-altitude
        box_z_pos = box_pos[2]
        z_error = target_pos[2] - box_z_pos
        
        # We multiply the error by a "gain" (Kp) to calculate how hard to push.
        # If z_error is positive (box is below target), we push up.
        # If z_error is negative (box is above target), we pull down.
        kp = 50.0 
        
        # data.ctrl holds the inputs for all actuators. 
        # We only have 1 actuator, so we write to index 0.
        data.ctrl = kp * z_error
        
        # 4. STEP: Advance the physics engine with our new control applied
        mujoco.mj_step(model, data)
        
        if i % 50 == 0:
            print(f"Step {i:03d} | Box Z: {box_z_pos:.4f}m | Reward: {reward:.4f} | Ctrl Force: {data.ctrl[0]:.4f}N")

if __name__ == "__main__":
    main()