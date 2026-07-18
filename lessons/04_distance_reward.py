"""
This file introduces the concept of a "reward" function.

Instead of just watching the box fall, we now define a goal: get close to the green target sphere.
We calculate the 3D distance between the box and the target at every step.
We use negative distance as the reward, meaning the reward gets "less negative" (better) as we get closer to the target.

* Note: There is currently no controller or actuator in this simulation to affect the box's position, so it will simply fall under gravity past the target.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script: uv run 04_distance_reward.py
"""

import mujoco
import numpy as np

def main():
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L4.xml")
    data = mujoco.MjData(model)
    
    print("Starting Lesson 4 simulation...")
    
    for i in range(100):
        # Sync physics and sensors (prevents pipeline lag)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        
        # 1. Get the 3D position vector of the box from our sensor
        box_pos = data.sensor("box_position").data
        
        # 2. Get the 3D position vector of the target site directly
        target_pos = data.site("target").xpos
        
        # 3. Calculate Euclidean distance
        distance = np.linalg.norm(box_pos - target_pos)
        
        # 4. Compute the reward! 
        # We use negative distance so the reward *increases* (gets closer to 0) 
        # as the box approaches the target.
        reward = -distance
        
        if i % 10 == 0:
            print(f"Step {i:03d} | Box Z: {box_pos[2]:.4f}m | Distance: {distance:.4f}m | Reward: {reward:.4f}")

if __name__ == "__main__":
    main()