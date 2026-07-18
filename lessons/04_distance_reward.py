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