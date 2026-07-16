"""
This file is a simple simulation of a box falling in a 3D space.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script: uv run 01_hello_mujoco.py
"""
import mujoco

def main():
    # 1. Load the static model from the XML file
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L1.xml")
    
    # 2. Create the dynamic state (data) object based on the model
    data = mujoco.MjData(model)
    
    print("Starting simulation...")
    
    # 3. Step the simulation forward 100 times
    for i in range(100):
        # mj_step advances the simulation by one timestep (default 0.002 seconds)
        mujoco.mj_step(model, data)
        
        # We query data.qpos, which stores the generalized positions of all joints.
        # Since our box has a free joint, its 3D position (X, Y, Z) and 
        # 4D rotation (quaternion) are stored here.
        # The Z-coordinate (altitude) is at index 2.
        z_position = data.qpos[2]
        
        if i % 10 == 0:
            print(f"Step {i:03d} | Time: {data.time:.3f}s | Box Z-altitude: {z_position:.4f}m")

if __name__ == "__main__":
    main()
