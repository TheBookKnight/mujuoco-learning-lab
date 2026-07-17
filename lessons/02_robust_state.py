"""
This file is a simple simulation of accessing data from a falling box in a 3D space, with a focus on robust state access.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script: uv run 02_robust_state.py
"""
import mujoco

def main():
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L2.xml")
    data = mujoco.MjData(model)
    
    print("Starting Lesson 2 simulation...")
    
    for i in range(100):
        mujoco.mj_step(model, data)
        
        # Method 1: Named access to the joint's position state
        # The .joint() accessor automatically finds the correct slice in data.qpos
        # A free joint has 7 values: [x, y, z, qw, qx, qy, qz]
        z_pos_joint = data.joint("box_joint").qpos[2]
        
        # Method 2: Named access to the sensor data
        # All sensors are concatenated in data.sensordata. 
        # The .sensor() accessor isolates the data for just this sensor.
        # framepos returns a 3D vector [x, y, z]
        z_pos_sensor = data.sensor("box_position").data[2]
        
        if i % 10 == 0:
            print(f"Step {i:03d} | Joint Z: {z_pos_joint:.4f}m | Sensor Z: {z_pos_sensor:.4f}m")

if __name__ == "__main__":
    main()