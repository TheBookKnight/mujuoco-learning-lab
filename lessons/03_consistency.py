"""
This file fixes the pipeline lag between physics and sensors by calling mj_forward after mj_step.

The pipeline lag is caused when it computes all sensor readings using the current state. And then 
it uses the numerical integrator to advance the positions and velocities to the next time step.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script: uv run 03_consistency.py
"""
import mujoco

def main():
    # Use the same model. This is to fix the pipeline lag between physics and sensors.
    model = mujoco.MjModel.from_xml_path("../assets/hello_mujoco_L2.xml")
    data = mujoco.MjData(model)
    
    print("Starting Lesson 3 simulation...")
    
    for i in range(100):
        # 1. Advance the physics state (positions/velocities move to t+1, sensors stay at t)
        mujoco.mj_step(model, data)
        
        # 2. Synchronize the pipeline! 
        # Re-evaluates all sensors, kinematics, and accelerations for the new state at t+1.
        mujoco.mj_forward(model, data)
        
        # Now both of these are guaranteed to be from the exact same moment in time
        z_pos_joint = data.joint("box_joint").qpos[2]
        z_pos_sensor = data.sensor("box_position").data[2]
        
        if i % 10 == 0:
            print(f"Step {i:03d} | Joint Z: {z_pos_joint:.4f}m | Sensor Z: {z_pos_sensor:.4f}m")

if __name__ == "__main__":
    main()