"""
This script inspects the kinematics and actuators of the Franka Emika Panda robot.
It programmatically reads joint types, physical limits, and actuator control ranges 
from the compiled MuJoCo model.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 11_robot_inspection.py
   - On Linux/Windows: uv run 11_robot_inspection.py
"""
import time
import mujoco
import mujoco.viewer
from robot_descriptions.panda_mj_description import MJCF_PATH

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    
    # Force a forward kinematics pass to populate structural dimensions
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*50)
    print("FRANKA PANDA MODEL DIAGNOSTICS")
    print("="*50)
    print(f"Total Degrees of Freedom (nv): {model.nv}")
    print(f"Total Actuators (nu): {model.nu}")
    
    # 1. Programmatically inspect all Joints
    print("\n--- JOINTS IN THE KINEMATIC TREE ---")
    print("Type 3 (hinge): 1 rotational degree of freedom (measured in radians).")
    print("Type 2 (slide): 1 translational degree of freedom (measured in meters).")
    print("Range represents the physical, mechanical limits of the joints.\n")
    for j_id in range(model.njnt):
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        j_type = model.jnt_type[j_id] # Type enum (hinge, slide, etc.)
        
        # Check if the joint has limits defined
        has_limit = model.jnt_limited[j_id]
        j_range = model.jnt_range[j_id] if has_limit else "No limits"
        
        print(f"Joint {j_id}: '{j_name}' | Type: {j_type} | Range: {j_range}")
        
    # 2. Programmatically inspect all Actuators
    print("\n--- ACTUATORS ---")
    print("Control Range represents the valid control signals for the actuators.")
    print("For arm position servos, this matches the joint limits (radians).")
    print("For the gripper (e.g., actuator8), it uses a normalized 8-bit integer (0 to 255)")
    print("to command the onboard micro-controller, which internally maps to meters.\n")
    for a_id in range(model.nu):
        a_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a_id)
        has_ctrl_limit = model.actuator_ctrllimited[a_id]
        ctrl_range = model.actuator_ctrlrange[a_id] if has_ctrl_limit else "No control limits"
        
        print(f"Actuator {a_id}: '{a_name}' | Control Range: {ctrl_range}")
    print("="*50 + "\n")

    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer running. Press Ctrl+C in terminal or close window to exit.")
        while viewer.is_running():
            step_start = time.time()
            
            # Step simulation (Panda arm actively holding itself at 0.0)
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()