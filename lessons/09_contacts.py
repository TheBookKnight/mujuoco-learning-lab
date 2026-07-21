"""
This script implements the BoxDropEnv class for the environment. It sets up an infinite contact plane.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 09_contacts.py
   - On Linux/Windows: uv run 09_contacts.py
"""
import time
import mujoco
import mujoco.viewer

class BoxDropEnv:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> dict:
        return {
            "box_pos": self.data.sensor("box_position").data.copy(),
            "box_vel": self.data.sensor("box_velocity").data.copy(),
            # Touch sensors return a 1D array with a single scalar value
            "touch_force": self.data.sensor("box_touch").data.copy()
        }

    def step(self, action: float):
        self.data.ctrl = action
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

def main():
    env = BoxDropEnv("../assets/hello_mujoco_L9.xml")
    obs = env.reset()
    
    print("Starting Lesson 9: Contacts...")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        
        # We will track whether we've printed the resting force yet
        settled = False
        
        while viewer.is_running():
            step_start = time.time()
            
            # Apply 0 force. Let gravity do the work.
            action = 0.0
            obs = env.step(action)
            
            force = obs["touch_force"][0]
            
            # If the box is experiencing a massive impact force
            if force > 20.0:
                print(f"IMPACT DETECTED! Force: {force:.2f} N")
            # If the box has settled on the ground
            elif 9.0 < force < 10.0 and not settled:
                print(f"Box settled. Resting Force: {force:.2f} N")
                settled = True
            
            viewer.sync()
            
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()