"""
This script implements the Environment class for the BoxHoverEnv environment.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 08_design_review.py
   - On Linux/Windows: uv run 08_design_review.py
"""

import time
import mujoco
import mujoco.viewer
import numpy as np

class BoxHoverEnv:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

    def reset(self):
        # Reset the physics state to the default configuration [3]
        mujoco.mj_resetData(self.model, self.data)
        
        # Forward the dynamics once to ensure sensors and kinematics are populated for t=0
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_observation()

    def get_observation(self) -> dict:
        # CRITICAL: We must use .copy()
        # MuJoCo exposes memory views. If we don't copy, the RL agent's 
        # historical observations will mutate every time the physics engine steps.
        return {
            "box_pos": self.data.sensor("box_position").data.copy(),
            "box_vel": self.data.sensor("box_velocity").data.copy(),
            "target_pos": self.data.site("target").xpos.copy()
        }

    def compute_reward(self, obs: dict) -> float:
        # The reward is purely a function of the observation
        distance = np.linalg.norm(obs["box_pos"] - obs["target_pos"])
        velocity_penalty = np.linalg.norm(obs["box_vel"])
        
        # Combine distance penalty and smoothness penalty
        return -distance - 0.05 * velocity_penalty

    def step(self, action: float):
        # 1. Apply the agent's action
        self.data.ctrl = action
        
        # 2. Step the physics engine
        mujoco.mj_step(self.model, self.data)
        
        # 3. Synchronize the pipeline lag!
        # After an mj_step, position-dependent sensor values correspond to the previous state [4].
        # We must call mj_forward to get accurate observations for the new state.
        mujoco.mj_forward(self.model, self.data)
        
        # 4. Generate the new observation and reward
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        
        return obs, reward


def main():
    # Instantiate our new environment
    # Uses last working model as reference
    env = BoxHoverEnv("../assets/hello_mujoco_L6.xml")
    obs = env.reset()
    
    print("Starting Design Review 1: Environment Architecture...")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- AGENT CODE ---
            # The agent only sees the 'obs' dictionary. It does not touch mjData directly.
            z_error = obs["target_pos"][2] - obs["box_pos"][2]
            v_error = 0.0 - obs["box_vel"][2]
            
            # Calculate action (PD Control)
            # 50 is the proportional gain, 5 is the derivative gain
            action = 50.0 * z_error + 5.0 * v_error
            
            # --- ENVIRONMENT CODE ---
            # Pass the action into the black-box environment
            obs, reward = env.step(action)
            
            # --- VISUALIZATION ---
            viewer.sync()
            
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
