"""
In this lesson, we refactor the Panda tracking code into our professional Object-Oriented Environment architecture.
Let's standardized our Panda observations, normalize our action spaces, and build our second production-ready environment class.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 13_design_review_2.py
   - On Linux/Windows: uv run 13_design_review_2.py
"""

import time
import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions.panda_mj_description import MJCF_PATH

class PandaReachEnv:
    def __init__(self, target_pos: np.ndarray):
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        self.target_pos = np.array(target_pos)
        
        # Extract physical actuator limits from the MJCF model for the 7 arm joints.
        # This allows us to perform action denormalization dynamically.
        # ctrlrange is a 2D array of shape (nu, 2)
        self.actuator_min = self.model.actuator_ctrlrange[:7, 0].copy()
        self.actuator_max = self.model.actuator_ctrlrange[:7, 1].copy()

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        
        # Warm up the pipeline so sensors/positions are populated for t=0
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation()

    def get_observation(self) -> dict:
        # Extract positions and velocities of the first 7 rotational joints
        qpos_arm = self.data.qpos[:7].copy()
        qvel_arm = self.data.qvel[:7].copy()
        
        # Extract hand end-effector global Cartesian position
        hand_pos = self.data.body("hand").xpos.copy()
        
        return {
            "joint_positions": qpos_arm,
            "joint_velocities": qvel_arm,
            "hand_pos": hand_pos,
            "target_pos": self.target_pos.copy()
        }

    def compute_reward(self, obs: dict) -> float:
        # 1. Primary Task: Minimize 3D Euclidean distance to target
        distance = np.linalg.norm(obs["hand_pos"] - obs["target_pos"])
        tracking_reward = -distance
        
        # 2. Smoothness/Energy: Penalize high joint velocities
        energy_penalty = -0.01 * np.sum(np.square(obs["joint_velocities"]))
        
        # 3. Safety Geofencing: Apply a massive penalty if the hand drops below Z = 0.25m
        # (This simulates a physical tabletop boundary we must avoid smashing into)
        safety_penalty = 0.0
        if obs["hand_pos"][2] < 0.25:
            safety_penalty = -10.0
            
        return tracking_reward + energy_penalty + safety_penalty

    def step(self, normalized_action: np.ndarray) -> tuple[dict, float]:
        """
        Expects a normalized action array of shape (7,) with values in [-1.0, 1.0].
        Maps normalized actions to the robot's physical actuator limits,
        steps the simulation, and returns (observation, reward).
        """
        # Ensure the action array is a numpy array clamped to [-1, 1]
        action = np.clip(np.array(normalized_action), -1.0, 1.0)
        
        # LINEAR DENORMALIZATION MATH:
        # Maps [-1, 1] to [actuator_min, actuator_max]
        denormalized_action = self.actuator_min + 0.5 * (action + 1.0) * (self.actuator_max - self.actuator_min)
        
        # Apply the 7 denormalized arm commands to ctrl array
        self.data.ctrl[:7] = denormalized_action
        
        # Keep gripper fingers stationary (Actuator 7 = index 7)
        self.data.ctrl[5] = 0.0
        
        # Step physics and synchronize the pipeline
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        obs = self.get_observation()
        reward = self.compute_reward(obs)
        
        return obs, reward


def main():
    # Target coordinate: X=0.2, Y=0.3, Z=0.6
    target = np.array([0.2, 0.3, 0.6])
    env = PandaReachEnv(target_pos=target)
    obs = env.reset()
    
    print("\n" + "="*50)
    print("Starting Design Review 2: Panda Environment")
    print("="*50 + "\n")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- AGENT CODE (Continuous Wave) ---
            t = env.data.time
            
            # The agent decides targets in radian-space, then we normalize them!
            # Let's define the desired physical target angles (in radians)
            target_joint1 = 1.0 * np.sin(t)
            target_joint2 = 0.5 * np.cos(t)
            target_joint3 = 0.8 * np.sin(t * 0.5)
            target_joint4 = -1.5 + 0.5 * np.sin(t)
            
            desired_joints = np.array([target_joint1, target_joint2, target_joint3, target_joint4, 0.0, 0.0, 0.0])
            
            # NORMALIZATION MATH (inverse of denormalization):
            # Maps physical joint angles back to [-1.0, 1.0] relative to ctrl limits
            low = env.actuator_min
            high = env.actuator_max
            normalized_action = 2.0 * (desired_joints - low) / (high - low) - 1.0
            
            # --- ENVIRONMENT STEP ---
            obs, reward = env.step(normalized_action)
            
            # Render & throttle
            viewer.sync()
            
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
