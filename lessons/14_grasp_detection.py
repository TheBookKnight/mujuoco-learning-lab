"""
In this lesson, we add a block to our Panda scene and try to grasp it.

To run this code:
1. Navigate to the directory containing this file: cd /Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons
2. Run the script:
   - On macOS: uv run mjpython 14_grasp_detection.py
   - On Linux/Windows: uv run 14_grasp_detection.py
"""

import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from robot_descriptions.panda_mj_description import MJCF_PATH


def create_scene_xml(output_path: str = "../assets/panda_arm_scene.xml"):
    """
    Programmatically generates an MJCF scene that includes the Panda robot,
    a table, and a small graspable box using MuJoCo MjSpec.
    """
    panda_dir = os.path.dirname(MJCF_PATH)

    # Use MjSpec to safely merge the Panda model with proper mesh assets
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    spec.meshdir = os.path.join(panda_dir, "assets")

    # Table top surface (top at Z = 0.50m)
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.554, 0.0, 0.25],
        size=[0.2, 0.2, 0.25],
        rgba=[0.4, 0.3, 0.2, 1],
        condim=3,
    )

    # Small blue block to grasp (4cm cube, resting on table at Z = 0.52m)
    block_body = spec.worldbody.add_body(name="target_block", pos=[0.554, 0.0, 0.52])
    block_body.add_freejoint(name="block_joint")
    block_body.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        rgba=[0.1, 0.2, 0.8, 1],
        mass=0.02,
        condim=4,
        friction=[2.0, 0.1, 0.01],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(spec.to_xml())
    print(f"Successfully generated {output_path}")


def get_contact_force(model, data, body1_name: str, body2_name: str) -> tuple[bool, float]:
    """
    Scans the active contact list and computes contact force (in Newtons)
    between any geoms belonging to two given bodies.
    """
    body1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    body2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
    if body1_id < 0 or body2_id < 0:
        return False, 0.0

    contact_detected = False
    total_force = 0.0

    for i in range(data.ncon):
        con = data.contact[i]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        if (b1 == body1_id and b2 == body2_id) or (b1 == body2_id and b2 == body1_id):
            contact_detected = True
            c_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, c_force)
            total_force += float(np.linalg.norm(c_force[:3]))

    return contact_detected, total_force


def main():
    # Determine correct relative/absolute path whether run from repo root or lessons/
    xml_path = "../assets/panda_arm_scene.xml" if os.path.exists("../assets") or not os.path.exists("assets") else "assets/panda_arm_scene.xml"

    # 1. Prepare scene XML
    create_scene_xml(output_path=xml_path)

    # 2. Load model & data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initialize arm to neutral home posture
    home_arm_pose = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    
    # Panda arm joints 1 to 7 in radians (hinge/revolute)
    data.qpos[:7] = home_arm_pose
    
    # Panda Gripper Fingers in meters (slide joints)
    data.qpos[7:9] = [0.04, 0.04]  # Gripper fingers open initially
    
    # Block 3D position in meters
    data.qpos[9:12] = [0.554, 0.0, 0.52]  # Block positioned between fingertips
    
    # Block 3D orientation in quaternion (w, x, y, z)
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]  # Neutral quaternion orientation

    # Set arm hold controls
    data.ctrl[:7] = home_arm_pose

    # In Panda MJCF, ctrl[7] = 255 is fully open (0.04m), ctrl[7] = 0.0 is closed
    data.ctrl[7] = 0.0  # Actively command gripper to close

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")

    print("\n" + "=" * 50)
    print("Starting Lesson 14: Grasp Detection & Force Sensing")
    print("=" * 50 + "\n")

    # Launch passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable contact force visualization by default
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        lift_start_time = None
        has_reported_grasp = False

        while viewer.is_running():
            step_start = time.time()

            # --- SENSING CONTACT FORCES ---
            left_touch, left_f = get_contact_force(model, data, "left_finger", "target_block")
            right_touch, right_f = get_contact_force(model, data, "right_finger", "target_block")
            is_grasped = left_touch and right_touch

            # --- CONTROL BEHAVIOR: GRASP THEN LIFT ---
            if is_grasped and lift_start_time is None:
                lift_start_time = data.time

            # After grasping stably for 0.5s, lift arm into the air
            if lift_start_time is not None and (data.time - lift_start_time) > 0.5:
                lift_progress = min(1.0, (data.time - lift_start_time - 0.5) / 1.0)
                # Elevate shoulder (joint2) and elbow (joint4)
                data.ctrl[:7] = [
                    0.0,
                    -0.4 * lift_progress,
                    0.0,
                    -1.57079 - 0.2 * lift_progress,
                    0.0,
                    1.57079,
                    -0.7853,
                ]

            # Step simulation
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # Periodic terminal feedback
            if int(data.time * 100) % 25 == 0:
                block_z = data.xpos[block_id][2]
                if is_grasped:
                    if not has_reported_grasp:
                        print(f"[{data.time:.3f}s] SUCCESS: Grasp secured! L force: {left_f:.2f}N | R force: {right_f:.2f}N")
                        has_reported_grasp = True
                    elif lift_start_time is not None and (data.time - lift_start_time) > 0.5:
                        print(f"[{data.time:.3f}s] LIFTING: Block height = {block_z:.3f}m | Total pinch force = {left_f + right_f:.2f}N")
                elif left_touch or right_touch:
                    print(f"[{data.time:.3f}s] Partial contact: L={left_touch} ({left_f:.2f}N) | R={right_touch} ({right_f:.2f}N)")

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()