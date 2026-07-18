# MuJoCo Learning Lab

Welcome to the **MuJoCo Learning Lab**! This repository is a dedicated space for learning, experimenting, and mastering physical simulations using the [MuJoCo](https://mujoco.org/) physics engine via its Python bindings. It walks through loading models, inspecting state variables, robust data access, and synchronizing physics pipelines with sensor updates.

---

## Lessons Summary

### 1. [Lesson 1: Hello MuJoCo](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/01_hello_mujoco.py)
* **Script:** [01_hello_mujoco.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/01_hello_mujoco.py)
* **XML Model:** [hello_mujoco_L1.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L1.xml)
* **Focus:** Basic simulation setup. 
* **Details:** Demonstrates loading a static 3D model XML (a falling box with a free joint), creating the dynamic simulation state (`mujoco.MjData`), stepping the simulation forward using `mujoco.mj_step`, and querying basic generalized position coordinate states (`data.qpos`) such as the box's Z-altitude.

### 2. [Lesson 2: Robust State Access](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/02_robust_state.py)
* **Script:** [02_robust_state.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/02_robust_state.py)
* **XML Model:** [hello_mujoco_L2.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L2.xml)
* **Focus:** Programmatic named access to joints and sensors.
* **Details:** Explores robust ways to retrieve coordinate and sensor data by name, avoiding hardcoded absolute offsets into the global `data.qpos` and `data.sensordata` arrays. It shows two main access methods:
  * **Joint Access:** Using `.joint("box_joint").qpos` to query the joint's local state slice.
  * **Sensor Access:** Using `.sensor("box_position").data` to isolate specific sensor readings.

### 3. [Lesson 3: Physics and Sensor Consistency](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/03_consistency.py)
* **Script:** [03_consistency.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/03_consistency.py)
* **XML Model:** [hello_mujoco_L2.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L2.xml)
* **Focus:** Fixing the pipeline lag between physics state and sensor data.
* **Details:** Resolves the pipeline lag where `mujoco.mj_step` advances positions and velocities to the next timestep $t+1$, but sensor readings remain evaluated using the previous timestep $t$. Calling `mujoco.mj_forward` immediately after `mj_step` re-evaluates kinematics, sensors, and accelerations, ensuring that both joint values and sensor data are perfectly synchronized at the exact same moment in time.

### 4. [Lesson 4: Distance Reward](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/04_distance_reward.py)
* **Script:** [04_distance_reward.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/04_distance_reward.py)
* **XML Model:** [hello_mujoco_L4.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L4.xml)
* **Focus:** Implementation of a continuous dense reward function based on target distance.
* **Details:** Introduces a target `<site>` in the XML model and computes the Euclidean distance between the falling box and this target at each simulation step:
  * **Site Access:** Using `data.site("target").xpos` to retrieve the target's 3D position vector.
  * **Dense Reward:** Computes the negative Euclidean distance as a continuous dense reward, which provides feedback at every step (getting "warmer" or "colder" as it approaches the target). Because the simulation lacks actuators, the box simply free-falls past the target due to gravity.

### 5. [Lesson 5: Actuation & Proportional (P) Control](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/05_actuation.py)
* **Script:** [05_actuation.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/05_actuation.py)
* **XML Model:** [hello_mujoco_L5.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L5.xml)
* **Focus:** Introducing actuators and closing the loop using feedback control.
* **Details:** Connects a virtual motor actuator to the slide joint of the box to actively counteract gravity and drive the box towards the target altitude:
  * **P Controller:** Implements a simple Proportional (P) controller that applies force based on the position error along the Z-axis: `data.ctrl[0] = Kp * z_error`.
  * **Behavior & Limitations:** The system exhibits sustained oscillations around the target and does not settle. This is because a P-only controller acts like a frictionless spring (providing restoring force but no energy dissipation/damping). Additionally, because of gravity, a pure P-controller will experience a steady-state error (offset) because it needs a non-zero error to output enough force to counteract gravity.
  * **Damping Resolution:** To stabilize the system, a **Derivative (D)** term (damping) must be introduced to oppose velocity. By measuring the joint velocity via `data.qvel[0]`, you can implement a Proportional-Derivative (PD) controller: `data.ctrl[0] = Kp * z_error - Kd * box_z_vel`, which removes the oscillations and allows the box to settle.

---

## Running the Lessons

To run any of the lessons, you can use `uv run`:

```bash
cd lessons
uv run <lesson_filename.py>
```