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

### 6. [Lesson 6: Proportional-Derivative (PD) Control](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/06_pd_control.py)
* **Script:** [06_pd_control.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/06_pd_control.py)
* **XML Model:** [hello_mujoco_L6.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L6.xml)
* **Focus:** Implementation of a full PD (Proportional-Derivative) controller to add damping and stabilize the system.
* **Details:** Builds upon the actuation ideas by implementing a full PD loop using the new velocity sensor to stabilize the box at the target altitude:
  * **Velocity Sensing:** Utilizes a `<framelinvel>` sensor (`box_velocity`) to measure the box's 3D velocity vector directly.
  * **PD Controller:** Implements a proportional-derivative control law where $u(t) = K_p(e_p) + K_d(e_v)$, where the error velocity is $0.0 - \text{box\_velocity\_z}$.
  * **Damping Effect:** Explores how the derivative term acts as a virtual shock absorber/viscous damper, dissipating kinetic energy and successfully settling the box at the target height without sustained oscillations.

### 7. [Lesson 7: Passive Viewer Visualization](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/07_visualization.py)
* **Script:** [07_visualization.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/07_visualization.py)
* **XML Model:** [hello_mujoco_L6.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L6.xml)
* **Focus:** Visualizing the simulation in real time using MuJoCo's interactive passive viewer.
* **Details:** Integrates real-time visual feedback with the PD control agent loop:
  * **Passive Viewer Integration:** Launches the viewer in a separate thread via `mujoco.viewer.launch_passive(model, data)` and synchronizes the physics state to the rendering pipeline at each step using `viewer.sync()`.
  * **macOS Thread Constraints:** Addresses macOS Cocoa framework limitations requiring GUI operations to run on the main thread, necessitating the use of the `mjpython` interpreter wrapper (e.g., `uv run mjpython 07_visualization.py`).
  * **Real-time Pacing:** Throttles the loop execution using dynamic sleep time based on the model's timestep option (`model.opt.timestep`) to align simulation time with real-world time.

### 8. [Lesson 8: Design Review - Strict API & Encapsulation](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/08_design_review.py)
* **Script:** [08_design_review.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/08_design_review.py)
* **XML Model:** [hello_mujoco_L6.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L6.xml)
* **Focus:** Designing a clean, encapsulated Gym-like environment wrapper to isolate the physics simulation from the control logic.
* **Details:** Refactors the simulation code into a structured Python class:
  * **Encapsulation & Security:** Implements a `BoxHoverEnv` class that hides direct access to the raw `mjData` and `mjModel` variables from the agent. The agent interacts with the simulation strictly through standard `reset()` and `step()` methods.
  * **Solving Pipeline Lag:** Explores why `mj_forward` must be called at the end of every step to sync Cartesian coordinates and sensor values immediately after `mj_step` integrates the position and velocity.
  * **Memory View Defense:** Prevents data mutation bugs by copying observation arrays (`.copy()`) rather than returning references to MuJoCo's internal memory views, protecting the agent's historical state tracking.

### 9. [Lesson 9: Contacts & Touch Sensors](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/09_contacts.py)
* **Script:** [09_contacts.py](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/lessons/09_contacts.py)
* **XML Model:** [hello_mujoco_L9.xml](file:///Users/joshuacadavez/Documents/GitHub/mujuoco-learning-lab/assets/hello_mujoco_L9.xml)
* **Focus:** Ground contact interaction and touch sensor integration.
* **Details:** Introduces collisions, contact geometry, and physical force sensing:
  * **Ground Geometry:** Declares an infinite ground `<geom name="floor" type="plane" .../>` to support collisions.
  * **Touch Sensing:** Integrates a `<site>` representing a collision zone around the box geom and attaches a `<touch>` sensor to measure normal contact forces.
  * **Force Analysis:** Steps through a free-fall collision to track and print transient high-impact forces (impact force > 20.0 N) and the resting contact force (around ~9.05 N) after the box settles on the floor.
  * **Array Extraction:** Unpacks the single-element numpy array returned by the touch sensor (`obs["touch_force"][0]`) to allow correct scalar comparison and precision formatting in Python.

---

## Running the Lessons

To run any of the lessons, navigate to the `lessons` directory:

```bash
cd lessons
```

* **Standard Script:**
  ```bash
  uv run <lesson_filename.py>
  ```
* **Interactive Viewer (macOS):**
  For scripts launching the passive viewer (like Lessons 7 and 8), macOS thread constraints require running under the `mjpython` wrapper:
  ```bash
  uv run mjpython <lesson_filename.py>
  ```