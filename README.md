# Servicing Arm Simulation Overview

This repository contains `servicing_arm_sim.py`, an educational simulation of a
three-degree-of-freedom planar servicing arm that performs a multi-phase capture
sequence. The simulator highlights the modeling, design, and analysis layers of a
cyber-physical system (CPS) workflow and mirrors many of the calculations that a
real on-orbit robotic arm would execute.

## How to Run the Simulator

```bash
python servicing_arm_sim.py [--steps N] [--dt DT] [--plots {0,1}] [--use_ros {0,1}]
```

* `--steps` (default `4000`) controls the number of discrete integration steps.
* `--dt` (default `0.005` seconds) sets the simulation time step.
* `--plots` toggles generation of tracking-error and torque plots in `out/plots/`.
* `--use_ros` enables the lightweight ROS bridge in `ros_shim.py` so external
  tools can publish target overrides or subscribe to state streams.

During runtime you can send JSON messages to the `/external/commands` topic via
ROS (or the shim) to update the target position, which lets you interactively
study off-nominal scenarios. Every run produces a JSON log at `out/latest_run.json`
that captures the state, control commands, and errors at each step.

## Simulation Flow

1. **Target model** – The target starts near `(1.0, 0.2)` meters and drifts
   slowly. Gaussian noise (`0.5 cm` σ) imitates pose sensing uncertainty.
2. **Finite-state machine (FSM)** – The mission transitions through SEARCH,
   APPROACH, ALIGN, CAPTURE, and RETREAT phases with distance and time-based
   guards. Each phase picks a different workspace goal relative to the noisy
   target estimate so you can demonstrate high-level autonomy.
3. **Planning** – A minimum-jerk interpolator produces smooth Cartesian targets
   for the end effector (EE) within each phase window, mimicking the way flight
   software enforces gentle, fuel-efficient motion profiles.
4. **Inverse kinematics (IK)** – The damped-least-squares solver converts the EE
   reference into joint-velocity commands so the 3-DOF planar arm avoids
   singularities, just as onboard guidance software regularizes IK for
   manipulators with limited reach.
5. **Joint control** – A joint-space PID controller with torque saturation sends
   commands through a second-order joint-dynamics integrator. The gains
   approximate how each joint drive on an actual arm rejects disturbances and
   respects motor limits.
6. **Analysis outputs** – After the run, plots quantify tracking error and torque
   history, while the JSON log provides per-step telemetry for offline studies.

## Connection to a Real Space Robotic Arm

Although simplified to a plane, each block mirrors a real subsystem:

* **Forward kinematics (`fk`) and Jacobian (`jacobian`)** compute the EE pose and
  sensitivity to joint motion. Real arms rely on the same trigonometric chains
  for pose estimation and momentum management.
* **IK and minimum-jerk planning** replicate how flight software converts
  supervisory goals into smooth joint references that avoid exciting structural
  modes or saturating reaction wheels.
* **PID regulation with torque limits** models the low-level joint controllers
  that keep link motion stable despite damping, flexible harnesses, or capture
  contact forces. The `K_TAU` and `BASE_DAMPING` parameters encapsulate actuator
  stiffness and passive dissipation present in hardware.
* **Finite-state autonomy** reflects mission timelines: SEARCH emulates coarse
  pointing, APPROACH mirrors guided motion near the client vehicle, ALIGN and
  CAPTURE correspond to fine alignment and latch closure, and RETREAT represents
  back-away maneuvers.
* **ROS I/O hook** stands in for spacecraft avionics or ground commanding, where
  operators can update target poses or monitor telemetry in real time.

By tuning the same parameters (link lengths, torque limits, damping, planner
horizons) that drive real mechanisms, you can explain how sensing uncertainty,
controller gains, and state-machine logic jointly determine whether a servicing
arm captures a drifting satellite.
