"""servicing_arm_sim.py
=======================
This educational simulator illustrates the CPS workflow for an ECE 830
space-servicing arm demonstration. The **modeling** layer includes
forward kinematics, Jacobian-based inverse kinematics, and a discrete
joint-dynamics integrator for a 3-DOF planar arm. The **design** layer
adds cyber controls: a finite-state machine (FSM), a minimum-jerk
planner, and a joint-space PID regulator augmented by optional ROS I/O
shims. The **analysis** layer captures run logs and produces plots that
quantify tracking error and control effort. The overall structure yields
an executable narrative suitable for a 20-minute CPS presentation: start
from modeling assumptions, walk through the supervisory control design,
and finish with analysis artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import select
import sys

try:
    import termios
    import tty
except Exception:  # pragma: no cover - optional keyboard support
    termios = None  # type: ignore
    tty = None  # type: ignore

try:  # Optional ROS shim
    import ros_shim as rospy  # type: ignore

    ROS_AVAILABLE = True
except SystemExit:
    # ros_shim exits when the remote simulator is absent; treat as offline
    ROS_AVAILABLE = False
except Exception:
    ROS_AVAILABLE = False

LINK_LENGTHS = np.array([0.5, 0.5, 0.4])
K_TAU = 2.0
BASE_DAMPING = 0.2
PID_GAINS = {"Kp": 6.0, "Kd": 0.3, "Ki": 0.0}
TORQUE_LIMIT = 5.0
LAM_DLS = 0.02
TARGET_NOISE_STD = 0.005
HOME_POSITION = np.array([0.5, 0.0])


class RosBridge:
    """Thin helper around ``ros_shim.py`` for publishing telemetry and overrides."""

    def __init__(self, requested: bool):
        self.requested = bool(requested)
        self.enabled = bool(requested and ROS_AVAILABLE)
        self.publishers: Dict[str, object] = {}
        self._target_override: Optional[np.ndarray] = None

    def start(self) -> None:
        if not self.enabled:
            if self.requested and not ROS_AVAILABLE:
                print("ROS shim unavailable; running offline.")
            return
        try:
            rospy.init_node("servicing_arm")
            self.publishers = {
                "state": rospy.Publisher("/arm/state", rospy.std_msgs.String),
                "cmd": rospy.Publisher("/arm/cmd", rospy.std_msgs.String),
                "target": rospy.Publisher("/target/pose", rospy.std_msgs.String),
            }

            def _override_cb(msg: str) -> None:
                try:
                    data = json.loads(msg)
                    if "target" in data:
                        self._target_override = np.array(data["target"], dtype=float)
                except Exception:
                    pass

            rospy.Subscriber("/external/commands", rospy.std_msgs.String, _override_cb)
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"ROS bridge disabled: {exc}")
            self.enabled = False

    @property
    def target_override(self) -> Optional[np.ndarray]:
        if self._target_override is None:
            return None
        return self._target_override.copy()

    def publish(self, q: np.ndarray, qd: np.ndarray, x: np.ndarray, tau: np.ndarray, target: np.ndarray, phase: str) -> None:
        if not self.enabled:
            return
        try:
            self.publishers["state"].publish(
                json.dumps({"q": q.tolist(), "qd": qd.tolist(), "x": x.tolist(), "phase": phase})
            )
            self.publishers["cmd"].publish(json.dumps({"tau": tau.tolist()}))
            self.publishers["target"].publish(json.dumps({"x_t": target.tolist()}))
        except Exception:
            self.enabled = False


class SimState:
    """Track interactive target state and provide non-blocking key polling."""

    def __init__(self, x_t0: np.ndarray, step: float = 0.02):
        self.x_t = x_t0.copy()
        self.manual_override = False
        self._step = step
        self._stdin_fd: Optional[int] = None
        self._orig_termios = None
        self.enabled = bool(
            termios is not None and tty is not None and sys.stdin.isatty()
        )
        if self.enabled:
            try:
                self._stdin_fd = sys.stdin.fileno()
                self._orig_termios = termios.tcgetattr(self._stdin_fd)
                tty.setcbreak(self._stdin_fd)
            except Exception:
                self.enabled = False
                self._stdin_fd = None
                self._orig_termios = None

    def close(self) -> None:
        if self.enabled and self._stdin_fd is not None and self._orig_termios is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._orig_termios)
        self.enabled = False

    def __enter__(self) -> "SimState":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def _input_ready(self) -> bool:
        if not self.enabled:
            return False
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(rlist)

    def _delta_for_key(self, key: str) -> Optional[np.ndarray]:
        mapping = {
            "w": np.array([0.0, self._step]),
            "s": np.array([0.0, -self._step]),
            "a": np.array([-self._step, 0.0]),
            "d": np.array([self._step, 0.0]),
        }
        key_lower = key.lower()
        if key_lower in mapping:
            return mapping[key_lower]
        if key_lower == "r":
            self.manual_override = False
        return None

    def poll_keyboard(self) -> None:
        if not self.enabled:
            return
        while self._input_ready():
            ch = sys.stdin.read(1)
            if not ch:
                break
            delta = self._delta_for_key(ch)
            if delta is not None:
                self.x_t = self.x_t + delta
                self.manual_override = True

    def current_target(self, nominal: np.ndarray) -> np.ndarray:
        if not self.manual_override:
            self.x_t = nominal.copy()
        return self.x_t.copy()


def fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics for planar arm returning the end-effector (x, y)."""

    c = np.cumsum(q)
    x = np.sum(LINK_LENGTHS * np.cos(c))
    y = np.sum(LINK_LENGTHS * np.sin(c))
    return np.array([x, y])


def jacobian(q: np.ndarray) -> np.ndarray:
    """Compute the 2x3 Jacobian relating joint velocities to EE velocity."""

    c = np.cumsum(q)
    J = np.zeros((2, 3))
    for i in range(3):
        sin_terms = np.sin(c[i:])
        cos_terms = np.cos(c[i:])
        J[0, i] = -np.sum(LINK_LENGTHS[i:] * sin_terms)
        J[1, i] = np.sum(LINK_LENGTHS[i:] * cos_terms)
    return J


def min_jerk(x0: np.ndarray, x1: np.ndarray, t: float, T: float) -> np.ndarray:
    """Minimum-jerk interpolation between x0 and x1 over duration T."""

    if T <= 0:
        return x1.copy()
    tau = np.clip(t / T, 0.0, 1.0)
    s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
    return x0 + s * (x1 - x0)


def fsm_next(
    phase: str,
    x: np.ndarray,
    x_t: np.ndarray,
    t: float,
    t0: float,
    thresholds: Dict[str, float],
) -> Tuple[str, float, bool]:
    """Determine the next FSM phase.

    Returns tuple of (next_phase, new_phase_start_time, capture_event).
    """

    dist = np.linalg.norm(x - x_t)
    next_phase = phase
    new_t0 = t0
    capture_event = False

    if phase == "SEARCH" and dist < thresholds["search_dist"]:
        next_phase = "APPROACH"
    elif phase == "APPROACH" and dist < thresholds["approach_dist"]:
        next_phase = "ALIGN"
    elif phase == "ALIGN" and (t - t0) > thresholds["align_time"]:
        next_phase = "CAPTURE"
    elif phase == "CAPTURE" and (t - t0) > thresholds["capture_time"]:
        next_phase = "RETREAT"

    if next_phase != phase:
        new_t0 = t
        capture_event = next_phase == "CAPTURE"

    return next_phase, new_t0, capture_event


def ik_dls(q: np.ndarray, x_des: np.ndarray, dt: float, lam: float) -> np.ndarray:
    """Damped least squares IK velocity command."""

    x = fk(q)
    v = (x_des - x) / max(dt, 1e-3)
    J = jacobian(q)
    JJt = J @ J.T
    inv = np.linalg.inv(JJt + (lam ** 2) * np.eye(2))
    dq = J.T @ (inv @ v)
    return dq


def pid_step(
    q: np.ndarray,
    qd: np.ndarray,
    q_des: np.ndarray,
    qd_des: np.ndarray,
    Ki_state: np.ndarray,
    dt: float,
    gains: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """One step of joint-space PID control."""

    err = q_des - q
    derr = qd_des - qd
    Ki_state = Ki_state + err * dt
    tau = gains["Kp"] * err + gains["Kd"] * derr + gains["Ki"] * Ki_state
    return tau, Ki_state


def save_json(log_dict: Dict, path: str) -> None:
    """Save JSON logs to disk."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2)


def make_plots(log_dict: Dict, outdir: str) -> None:
    """Create tracking-error and torque plots."""

    os.makedirs(outdir, exist_ok=True)
    t = [entry["t"] for entry in log_dict["records"]]
    err = [entry["error"] for entry in log_dict["records"]]
    torque = [entry["tau"] for entry in log_dict["records"]]
    torque = np.array(torque)

    plt.figure()
    plt.plot(t, err)
    plt.xlabel("Time [s]")
    plt.ylabel("Tracking error [m]")
    plt.title("EE Tracking Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "error.png"))
    plt.close()

    plt.figure()
    plt.plot(t, torque)
    plt.xlabel("Time [s]")
    plt.ylabel("Joint torque [Nm]")
    plt.title("Joint Torque Commands")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "torque.png"))
    plt.close()


def simulate(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    """Run the servicing arm simulation and return logs plus summary info."""

    dt = float(args.dt)
    steps = int(args.steps)
    use_ros = bool(args.use_ros)
    rng = np.random.default_rng(42)

    ros_bridge = RosBridge(use_ros)
    ros_bridge.start()

    q = np.zeros(3)
    qd = np.zeros(3)
    q_des = q.copy()
    qd_des = np.zeros(3)
    Ki_state = np.zeros(3)
    damping = BASE_DAMPING
    capture_flag = False

    target_origin = np.array([1.0, 0.2])
    target_drift = np.array([-0.01, 0.0])

    phase = "SEARCH"
    phase_t0 = 0.0
    phase_durations = {
        "SEARCH": 5.0,
        "APPROACH": 4.0,
        "ALIGN": 2.0,
        "CAPTURE": 1.5,
        "RETREAT": 3.0,
    }
    thresholds = {
        "search_dist": 0.6,
        "approach_dist": 0.15,
        "align_time": 1.0,
        "capture_time": 0.8,
    }

    x_ref = HOME_POSITION.copy()
    phase_params = {
        "start_time": 0.0,
        "duration": phase_durations[phase],
        "x_start": x_ref.copy(),
        "x_goal": HOME_POSITION.copy(),
    }

    def phase_goal(current_phase: str, target_est: np.ndarray) -> np.ndarray:
        offsets = {
            "SEARCH": np.array([-0.3, 0.3]),
            "APPROACH": np.array([-0.1, 0.05]),
            "ALIGN": np.zeros(2),
            "CAPTURE": np.zeros(2),
            "RETREAT": np.array([-0.4, 0.4]),
        }
        if current_phase == "SEARCH":
            return target_est + offsets["SEARCH"]
        if current_phase == "RETREAT":
            return target_est + offsets["RETREAT"]
        return target_est + offsets.get(current_phase, np.zeros(2))

    phase_params["x_goal"] = phase_goal(phase, target_origin)

    log_records: List[Dict] = []
    sim_state = SimState(target_origin)

    try:
        for step in range(steps):
            t = step * dt

            sim_state.poll_keyboard()

            target_nominal = target_origin + target_drift * t
            target_true = sim_state.current_target(target_nominal)
            if ros_bridge.target_override is not None:
                target_true = ros_bridge.target_override
            target_meas = target_true + rng.normal(0.0, TARGET_NOISE_STD, size=2)

            phase_elapsed = t - phase_params["start_time"]
            T = phase_params["duration"]
            x_ref = min_jerk(
                phase_params["x_start"], phase_params["x_goal"], phase_elapsed, T
            )

            dq = ik_dls(q_des, x_ref, dt, LAM_DLS)
            q_des = q_des + dq * dt
            qd_des = dq

            tau, Ki_state = pid_step(q, qd, q_des, qd_des, Ki_state, dt, PID_GAINS)
            tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)

            qdd = K_TAU * tau - damping * qd
            qd = qd + qdd * dt
            q = q + qd * dt
            x = fk(q)

            next_phase, new_t0, capture_event = fsm_next(
                phase, x, target_meas, t, phase_t0, thresholds
            )
            if next_phase != phase:
                phase = next_phase
                phase_t0 = new_t0
                phase_params = {
                    "start_time": t,
                    "duration": phase_durations.get(phase, 2.0),
                    "x_start": x_ref.copy(),
                    "x_goal": phase_goal(phase, target_meas),
                }
            if capture_event and not capture_flag:
                damping += 0.2
                capture_flag = True

            ros_bridge.publish(q, qd, x, tau, target_meas, phase)

            error = float(np.linalg.norm(x - target_meas))
            log_records.append(
                {
                    "t": t,
                    "phase": phase,
                    "q": q.tolist(),
                    "qd": qd.tolist(),
                    "x": x.tolist(),
                    "target": target_meas.tolist(),
                    "tau": tau.tolist(),
                    "error": error,
                }
            )
    finally:
        sim_state.close()

    final_info = {
        "final_x": x.tolist(),
        "final_error": error,
        "phase": phase,
        "capture": capture_flag,
        "steps": steps,
        "dt": dt,
    }

    log_dict = {"records": log_records, "meta": final_info}
    return log_dict, final_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Space servicing arm simulator")
    parser.add_argument("--steps", type=int, default=4000, help="Number of integration steps")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step [s]")
    parser.add_argument("--plots", type=int, default=1, help="Generate plots (1/0)")
    parser.add_argument("--use_ros", type=int, default=0, help="Enable ROS shim (1/0)")
    args = parser.parse_args()

    log_dict, final_info = simulate(args)
    out_json = os.path.join("out", "latest_run.json")
    save_json(log_dict, out_json)

    if args.plots:
        make_plots(log_dict, os.path.join("out", "plots"))

    print("Final EE position:", final_info["final_x"])
    print("Final error:", final_info["final_error"])
    print("Final FSM phase:", final_info["phase"])
    print("Capture occurred:", final_info["capture"])
    print("Log saved to:", out_json)
    if args.plots:
        print("Plots saved to:", os.path.join("out", "plots"))


if __name__ == "__main__":
    main()
