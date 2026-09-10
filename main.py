#!/usr/bin/env python3
"""RL-ready joint-space PD interface for Mini ODrive AT32F435, vendor FW 0.6.3.

Based on the user's stage1-v3-adjustable-pd bench controller.

Hardware model used here:
    Mini ODrive -> motor -> 15:1 gearbox -> one joint

Control architecture:
    policy action @ 50 Hz: action in [-1, +1]
            |
            v
    q_des = action * action_range
            |
            v
    joint-space PD @ 100 Hz
            |
            v
    motor torque command = joint torque / gear_ratio

Important:
- Local zero is the motor encoder position captured at reset(). It is NOT absolute homing.
- Output q/dq are motor-derived through the gear ratio. Gearbox backlash is not measured.
- The safety envelope intentionally remains conservative:
    current limit = 1 A
    watchdog = 0.5 s
    default motor torque cap = 0.036 N.m
    joint position limit = +/-60 deg
    joint speed limit = +/-180 deg/s
- No calibration and no persistent parameter save are performed.

Install:
    python -m pip install python-can

Bring SocketCAN up first, for example if your adapter setup already uses can0.

Safe hardware-interface test:
    python rl_pd_odrive.py test --node 1 --seconds 8 --pattern square \
        --action-amplitude 0.4 --kp 1.0 --kd 0.03

Import from an RL/inference program:
    from rl_pd_odrive import RLReadyPDActuator

    with RLReadyPDActuator(node=1, kp=1.0, kd=0.03) as env:
        obs = env.reset()
        while True:
            action = policy(obs)      # scalar in [-1, +1]
            obs = env.step(action)    # one 50 Hz policy period
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import signal
import struct
import time
from typing import Dict, Optional

import can

VERSION = "rl-pd-v2-strong"
VENDOR_FW = "0.6.3"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def joint_pd_to_motor_torque(
    q: float,
    dq: float,
    q_des: float,
    dq_des: float,
    kp: float,
    kd: float,
    gear_ratio: float,
    motor_torque_limit: float,
) -> tuple[float, float]:
    """Return (joint_torque_requested, motor_torque_command) in SI units."""
    joint_torque = kp * (q_des - q) + kd * (dq_des - dq)
    motor_torque = joint_torque / gear_ratio
    motor_torque = clamp(motor_torque, -motor_torque_limit, motor_torque_limit)
    return joint_torque, motor_torque


class MiniODrive:
    """Minimal CAN transport/decoder retained from the proven bench script."""

    def __init__(self, bus: can.BusABC, node: int):
        self.bus = bus
        self.node = node
        self.values: Dict[str, float | int | str] = {}
        self.seen: Dict[int, float] = {}

    def send(self, cmd: int, data: bytes = b"", remote: bool = False,
             dlc: Optional[int] = None) -> None:
        kw = dict(
            arbitration_id=(self.node << 5) | cmd,
            is_extended_id=False,
            is_remote_frame=remote,
            data=data,
        )
        if dlc is not None:
            kw["dlc"] = dlc
        self.bus.send(can.Message(**kw), timeout=0.01)

    def request(self, cmd: int) -> None:
        self.send(cmd, remote=True, dlc=8)

    def torque(self, nm: float) -> None:
        if not math.isfinite(nm):
            raise RuntimeError("Nonfinite torque")
        self.send(0x0E, struct.pack("<f", nm))

    def state(self, state: int) -> None:
        self.send(0x07, struct.pack("<I", state))

    def write_param(self, endpoint: int, raw: bytes) -> None:
        self.send(4, struct.pack("<BHB", 1, endpoint, 0) + raw)

    def read_param(self, endpoint: int, feed: bool = False) -> bytes:
        self.drain()
        self.send(4, struct.pack("<BHB", 0, endpoint, 0) + bytes(4))
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            if feed:
                self.torque(0.0)
            msg = self.bus.recv(0.01)
            if msg is None:
                continue
            self.decode(msg)
            if (
                not msg.is_extended_id
                and not msg.is_remote_frame
                and msg.arbitration_id == ((self.node << 5) | 5)
                and len(msg.data) == 8
                and struct.unpack_from("<H", msg.data, 1)[0] == endpoint
                and abs(time.time() - msg.timestamp) < 0.1
            ):
                return bytes(msg.data[4:8])
        raise RuntimeError(f"No parameter readback: {endpoint}")

    def decode(self, msg: can.Message) -> None:
        if msg.is_error_frame:
            raise RuntimeError("CAN error frame")
        if msg.is_remote_frame or msg.is_extended_id:
            return
        if msg.arbitration_id >> 5 != self.node:
            return

        age = time.time() - msg.timestamp
        if not -0.02 <= age <= 0.10:
            return

        cmd = msg.arbitration_id & 31
        data = msg.data

        if cmd == 1 and len(data) >= 8:
            self.values.update(
                axis_error=struct.unpack_from("<I", data)[0],
                axis_state=data[4],
                byte6=data[6],
            )
        elif cmd == 3 and len(data) >= 4:
            self.values["error_reply"] = struct.unpack_from("<I", data)[0]
        elif cmd == 0 and len(data) >= 8:
            self.values["version"] = ".".join(map(str, data[4:7]))
        elif cmd in (9, 0x14, 0x17) and len(data) == 8:
            a, b = struct.unpack("<ff", data)
            if not all(map(math.isfinite, (a, b))):
                raise RuntimeError("Nonfinite feedback")
            names = {
                9: ("turns", "turns_s"),
                0x14: ("iq_target", "iq"),
                0x17: ("volts", "bus_a"),
            }[cmd]
            self.values.update(zip(names, (a, b)))
        else:
            return

        self.seen[cmd] = time.monotonic() - max(0.0, age)

    def drain(self) -> None:
        for _ in range(300):
            msg = self.bus.recv(0)
            if msg is None:
                return
            self.decode(msg)
        raise RuntimeError("Excessive receive backlog")

    def fresh(self, cmd: int, seconds: float) -> bool:
        return time.monotonic() - self.seen.get(cmd, -math.inf) < seconds

    def stop(self) -> None:
        """Best effort; configured board watchdog is the final lost-PC guard."""
        for cmd, data in [
            (0x0E, struct.pack("<f", 0.0)),
            (0x07, struct.pack("<I", 1)),
        ]:
            try:
                self.send(cmd, data)
            except Exception as exc:  # emergency path: report, do not mask original fault
                print("STOP SEND FAILED:", exc)


class RLReadyPDActuator:
    """One-DOF RL actuator environment with normalized position actions.

    Public API:
        reset() -> observation dict
        step(action) -> observation dict
        observe() -> observation dict
        close()

    `step(action)` consumes exactly one policy period. With the defaults this
    means one RL action at 50 Hz and two inner PD updates at 100 Hz.
    """

    def __init__(
        self,
        channel: str = "can0",
        node: int = 1,
        gear_ratio: float = 15.0,
        kp: float = 1.0,
        kd: float = 0.03,
        action_range_deg: float = 50.0,
        control_hz: float = 100.0,
        policy_hz: float = 50.0,
        motor_torque_limit: float = 0.036,
        joint_position_limit_deg: float = 60.0,
        joint_speed_limit_deg_s: float = 180.0,
        measured_current_limit_a: float = 1.10,
        log_csv: Optional[str] = None,
    ):
        self.channel = channel
        self.node = node
        self.gear_ratio = gear_ratio
        self.kp = kp
        self.kd = kd
        self.action_range_rad = math.radians(action_range_deg)
        self.control_hz = control_hz
        self.policy_hz = policy_hz
        self.motor_torque_limit_requested = motor_torque_limit
        self.joint_position_limit_rad = math.radians(joint_position_limit_deg)
        self.joint_speed_limit_rad_s = math.radians(joint_speed_limit_deg_s)
        self.measured_current_limit_a = measured_current_limit_a
        self.log_csv = log_csv

        self._validate_config()

        self.inner_steps = int(round(self.control_hz / self.policy_hz))
        self.control_period = 1.0 / self.control_hz

        self.bus: Optional[can.BusABC] = None
        self.dev: Optional[MiniODrive] = None
        self.armed = False
        self.started = False

        self.p0_turns = 0.0
        self.baseline_byte6 = 0
        self.kt = 0.0
        self.motor_torque_limit = motor_torque_limit
        self.prev_action = 0.0
        self.q_des = 0.0
        self.last_joint_torque_requested = 0.0
        self.last_motor_torque_cmd = 0.0

        self.start_time = 0.0
        self.previous_tick = 0.0
        self.next_tick = 0.0
        self.next_diag = 0.0
        self.next_slow = 0.0

        self._log_file = None
        self._log_writer = None

    def _validate_config(self) -> None:
        if not 0 <= self.node <= 62:
            raise ValueError("node must be 0..62")
        if not math.isfinite(self.gear_ratio) or self.gear_ratio <= 0:
            raise ValueError("gear_ratio must be > 0")
        if not math.isfinite(self.kp) or not 0 < self.kp <= 50.0:
            raise ValueError("kp must be >0 and <=50 Nm/rad")
        if not math.isfinite(self.kd) or not 0 <= self.kd <= 0.3:
            raise ValueError("kd must be >=0 and <=0.3 Nm/(rad/s)")
        if not 0 < math.degrees(self.action_range_rad) <= 60.0:
            raise ValueError("action_range_deg must be >0 and <=60")
        if not 20 <= self.control_hz <= 500:
            raise ValueError("control_hz must be in [20, 500]")
        if not 1 <= self.policy_hz <= self.control_hz:
            raise ValueError("policy_hz must be <= control_hz")
        ratio = self.control_hz / self.policy_hz
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("control_hz / policy_hz must be an integer")
        if not 0 < self.motor_torque_limit_requested <= 0.040:
            raise ValueError("motor_torque_limit must be >0 and <=0.040 N.m with the 1 A driver limit")
        if not 0 < math.degrees(self.joint_position_limit_rad) <= 90.0:
            raise ValueError("joint_position_limit_deg must be >0 and <=90")
        if not 0 < math.degrees(self.joint_speed_limit_rad_s) <= 360.0:
            raise ValueError("joint_speed_limit_deg_s must be >0 and <=360")
        if not 0 < self.measured_current_limit_a <= 1.2:
            raise ValueError("measured_current_limit_a must be >0 and <=1.2")

    def __enter__(self) -> "RLReadyPDActuator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reset(self) -> dict:
        """Initialize/arm safely and capture local joint zero."""
        if self.started:
            raise RuntimeError("Already started; close() before reset() again")

        self.bus = can.Bus(
            interface="socketcan",
            channel=self.channel,
            can_filters=[{
                "can_id": self.node << 5,
                "can_mask": 0x7E0,
                "extended": False,
            }],
        )
        self.dev = MiniODrive(self.bus, self.node)

        try:
            print(VERSION, flush=True)
            self.dev.request(0)

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                self.dev.drain()
                if self.dev.values.get("version") is not None:
                    break
                time.sleep(0.01)

            if self.dev.values.get("version") != VENDOR_FW:
                raise RuntimeError(
                    f"Parameter mapping requires vendor firmware {VENDOR_FW}; "
                    f"got {self.dev.values.get('version')}"
                )

            # From this point onward every failure attempts zero torque + IDLE.
            self.armed = True
            self.dev.state(1)

            # Temporarily disable watchdog only during controlled setup.
            self.dev.write_param(56, bytes(4))
            if self.dev.read_param(56)[0] != 0:
                raise RuntimeError("Could not disable watchdog for setup")

            # Clear old errors exactly once, never during active running.
            self.dev.send(0x18)
            wait_until = time.monotonic() + 0.25
            while time.monotonic() < wait_until:
                self.dev.drain()
                time.sleep(0.01)

            for cmd in (1, 3, 9, 0x14, 0x17):
                self.dev.seen.pop(cmd, None)

            self._wait_for_initial_feedback()
            print("Initial state:", self.dev.values)

            v = self.dev.values
            self.p0_turns = float(v["turns"])
            self.baseline_byte6 = int(v["byte6"])

            if v["axis_error"] or v["error_reply"]:
                raise RuntimeError("Driver still reports an error after startup clear")
            if v["axis_state"] != 1:
                raise RuntimeError("Begin in IDLE")
            if abs(float(v["turns_s"])) > 0.05:
                raise RuntimeError("Joint must be stationary at startup")
            if self.dev.read_param(198)[0]:
                raise RuntimeError("System error is nonzero")

            self.kt = struct.unpack("<f", self.dev.read_param(126))[0]
            if not math.isfinite(self.kt) or not 0 < self.kt <= 0.1:
                raise RuntimeError(f"Invalid configured Kt: {self.kt}")

            # Preserve the original conservative torque rule as a second cap.
            self.motor_torque_limit = min(
                self.motor_torque_limit_requested,
                0.95 * self.kt,
            )

            self.dev.torque(0.0)

            # Preserve the bench script's temporary 2 turns/s velocity limit and
            # 1 A current limit.
            self.dev.send(0x0F, struct.pack("<ff", 2.0, 1.0))

            # Torque control + passthrough input mode, matching the bench script.
            self.dev.send(0x0B, struct.pack("<II", 1, 1))

            current_limit = struct.unpack("<f", self.dev.read_param(122))[0]
            if not math.isclose(current_limit, 1.0, abs_tol=0.01):
                raise RuntimeError(f"1 A limit not confirmed: {current_limit}")

            self.dev.write_param(55, struct.pack("<f", 0.5))
            watchdog_timeout = struct.unpack("<f", self.dev.read_param(55))[0]
            if not math.isclose(watchdog_timeout, 0.5, abs_tol=1e-5):
                raise RuntimeError("Watchdog timeout readback mismatch")

            self.dev.torque(0.0)
            self.dev.write_param(56, struct.pack("<I", 1))
            if self.dev.read_param(56, feed=True)[0] != 1:
                raise RuntimeError("Watchdog enable readback mismatch")

            print(
                f"Kt={self.kt:.5f} N.m/A; current limit=1 A; "
                f"motor torque cap={self.motor_torque_limit:.4f} N.m; "
                "watchdog=0.5s ON",
                flush=True,
            )

            self.dev.state(8)
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                self.dev.torque(0.0)
                time.sleep(0.01)
                self.dev.drain()
                if self.dev.values.get("axis_error", 0):
                    raise RuntimeError("Error on enable")
                if self.dev.values.get("axis_state") == 8:
                    break

            if self.dev.values.get("axis_state") != 8:
                raise RuntimeError(
                    "Closed loop not confirmed; calibration may be needed"
                )

            self.prev_action = 0.0
            self.q_des = 0.0
            self.last_joint_torque_requested = 0.0
            self.last_motor_torque_cmd = 0.0

            now = time.monotonic()
            self.start_time = now
            self.previous_tick = now
            self.next_tick = now
            self.next_diag = now
            self.next_slow = now
            self.started = True

            self._open_log()

            print(
                f"RL interface: policy={self.policy_hz:g} Hz, "
                f"PD={self.control_hz:g} Hz, inner_steps={self.inner_steps}, "
                f"action_range=+/-{math.degrees(self.action_range_rad):g} deg",
                flush=True,
            )
            print("Local zero captured. q/dq are motor-derived through the gearbox.")

            return self.observe()
        except Exception:
            self.close()
            raise

    def _wait_for_initial_feedback(self) -> None:
        assert self.dev is not None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            for cmd in (3, 0x14, 0x17):
                self.dev.request(cmd)
            time.sleep(0.05)
            self.dev.drain()
            if all(self.dev.fresh(c, 0.3) for c in (1, 3, 9, 0x14, 0x17)):
                return
        raise RuntimeError("Missing fresh heartbeat/encoder/diagnostic feedback")

    def _service_feedback(self, now: float) -> None:
        assert self.dev is not None
        self.dev.drain()

        if now >= self.next_diag:
            self.dev.request(0x14)
            self.next_diag = now + 0.05

        if now >= self.next_slow:
            self.dev.request(3)
            self.dev.request(0x17)
            self.next_slow = now + 0.5

    def _joint_state(self) -> tuple[float, float]:
        assert self.dev is not None
        v = self.dev.values
        q = (float(v["turns"]) - self.p0_turns) * 2.0 * math.pi / self.gear_ratio
        dq = float(v["turns_s"]) * 2.0 * math.pi / self.gear_ratio
        return q, dq

    def _safety_check(self, dt: float, q: float, dq: float) -> None:
        assert self.dev is not None
        v = self.dev.values

        if dt > 0.05:
            raise RuntimeError(f"Control-loop stall >50 ms ({dt:.4f}s)")
        if not self.dev.fresh(9, 0.10) or not self.dev.fresh(1, 0.35):
            raise RuntimeError("Encoder or heartbeat timeout")
        if not self.dev.fresh(0x14, 0.20) or not self.dev.fresh(3, 1.0):
            raise RuntimeError("Current/error feedback timeout")
        if v.get("axis_error", 0) or v.get("error_reply", 0):
            raise RuntimeError("Driver fault")
        if v.get("axis_state") != 8:
            raise RuntimeError("Unexpected axis state")
        if int(v.get("byte6", -1)) != self.baseline_byte6:
            raise RuntimeError("Undocumented heartbeat byte6 changed")
        if abs(q) > self.joint_position_limit_rad:
            raise RuntimeError("Joint position limit exceeded")
        if abs(dq) > self.joint_speed_limit_rad_s:
            raise RuntimeError("Joint speed limit exceeded")
        if abs(float(v.get("iq", 0.0))) > self.measured_current_limit_a:
            raise RuntimeError("Measured current limit exceeded")

    def observe(self) -> dict:
        """Return latest hardware state without sending a new action."""
        if not self.started or self.dev is None:
            raise RuntimeError("Call reset() first")

        self.dev.drain()
        q, dq = self._joint_state()
        v = self.dev.values
        return {
            "q": q,
            "dq": dq,
            "q_deg": math.degrees(q),
            "dq_deg_s": math.degrees(dq),
            "q_des": self.q_des,
            "q_des_deg": math.degrees(self.q_des),
            "action": self.prev_action,
            "joint_torque_requested": self.last_joint_torque_requested,
            "motor_torque_cmd": self.last_motor_torque_cmd,
            "iq": float(v.get("iq", math.nan)),
            "iq_target": float(v.get("iq_target", math.nan)),
            "bus_voltage": float(v.get("volts", math.nan)),
            "bus_current": float(v.get("bus_a", math.nan)),
            "axis_state": int(v.get("axis_state", -1)),
            "axis_error": int(v.get("axis_error", -1)),
            "t_s": time.monotonic() - self.start_time,
        }

    def observation_vector(self, target_rad: float = 0.0) -> list[float]:
        """Convenience vector for an RL policy: [q, dq, target, previous_action]."""
        obs = self.observe()
        return [
            float(obs["q"]),
            float(obs["dq"]),
            float(target_rad),
            float(obs["action"]),
        ]

    def step(self, action: float) -> dict:
        """Execute one normalized RL action for one policy period.

        action=-1 -> -action_range
        action= 0 -> local zero
        action=+1 -> +action_range

        The action is held constant while the 100 Hz inner PD loop executes.
        """
        if not self.started or self.dev is None:
            raise RuntimeError("Call reset() first")
        if not math.isfinite(action):
            raise ValueError("action must be finite")

        action = clamp(float(action), -1.0, 1.0)
        self.prev_action = action
        self.q_des = action * self.action_range_rad
        dq_des = 0.0

        for _ in range(self.inner_steps):
            self._pd_tick(self.q_des, dq_des)

        return self.observe()

    def _pd_tick(self, q_des: float, dq_des: float) -> None:
        assert self.dev is not None

        # Absolute-period scheduling: late cycles never generate catch-up bursts.
        self.next_tick += self.control_period

        now = time.monotonic()
        dt = now - self.previous_tick
        self.previous_tick = now

        self._service_feedback(now)
        q, dq = self._joint_state()
        self._safety_check(dt, q, dq)

        joint_tau_req, motor_tau_cmd = joint_pd_to_motor_torque(
            q=q,
            dq=dq,
            q_des=q_des,
            dq_des=dq_des,
            kp=self.kp,
            kd=self.kd,
            gear_ratio=self.gear_ratio,
            motor_torque_limit=self.motor_torque_limit,
        )

        self.last_joint_torque_requested = joint_tau_req
        self.last_motor_torque_cmd = motor_tau_cmd
        self.dev.torque(motor_tau_cmd)

        self._write_log(now, dt, q, dq)

        delay = self.next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            # Resynchronize; do not send multiple torque commands back-to-back.
            self.next_tick = time.monotonic()

    def _open_log(self) -> None:
        if not self.log_csv:
            return

        path = Path(self.log_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = path.open("w", newline="")
        fields = [
            "t_s",
            "dt_s",
            "action",
            "q_des_rad",
            "q_des_deg",
            "q_rad",
            "q_deg",
            "dq_rad_s",
            "dq_deg_s",
            "joint_torque_requested",
            "motor_torque_cmd",
            "iq_target",
            "iq",
            "volts",
            "bus_a",
            "axis_state",
            "axis_error",
            "encoder_age_s",
            "iq_age_s",
            "kp",
            "kd",
            "gear_ratio",
        ]
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=fields)
        self._log_writer.writeheader()
        self._log_file.flush()
        print("Logging:", path)

    def _write_log(self, now: float, dt: float, q: float, dq: float) -> None:
        if self._log_writer is None or self.dev is None:
            return

        v = self.dev.values
        self._log_writer.writerow({
            "t_s": now - self.start_time,
            "dt_s": dt,
            "action": self.prev_action,
            "q_des_rad": self.q_des,
            "q_des_deg": math.degrees(self.q_des),
            "q_rad": q,
            "q_deg": math.degrees(q),
            "dq_rad_s": dq,
            "dq_deg_s": math.degrees(dq),
            "joint_torque_requested": self.last_joint_torque_requested,
            "motor_torque_cmd": self.last_motor_torque_cmd,
            "iq_target": v.get("iq_target", ""),
            "iq": v.get("iq", ""),
            "volts": v.get("volts", ""),
            "bus_a": v.get("bus_a", ""),
            "axis_state": v.get("axis_state", ""),
            "axis_error": v.get("axis_error", ""),
            "encoder_age_s": time.monotonic() - self.dev.seen.get(9, -math.inf),
            "iq_age_s": time.monotonic() - self.dev.seen.get(0x14, -math.inf),
            "kp": self.kp,
            "kd": self.kd,
            "gear_ratio": self.gear_ratio,
        })
        self._log_file.flush()

    def close(self) -> None:
        """Immediately request zero torque + IDLE and release SocketCAN."""
        try:
            if self.dev is not None and self.armed:
                self.dev.stop()
                print("Zero torque and IDLE requested. Confirm the arm releases.")
        finally:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
            self._log_file = None
            self._log_writer = None

            if self.bus is not None:
                try:
                    self.bus.shutdown()
                except Exception:
                    pass

            self.bus = None
            self.dev = None
            self.started = False
            self.armed = False


def scripted_action(pattern: str, t: float, amplitude: float,
                    frequency_hz: float) -> float:
    """Deterministic action generator using the exact same step(action) RL API."""
    if pattern == "zero":
        return 0.0
    if pattern == "step":
        # First second at zero, then positive amplitude.
        return 0.0 if t < 1.0 else amplitude
    if pattern == "square":
        # First second at zero, then alternate sign each second.
        if t < 1.0:
            return 0.0
        return amplitude if int(t - 1.0) % 2 == 0 else -amplitude
    if pattern == "sine":
        return amplitude * math.sin(2.0 * math.pi * frequency_hz * t)
    raise ValueError(pattern)


def run_test(args: argparse.Namespace) -> None:
    log_path = args.log
    if log_path == "auto":
        Path("logs").mkdir(exist_ok=True)
        log_path = str(Path("logs") / f"rl_pd_{time.time_ns()}.csv")

    with RLReadyPDActuator(
        channel=args.channel,
        node=args.node,
        gear_ratio=args.gear_ratio,
        kp=args.kp,
        kd=args.kd,
        action_range_deg=args.action_range_deg,
        control_hz=args.control_hz,
        policy_hz=args.policy_hz,
        motor_torque_limit=args.motor_torque_limit,
        joint_position_limit_deg=args.joint_position_limit_deg,
        joint_speed_limit_deg_s=args.joint_speed_limit_deg_s,
        measured_current_limit_a=args.measured_current_limit_a,
        log_csv=log_path,
    ) as actuator:
        actuator.reset()

        start = time.monotonic()
        next_print = start
        while True:
            t = time.monotonic() - start
            if t >= args.seconds:
                break

            action = scripted_action(
                args.pattern,
                t,
                args.action_amplitude,
                args.frequency_hz,
            )
            obs = actuator.step(action)

            now = time.monotonic()
            if now >= next_print:
                print(
                    f"t={obs['t_s']:6.2f}s  "
                    f"a={obs['action']:+.3f}  "
                    f"q_des={obs['q_des_deg']:+6.2f} deg  "
                    f"q={obs['q_deg']:+7.2f} deg  "
                    f"dq={obs['dq_deg_s']:+7.2f} deg/s  "
                    f"Iq={obs['iq']:+.3f} A  "
                    f"tau_m={obs['motor_torque_cmd']:+.4f} N.m"
                )
                next_print = now + 0.25


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RL-ready Mini ODrive 15:1 joint-space PD controller"
    )
    p.add_argument("mode", choices=["test"], help="scripted test through RL step(action)")
    p.add_argument("--channel", default="can0")
    p.add_argument("--node", type=int, default=1)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--gear-ratio", type=float, default=15.0)
    p.add_argument("--kp", type=float, default=8.0, help="joint Nm/rad, max 50")
    p.add_argument("--kd", type=float, default=0.08, help="joint Nm/(rad/s), max 0.3")
    p.add_argument("--action-range-deg", type=float, default=50.0,
                   help="action +/-1 maps to +/- this many output degrees, max 60")
    p.add_argument("--control-hz", type=float, default=100.0)
    p.add_argument("--policy-hz", type=float, default=50.0)
    p.add_argument("--motor-torque-limit", type=float, default=0.036,
                   help="motor-side N.m cap; 0.036 ~= 0.9 A for Kt=0.04")
    p.add_argument("--joint-position-limit-deg", type=float, default=60.0,
                   help="hard software stop relative to startup zero")
    p.add_argument("--joint-speed-limit-deg-s", type=float, default=180.0,
                   help="hard software speed stop")
    p.add_argument("--measured-current-limit-a", type=float, default=1.10,
                   help="software measured-Iq stop; ODrive current limit remains 1 A")
    p.add_argument("--pattern", choices=["zero", "step", "square", "sine"],
                   default="square")
    p.add_argument("--action-amplitude", type=float, default=0.4,
                   help="normalized scripted action magnitude, 0..1")
    p.add_argument("--frequency-hz", type=float, default=0.2,
                   help="used by sine pattern")
    p.add_argument("--log", default="auto",
                   help="CSV path, 'auto' for logs/rl_pd_*.csv, or '' for no log")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        p.error("--seconds must be >0 and <=60 for the initial hardware tests")
    if not math.isfinite(args.action_amplitude) or not 0 <= args.action_amplitude <= 1:
        p.error("--action-amplitude must be in [0,1]")
    if not math.isfinite(args.frequency_hz) or args.frequency_hz <= 0:
        p.error("--frequency-hz must be >0")

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)

    try:
        if args.mode == "test":
            run_test(args)
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as exc:
        p.exit(1, f"STOP: {exc}\n")


if __name__ == "__main__":
    main()
