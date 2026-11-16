"""Digital PID from PID.cpp.

One iteration of ``PID::operator()``:

1. ``error = setpoint - measurement``, cleared when ``|error| < dead_zone``
2. ``acc = clamp(error + (1 - smoothing) * acc, ±anti_windup)``
3. ``p = Kp * error``, ``i = acc * dt * Ki``, ``d = Kd * (error - last_error) / dt``
4. ``out`` is lifted to ``±min_output`` and clipped to ``±max_output``

The first sample takes ``last_error = 0``, so the derivative sees the whole error.
"""

from dataclasses import dataclass, field, replace

import control
import numpy as np

from plantid.plant import initial_state, step

DT = 15e-3
ENCODER_STEP = 360.0 / 2048.0


@dataclass(frozen=True)
class Gains:
    """Proportional, integral and derivative gains."""

    kp: float
    ki: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True)
class Limits:
    """``PID_Config`` fields other than the three gains."""

    max_output: float = 100.0
    min_output: float = 0.0
    anti_windup: float = 100.0
    dead_zone: float = 0.0
    smoothing: float = 0.0


@dataclass
class FirmwarePID:
    """Stateful copy of ``kalman_electronics::regulators::PID``."""

    gains: Gains
    limits: Limits = field(default_factory=Limits)
    acc: float = 0.0
    last_error: float = 0.0

    def reset(self) -> None:
        self.acc = 0.0
        self.last_error = 0.0

    def step(self, setpoint: float, measurement: float, dt: float) -> float:
        error = float(setpoint - measurement)
        if abs(error) < self.limits.dead_zone:
            error = 0.0
        acc = error + (1.0 - self.limits.smoothing) * self.acc
        acc = float(np.clip(acc, -self.limits.anti_windup, self.limits.anti_windup))
        d_term = 0.0 if dt <= 0.0 else self.gains.kd * (error - self.last_error) / dt
        out = self.gains.kp * error + acc * dt * self.gains.ki + d_term
        if abs(out) < self.limits.min_output:
            out = float(np.sign(out) * self.limits.min_output)
        if abs(out) > self.limits.max_output:
            out = float(np.sign(out) * self.limits.max_output)
        self.acc = acc
        self.last_error = error
        return out


class Trajectory(tuple):
    """``(y, u)`` from :func:`simulate`. Unpacks like a pair."""

    def __new__(cls, y: np.ndarray, u: np.ndarray):
        return super().__new__(cls, (y, u))

    @property
    def y(self) -> np.ndarray:
        return self[0]

    @property
    def u(self) -> np.ndarray:
        return self[1]


def steps(levels, hold: float, dt: float) -> np.ndarray:
    """Piecewise-constant reference. Each level is held for ``hold`` seconds."""
    count = int(round(hold / dt))
    if count < 1:
        raise ValueError("hold must cover at least one sample")
    return np.repeat(np.asarray(list(levels), dtype=float), count)


def simulate(
    plant: control.NonlinearIOSystem,
    reference: np.ndarray,
    gains: Gains,
    *,
    dt: float,
    limits: Limits | None = None,
    dead_zone: float | None = None,
    encoder_step: float | None = ENCODER_STEP,
    y0: float = 0.0,
) -> Trajectory:
    """Run ``plant`` under :class:`FirmwarePID`. Measurement is encoder-quantised.

    ``dead_zone`` is the firmware ``DeadZone``. It overrides ``limits.dead_zone``
    when given.
    """
    limits = _with_dead_zone(limits, dead_zone)
    pid = FirmwarePID(gains, limits)
    state = initial_state(plant, y0)
    y_now = float(y0)
    y = np.empty(len(reference))
    u = np.empty(len(reference))
    for i, setpoint in enumerate(reference):
        measured = _quantise(y_now, encoder_step)
        command = pid.step(float(setpoint), measured, dt)
        state, y_now = step(plant, state, command, dt)
        y[i] = y_now
        u[i] = command
    return Trajectory(y, u)


def ki_that_holds(plant: control.NonlinearIOSystem, levels, limits: Limits, dt: float) -> float:
    """Smallest ``Ki`` for which the integrator alone can hold every level.

    The holding command is ``Fc + |y| / K`` for an inertial plant and ``Fc``
    for an integrator. ``i_max = anti_windup * dt * Ki`` has to cover it.
    """
    needed = max(_holding_command(plant, float(level)) for level in levels)
    if needed <= 0.0:
        return 0.0
    rate = limits.anti_windup * dt
    if rate <= 0.0:
        raise ValueError("anti_windup and dt must be positive to remove steady-state error")
    return needed / rate


def _with_dead_zone(limits: Limits | None, dead_zone: float | None) -> Limits:
    base = limits if limits is not None else Limits()
    if dead_zone is None:
        return base
    return replace(base, dead_zone=float(dead_zone))


def _quantise(y: float, encoder_step: float | None) -> float:
    if encoder_step is None or encoder_step <= 0.0:
        return float(y)
    return float(np.round(y / encoder_step) * encoder_step)


def _holding_command(plant: control.NonlinearIOSystem, setpoint: float) -> float:
    """Actuator command that keeps ``setpoint`` once the plant has stopped."""
    fc = float(plant.params["fc"])
    if plant.params["kind"] == "integrator" or float(plant.params["gain"]) <= 0.0:
        return fc
    return fc + abs(float(setpoint)) / float(plant.params["gain"])
