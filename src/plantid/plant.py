"""Identified plant as a python-control system.

The linear part is a :class:`control.TransferFunction` and the matching
:class:`control.StateSpace` in phase-variable form, so the last state is the
output. Coulomb friction is an input dead-zone on a
:class:`control.NonlinearIOSystem`. Discrete steps use
:func:`control.sample_system` (zero-order hold), which is what the firmware
loop does with a held command.
"""

import json
from pathlib import Path

import control
import numpy as np

from plantid.systems import companion


def deadzone(command: float, fc: float) -> float:
    """Coulomb dead-zone. ``fc <= 0`` leaves the command unchanged."""
    if fc <= 0.0:
        return float(command)
    return float(np.sign(command) * max(abs(command) - fc, 0.0))


def io_system(
    kind: str, gain: float, taus: tuple[float, ...], fc: float
) -> control.NonlinearIOSystem:
    """Continuous plant: linear dynamics with a Coulomb dead-zone on the input."""
    linear, states = _realization(kind, gain, taus)
    params = {
        "A": states.A,
        "B": np.asarray(states.B).reshape(-1),
        "C": np.asarray(states.C).reshape(-1),
        "fc": float(fc),
        "kind": kind,
        "gain": float(gain),
        "taus": tuple(float(tau) for tau in taus),
    }

    def update(_t, x, u, params):
        ue = deadzone(float(np.asarray(u).reshape(-1)[0]), params["fc"])
        return params["A"] @ x + params["B"] * ue

    def output(_t, x, _u, params):
        return np.atleast_1d(params["C"] @ x)

    system = control.nlsys(
        update,
        output,
        inputs=1,
        outputs=1,
        states=states.nstates,
        params=params,
    )
    system.linear = linear
    system.linear_ss = states
    return system


def initial_state(system: control.NonlinearIOSystem, y0: float) -> np.ndarray:
    """Rest state whose output is ``y0``. The last state is the output."""
    state = np.zeros(system.nstates)
    if system.nstates:
        state[-1] = y0
    return state


def step(system: control.NonlinearIOSystem, state: np.ndarray, command: float, dt: float):
    """Advance one zero-order hold. ``command`` is the controller output, before friction."""
    if dt <= 0.0:
        return state, float(system.params["C"] @ state)
    ue = deadzone(command, system.params["fc"])
    discrete = _discrete(system, dt)
    nxt = np.asarray(discrete.A @ state + np.asarray(discrete.B).reshape(-1) * ue).reshape(-1)
    return nxt, float(np.asarray(discrete.C @ nxt).reshape(-1)[0])


def proportional_response(
    system: control.NonlinearIOSystem,
    time: np.ndarray,
    setpoint: np.ndarray,
    kp: float,
    y0: float,
) -> np.ndarray:
    """Closed loop ``u = kp (r - y)`` sampled on ``time``."""
    state = initial_state(system, y0)
    out = np.empty(len(time))
    out[0] = y0
    for i in range(len(time) - 1):
        command = kp * (setpoint[i] - out[i])
        state, out[i + 1] = step(system, state, command, time[i + 1] - time[i])
    return out


def save(system: control.NonlinearIOSystem, path, **meta) -> None:
    """Write the plant parameters as JSON. ``meta`` is stored beside them."""
    params = system.params
    payload = {
        "kind": params["kind"],
        "gain": params["gain"],
        "taus": list(params["taus"]),
        "fc": params["fc"],
    }
    payload.update(meta)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def load(path) -> control.NonlinearIOSystem:
    """Read a plant saved with :func:`save`."""
    data = json.loads(Path(path).read_text())
    kind = str(data["kind"])
    if kind not in ("inertial", "integrator"):
        raise ValueError(f"unknown plant kind {kind!r}")
    taus = tuple(float(tau) for tau in data["taus"])
    return io_system(kind, float(data["gain"]), taus, float(data["fc"]))


def _realization(kind: str, gain: float, taus: tuple[float, ...]):
    """Phase-variable :class:`control.StateSpace` of the linear plant."""
    linear, a, b, c = companion(kind, gain, taus)
    states = control.ss(a, b.reshape(-1, 1), c.reshape(1, -1), 0)
    return linear, states


def _discrete(system: control.NonlinearIOSystem, dt: float) -> control.StateSpace:
    key = round(float(dt), 6)
    cache = getattr(system, "_zoh", None)
    if cache is None:
        cache = {}
        system._zoh = cache
    hit = cache.get(key)
    if hit is None:
        hit = control.sample_system(system.linear_ss, key)
        cache[key] = hit
    return hit
