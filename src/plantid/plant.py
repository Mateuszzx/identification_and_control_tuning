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
    """Coulomb dead-zone.

    Parameters
    ----------
    command : float
        Actuator command before friction.
    fc : float
        Coulomb threshold. ``fc <= 0`` leaves the command unchanged.

    Returns
    -------
    float
        ``0`` when ``abs(command) < fc``, otherwise ``command - fc * sign(command)``.
    """
    if fc <= 0.0:
        return float(command)
    return float(np.sign(command) * max(abs(command) - fc, 0.0))


def io_system(
    kind: str, gain: float, taus: tuple[float, ...], fc: float
) -> control.NonlinearIOSystem:
    """Continuous plant: linear dynamics with a Coulomb dead-zone on the input.

    Parameters
    ----------
    kind : {"inertial", "integrator"}
        Plant family.
    gain : float
        Numerator gain.
    taus : tuple of float
        Real time constants, longest first after identification.
    fc : float
        Coulomb threshold. Zero disables the dead-zone.

    Returns
    -------
    control.NonlinearIOSystem
        Plant whose ``params`` hold ``A``, ``B``, ``C``, ``fc``, ``kind``,
        ``gain``, and ``taus``. ``linear`` is the transfer function and
        ``linear_ss`` is the phase-variable state space.
    """
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
        """Continuous state derivative with the input dead-zone applied.

        Parameters
        ----------
        _t : float
            Unused time.
        x : numpy.ndarray
            Current state.
        u : array_like
            Actuator command, before friction.
        params : dict
            System parameters. ``fc`` is the Coulomb threshold.

        Returns
        -------
        numpy.ndarray
            ``A x + B * deadzone(u, fc)``.
        """
        ue = deadzone(float(np.asarray(u).reshape(-1)[0]), params["fc"])
        return params["A"] @ x + params["B"] * ue

    def output(_t, x, _u, params):
        """Plant output, the last state.

        Parameters
        ----------
        _t : float
            Unused time.
        x : numpy.ndarray
            Current state.
        _u : array_like
            Unused input.
        params : dict
            System parameters. ``C`` selects the output.

        Returns
        -------
        numpy.ndarray
            ``C x`` as a length-1 vector.
        """
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
    """Rest state whose output is ``y0``.

    The last state is the output. Every other state starts at zero.

    Parameters
    ----------
    system : control.NonlinearIOSystem
        Plant from :func:`io_system`.
    y0 : float
        Initial output.

    Returns
    -------
    numpy.ndarray
        State vector of length ``system.nstates``.
    """
    state = np.zeros(system.nstates)
    if system.nstates:
        state[-1] = y0
    return state


def step(system: control.NonlinearIOSystem, state: np.ndarray, command: float, dt: float):
    """Advance one zero-order hold.

    Parameters
    ----------
    system : control.NonlinearIOSystem
        Plant from :func:`io_system`.
    state : numpy.ndarray
        State at the beginning of the step.
    command : float
        Controller output, before friction.
    dt : float
        Step length in seconds. A non-positive value holds the state.

    Returns
    -------
    state : numpy.ndarray
        State at the end of the step.
    output : float
        Plant output at the end of the step.
    """
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
    """Closed loop ``u = kp (r - y)`` sampled on ``time``.

    Parameters
    ----------
    system : control.NonlinearIOSystem
        Plant from :func:`io_system`.
    time : numpy.ndarray
        Sample times. The interval between samples is the hold time.
    setpoint : numpy.ndarray
        Reference, the same length as ``time``.
    kp : float
        Proportional gain.
    y0 : float
        Output at ``time[0]``.

    Returns
    -------
    numpy.ndarray
        Simulated output, the same length as ``time``.
    """
    state = initial_state(system, y0)
    out = np.empty(len(time))
    out[0] = y0
    for i in range(len(time) - 1):
        command = kp * (setpoint[i] - out[i])
        state, out[i + 1] = step(system, state, command, time[i + 1] - time[i])
    return out


def save(system: control.NonlinearIOSystem, path, **meta) -> None:
    """Write the plant parameters as JSON.

    Parameters
    ----------
    system : control.NonlinearIOSystem
        Plant from :func:`io_system`.
    path : str or pathlib.Path
        Destination. Parent directories are created.
    **meta
        Extra JSON fields stored beside ``kind``, ``gain``, ``taus``, and ``fc``.
    """
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
    """Read a plant saved with :func:`save`.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON file.

    Returns
    -------
    control.NonlinearIOSystem
        Plant rebuilt from ``kind``, ``gain``, ``taus``, and ``fc``.

    Raises
    ------
    ValueError
        ``kind`` is not ``"inertial"`` or ``"integrator"``.
    """
    data = json.loads(Path(path).read_text())
    kind = str(data["kind"])
    if kind not in ("inertial", "integrator"):
        raise ValueError(f"unknown plant kind {kind!r}")
    taus = tuple(float(tau) for tau in data["taus"])
    return io_system(kind, float(data["gain"]), taus, float(data["fc"]))


def _realization(kind: str, gain: float, taus: tuple[float, ...]):
    """Phase-variable state space of the linear plant.

    Parameters
    ----------
    kind : {"inertial", "integrator"}
        Plant family.
    gain : float
        Numerator gain.
    taus : tuple of float
        Real time constants.

    Returns
    -------
    linear : control.TransferFunction
        Same transfer function as the state space.
    states : control.StateSpace
        Phase-variable realization whose last state is the output.
    """
    linear, a, b, c = companion(kind, gain, taus)
    states = control.ss(a, b.reshape(-1, 1), c.reshape(1, -1), 0)
    return linear, states


def _discrete(system: control.NonlinearIOSystem, dt: float) -> control.StateSpace:
    """Zero-order-hold discretization, cached on ``system._zoh``.

    Parameters
    ----------
    system : control.NonlinearIOSystem
        Plant whose ``linear_ss`` attribute is sampled.
    dt : float
        Sample time in seconds. The cache key is ``dt`` rounded to 6 decimals.

    Returns
    -------
    control.StateSpace
        Discrete linear plant. Friction is applied outside this system.
    """
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
