"""Discrete simulation of a plant driven by a proportional controller."""

from scipy.linalg import expm
import numpy as np

from plantid.systems import companion


def deadzone(command: float, fc: float) -> float:
    """Coulomb dead-zone. ``fc <= 0`` leaves the command unchanged."""
    if fc <= 0.0:
        return float(command)
    return float(np.sign(command) * max(abs(command) - fc, 0.0))


class Plant:
    """Fitted plant: a python-control transfer function plus an input dead-zone."""

    def __init__(self, kind: str, gain: float, taus: tuple[float, ...], fc: float):
        system, a, b, c = companion(kind, gain, taus)
        self.system = system
        self.kind = kind
        self.gain = float(gain)
        self.taus = tuple(float(tau) for tau in taus)
        self.fc = float(fc)
        self.A = a
        self.B = b
        self.C = c
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def initial_state(self, y0: float) -> np.ndarray:
        """State at rest with output ``y0``."""
        state = np.zeros(self.A.shape[0])
        state[-1] = y0
        return state

    def step(self, state: np.ndarray, command: float, dt: float) -> tuple[np.ndarray, float]:
        """Advance one sample. ``command`` is the controller output, before friction."""
        if dt <= 0.0:
            return state, float(self.C @ state)
        ue = deadzone(command, self.fc)
        ad, bd = self._discrete(dt)
        nxt = ad @ state + bd * ue
        return nxt, float(self.C @ nxt)

    def simulate(self, time: np.ndarray, setpoint: np.ndarray, kp: float, y0: float) -> np.ndarray:
        """Closed loop ``u = kp (r - y)`` sampled on ``time``."""
        state = self.initial_state(y0)
        out = np.empty(len(time))
        out[0] = y0
        for i in range(len(time) - 1):
            command = kp * (setpoint[i] - out[i])
            state, out[i + 1] = self.step(state, command, time[i + 1] - time[i])
        return out

    def _discrete(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(dt), 6)
        hit = self._cache.get(key)
        if hit is None:
            n = self.A.shape[0]
            aug = np.zeros((n + 1, n + 1))
            aug[:n, :n] = self.A
            aug[:n, n] = self.B
            step = expm(aug * key)
            hit = (step[:n, :n], step[:n, n])
            self._cache[key] = hit
        return hit
