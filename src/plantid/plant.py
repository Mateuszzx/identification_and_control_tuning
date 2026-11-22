"""Identified plant.

The plant is a :class:`control.TransferFunction` and the matching
:class:`control.StateSpace` in phase-variable form, so the last state is the
output. Coulomb friction is a dead-zone on the plant input. Discrete steps
use :func:`control.sample_system` (zero-order hold), which is what a held
command does. The controller dead zone and accumulator stay in :mod:`pid`.
"""

import json
from pathlib import Path

import control
import numpy as np


class Plant:
    """Linear dynamics with a Coulomb dead-zone on the input.

    Parameters
    ----------
    kind : {"inertial", "integrator"}
        Plant family.
    gain : float
        Numerator gain, ``K`` or ``Kv``.
    taus : tuple of float, optional
        Real pole time constants. Default is no extra poles.
    fc : float, optional
        Coulomb threshold. Zero disables the dead-zone. Default is 0.
    integrators : int, optional
        Power of :math:`1/s`. Default is 1 when ``kind`` is ``"integrator"``
        and 0 otherwise.
    zeros : tuple of float, optional
        Real zero time constants, each a factor :math:`(\\zeta s + 1)`.
        Default is no zeros.

    Attributes
    ----------
    kind : str
        ``"integrator"`` when ``integrators`` is positive, otherwise ``"inertial"``.
    gain : float
        Numerator gain.
    taus : tuple of float
        Real pole time constants.
    fc : float
        Coulomb threshold.
    integrators : int
        Power of :math:`1/s`.
    zeros : tuple of float
        Real zero time constants, sorted from slowest to fastest.
    linear : control.TransferFunction
        Linear plant, without friction.
    """

    def __init__(
        self,
        kind: str,
        gain: float,
        taus: tuple[float, ...] = (),
        fc: float = 0.0,
        *,
        integrators: int | None = None,
        zeros: tuple[float, ...] = (),
    ):
        """Store the parameters and build the phase-variable realization.

        Raises
        ------
        ValueError
            ``kind`` is unknown, or the zeros make the plant improper.
        """
        if integrators is None:
            if kind not in ("inertial", "integrator"):
                raise ValueError(f"unknown plant kind {kind!r}")
            integrators = 1 if kind == "integrator" else 0
        zeros = tuple(float(zero) for zero in zeros)
        if int(integrators) < 0 or any(zero <= 0.0 for zero in zeros):
            raise ValueError("integrators and zero time constants must be positive")
        if int(integrators) + len(taus) < 1:
            raise ValueError("the plant needs at least one pole")
        if len(zeros) > int(integrators) + len(taus):
            raise ValueError("zeros must not outnumber poles, or the plant is improper")
        self.kind = "integrator" if int(integrators) > 0 else "inertial"
        self.gain = float(gain)
        self.taus = tuple(float(tau) for tau in taus)
        self.fc = float(fc)
        self.integrators = int(integrators)
        self.zeros = tuple(sorted(zeros, reverse=True))
        self.linear, self._states = self._realization()
        self._zoh: dict[float, control.StateSpace] = {}

    @classmethod
    def load(cls, path) -> "Plant":
        """Read a plant saved with :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            JSON file.

        Returns
        -------
        Plant
            Plant rebuilt from ``kind``, ``gain``, ``taus``, ``fc``,
            ``integrators``, and ``zeros``. A missing ``fc`` is 0. A missing
            ``integrators`` follows ``kind``. Missing ``zeros`` means none.

        Raises
        ------
        ValueError
            ``kind`` is not ``"inertial"`` or ``"integrator"``.
        """
        data = json.loads(Path(path).read_text())
        taus = tuple(float(tau) for tau in data["taus"])
        zeros = tuple(float(zero) for zero in data.get("zeros", ()))
        integrators = data.get("integrators")
        return cls(
            str(data["kind"]),
            float(data["gain"]),
            taus,
            float(data.get("fc", 0.0)),
            integrators=None if integrators is None else int(integrators),
            zeros=zeros,
        )

    def save(self, path, **meta) -> None:
        """Write the plant parameters as JSON.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination. Parent directories are created.
        **meta
            Extra JSON fields stored beside the plant parameters.
        """
        payload = {
            "kind": self.kind,
            "integrators": self.integrators,
            "gain": self.gain,
            "taus": list(self.taus),
            "zeros": list(self.zeros),
            "fc": self.fc,
        }
        payload.update(meta)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")

    def initial_state(self, y0: float) -> np.ndarray:
        """Rest state whose output is ``y0``.

        Derivatives of the phase variable start at zero. The constant term of
        the numerator carries ``y0``.

        Parameters
        ----------
        y0 : float
            Initial output.

        Returns
        -------
        numpy.ndarray
            State vector.
        """
        state = np.zeros(self._states.nstates)
        output = np.asarray(self._states.C).reshape(-1)
        if output.size and abs(float(output[-1])) > 1e-15:
            state[-1] = y0 / float(output[-1])
        return state

    def step(self, state: np.ndarray, u: float, dt: float):
        """Advance one zero-order hold.

        ``u`` is the controller output :math:`u(t)` on the closed-loop drawing.
        The disturbance :math:`z(t)` is zero. The returned output is :math:`y(t)`.

        Parameters
        ----------
        state : numpy.ndarray
            State at the beginning of the step.
        u : float
            Controller output :math:`u(t)`, before friction.
        dt : float
            Step length in seconds. A non-positive value holds the state.

        Returns
        -------
        state : numpy.ndarray
            State at the end of the step.
        y : float
            Plant output :math:`y(t)` at the end of the step.
        """
        ue = _deadzone(u, self.fc)
        if dt <= 0.0:
            return state, _output(self._states, state, ue)
        discrete = self._discrete(dt)
        nxt = np.asarray(discrete.A @ state + np.asarray(discrete.B).reshape(-1) * ue).reshape(-1)
        return nxt, _output(discrete, nxt, ue)

    def proportional_response(
        self,
        time: np.ndarray,
        w: np.ndarray,
        kp: float,
        y0: float,
    ) -> np.ndarray:
        """Closed loop ``u = kp * (w - ym)`` sampled on ``time``.

        The sensor is the identity, so ``ym`` is the simulated output ``y``.
        The disturbance ``z`` is zero.

        Parameters
        ----------
        time : numpy.ndarray
            Sample times. The interval between samples is the hold time.
        w : numpy.ndarray
            Setpoint :math:`w(t)`, the same length as ``time``.
        kp : float
            Proportional gain.
        y0 : float
            Output :math:`y(t)` at ``time[0]``.

        Returns
        -------
        numpy.ndarray
            Simulated plant output :math:`y(t)`, the same length as ``time``.
        """
        state = self.initial_state(y0)
        y = np.empty(len(time))
        y[0] = y0
        for i in range(len(time) - 1):
            ym = y[i]
            u = kp * (w[i] - ym)
            state, y[i + 1] = self.step(state, u, time[i + 1] - time[i])
        return y

    def closed_loop(self, kp: float) -> control.TransferFunction:
        """Linear loop ``Kp G / (1 + Kp G)``.

        Parameters
        ----------
        kp : float
            Proportional gain wrapped around the plant.

        Returns
        -------
        control.TransferFunction
            Unity-feedback closed loop.
        """
        return control.feedback(kp * self.linear, 1)

    def _realization(self):
        """Phase-variable state space of the linear plant.

        Integrator powers and real zeros are in this realization. A matching
        number of zeros and poles leaves a direct term.

        Returns
        -------
        linear : control.TransferFunction
            Same transfer function as the state space.
        states : control.StateSpace
            Phase-variable realization.
        """
        system, a, b, c, direct = _companion(self.integrators, self.gain, self.taus, self.zeros)
        states = control.ss(a, b.reshape(-1, 1), c.reshape(1, -1), direct)
        return system, states

    def _discrete(self, dt: float) -> control.StateSpace:
        """Zero-order-hold discretization, cached on this plant.

        Parameters
        ----------
        dt : float
            Sample time in seconds. The cache key is ``dt`` rounded to 6 decimals.

        Returns
        -------
        control.StateSpace
            Discrete linear plant. Friction is applied outside this system.
        """
        key = round(float(dt), 6)
        hit = self._zoh.get(key)
        if hit is None:
            hit = control.sample_system(self._states, key)
            self._zoh[key] = hit
        return hit


def _output(system: control.StateSpace, state: np.ndarray, ue: float) -> float:
    """Output of a linear state space, including a direct term.

    Parameters
    ----------
    system : control.StateSpace
        Continuous or discrete plant. Friction is already removed from ``ue``.
    state : numpy.ndarray
        State at which the output is evaluated.
    ue : float
        Plant input after the Coulomb dead-zone.

    Returns
    -------
    float
        ``C x + D u``.
    """
    feed = float(np.asarray(system.D).reshape(-1)[0]) * ue
    return float(np.asarray(system.C @ state).reshape(-1)[0] + feed)


def _deadzone(u: float, fc: float) -> float:
    """Coulomb dead-zone on the plant input :math:`u(t)`.

    Parameters
    ----------
    u : float
        Controller output :math:`u(t)`, before friction.
    fc : float
        Coulomb threshold. ``fc <= 0`` leaves ``u`` unchanged.

    Returns
    -------
    float
        ``0`` when ``abs(u) < fc``, otherwise ``u - fc * sign(u)``.
    """
    if fc <= 0.0:
        return float(u)
    return float(np.sign(u) * max(abs(u) - fc, 0.0))


def _companion(integrators: int, gain: float, taus: tuple[float, ...], zeros: tuple[float, ...]):
    """Phase-variable matrices of the linear plant.

    Parameters
    ----------
    integrators : int
        Power of :math:`1/s` in the denominator.
    gain : float
        Numerator gain, ``K`` or ``Kv``.
    taus : tuple of float
        Pole time constants. Each contributes a factor ``(tau s + 1)``.
    zeros : tuple of float
        Zero time constants. Each contributes a factor ``(zeta s + 1)``.

    Returns
    -------
    system : control.TransferFunction
        Linear plant ``G(s)``.
    a : numpy.ndarray
        State matrix of shape ``(n, n)``.
    b : numpy.ndarray
        Input vector of shape ``(n,)``.
    c : numpy.ndarray
        Output row of shape ``(n,)``. Numerator coefficients, highest power first.

    Raises
    ------
    ValueError
        The plant has no poles or is improper.
    """
    denominator = np.array([1.0])
    for _ in range(int(integrators)):
        denominator = np.convolve(denominator, [1.0, 0.0])
    for tau in taus:
        denominator = np.convolve(denominator, [float(tau), 1.0])
    numerator = np.array([float(gain)])
    for zeta in zeros:
        numerator = np.convolve(numerator, [float(zeta), 1.0])
    if denominator.size < 2 or numerator.size > denominator.size:
        raise ValueError("the plant needs at least one pole and must not be improper")
    lead = float(denominator[0])
    if numerator.size == denominator.size:
        direct = float(numerator[0] / lead)
        strict = (numerator - direct * denominator)[1:]
    else:
        direct = 0.0
        strict = numerator
    coeff = denominator[1:] / lead
    n = int(coeff.size)
    a = np.zeros((n, n))
    a[0, :] = -coeff
    if n > 1:
        a[1:, :-1] = np.eye(n - 1)
    b = np.zeros(n)
    b[0] = 1.0
    scaled = strict / lead
    c = np.zeros(n)
    if scaled.size:
        c[-scaled.size :] = scaled
    system = control.tf(numerator, denominator)
    return system, a, b, c, direct
