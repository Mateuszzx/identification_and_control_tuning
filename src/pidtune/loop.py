"""Closed loop of the plant and the controller in :mod:`pid`.

The controller itself is not defined here. :func:`simulate` calls
:func:`pid.controller.make` and :meth:`pid.controller.PID.step`.
"""

import numpy as np

from pid import Gains, Limits, make
from plantid.plant import Plant


class Trajectory(tuple):
    """``(y, u)`` from :func:`simulate`. Unpacks like a pair.

    ``y`` is the plant output :math:`y(t)` and ``u`` is the controller output :math:`u(t)`.
    """

    def __new__(cls, y: np.ndarray, u: np.ndarray):
        """Pair the output and the command.

        Parameters
        ----------
        y : numpy.ndarray
            Simulated plant output :math:`y(t)`.
        u : numpy.ndarray
            Controller output :math:`u(t)`.

        Returns
        -------
        Trajectory
            Tuple ``(y, u)``.
        """
        return super().__new__(cls, (y, u))

    @property
    def y(self) -> np.ndarray:
        """Simulated plant output.

        Returns
        -------
        numpy.ndarray
            First element of the pair.
        """
        return self[0]

    @property
    def u(self) -> np.ndarray:
        """Controller output.

        Returns
        -------
        numpy.ndarray
            Second element of the pair.
        """
        return self[1]


def steps(levels, hold: float, dt: float) -> np.ndarray:
    """Piecewise-constant reference. Each level is held for ``hold`` seconds.

    Parameters
    ----------
    levels : sequence of float
        Setpoint values, in order.
    hold : float
        Hold time of each level, in seconds.
    dt : float
        Sample time in seconds.

    Returns
    -------
    numpy.ndarray
        Reference sampled at ``dt``. Each level is repeated
        ``round(hold / dt)`` times.

    Raises
    ------
    ValueError
        ``hold`` covers less than one sample.
    """
    count = int(round(hold / dt))
    if count < 1:
        raise ValueError("hold must cover at least one sample")
    return np.repeat(np.asarray(list(levels), dtype=float), count)


def simulate(
    plant: Plant,
    reference: np.ndarray,
    gains: Gains,
    *,
    dt: float,
    limits: Limits | None = None,
    y0: float = 0.0,
) -> Trajectory:
    """Run ``plant`` under the controller from :func:`pid.controller.make`.

    The sensor is the identity, so :math:`y_m(t) = y(t)`. The disturbance
    :math:`z(t)` is zero. ``reference`` is the setpoint :math:`w(t)`.

    Parameters
    ----------
    plant : plantid.plant.Plant
        Plant to simulate.
    reference : numpy.ndarray
        Sampled setpoint :math:`w(t)`.
    gains : pid.controller.Gains
        PID gains.
    dt : float
        Sample time in seconds. The config supplies this as ``tune.dt``.
    limits : pid.controller.Limits, optional
        Controller limits. Default is constructed by :func:`pid.controller.make`.
    y0 : float, optional
        Initial plant output :math:`y(0)`. Default is 0.

    Returns
    -------
    Trajectory
        Plant output :math:`y(t)` and controller command :math:`u(t)`.
    """
    pid = make(gains, limits=limits)
    state = plant.initial_state(y0)
    y_now = float(y0)
    y = np.empty(len(reference))
    u = np.empty(len(reference))
    for i, w in enumerate(reference):
        ym = y_now
        u_now = pid.step(float(w), ym, dt)
        state, y_now = plant.step(state, u_now, dt)
        y[i] = y_now
        u[i] = u_now
    return Trajectory(y, u)
