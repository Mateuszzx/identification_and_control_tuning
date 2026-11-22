"""Closed loop of the plant and the controller in :mod:`pid`.

The controller itself is not defined here. :func:`simulate` calls
:func:`pid.controller.make` and :meth:`pid.controller.PID.step`.
"""

import numpy as np

from pid import Gains, Limits, integral_rate, make
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
    dead_zone: float | None = None,
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
    dead_zone : float, optional
        Passed to :func:`pid.controller.make`.
    y0 : float, optional
        Initial plant output :math:`y(0)`. Default is 0.

    Returns
    -------
    Trajectory
        Plant output :math:`y(t)` and controller command :math:`u(t)`.
    """
    pid = make(gains, limits=limits, dead_zone=dead_zone)
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


def ki_that_holds(plant: Plant, levels, limits: Limits, dt: float) -> float:
    """Smallest ``Ki`` for which the integrator alone can hold every level.

    The holding command is ``Fc + |y| / K`` for an inertial plant and ``Fc``
    for an integrator. It is divided by :func:`pid.controller.integral_rate`.

    Parameters
    ----------
    plant : plantid.plant.Plant
        Plant whose ``kind``, ``gain``, and ``fc`` set the holding command.
    levels : sequence of float
        Setpoints that must be held.
    limits : pid.controller.Limits
        Passed to :func:`pid.controller.integral_rate`.
    dt : float
        Sample time in seconds.

    Returns
    -------
    float
        Required ``Ki``. Zero when no holding command is needed.

    Raises
    ------
    ValueError
        :func:`pid.controller.integral_rate` is not positive while a holding command is needed.
    """
    needed = max(_holding_command(plant, float(level)) for level in levels)
    if needed <= 0.0:
        return 0.0
    rate = integral_rate(limits, dt)
    if rate <= 0.0:
        raise ValueError("the integral rate must be positive to remove steady-state error")
    return needed / rate


def _holding_command(plant: Plant, w: float) -> float:
    """Command :math:`u` that holds setpoint :math:`w` once the plant has stopped.

    Parameters
    ----------
    plant : plantid.plant.Plant
        Plant whose ``kind``, ``gain``, and ``fc`` set the command.
    w : float
        Setpoint to hold.

    Returns
    -------
    float
        ``Fc + abs(w) / K`` for an inertial plant, and ``Fc`` for an
        integrator or a non-positive gain.
    """
    if plant.kind == "integrator" or plant.gain <= 0.0:
        return plant.fc
    return plant.fc + abs(float(w)) / plant.gain
