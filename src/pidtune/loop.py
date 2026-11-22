"""Closed loop of the plant and the controller in :mod:`pid`.

The controller itself is not defined here. :func:`simulate` calls
:func:`pid.controller.make` and :meth:`pid.controller.PID.step`.
"""

import control
import numpy as np

from pid import ENCODER_STEP, Gains, Limits, integral_rate, make
from plantid.plant import initial_state, step


class Trajectory(tuple):
    """``(y, u)`` from :func:`simulate`. Unpacks like a pair.

    ``y`` is the simulated plant output and ``u`` is the controller command.
    """

    def __new__(cls, y: np.ndarray, u: np.ndarray):
        """Pair the output and the command.

        Parameters
        ----------
        y : numpy.ndarray
            Simulated plant output.
        u : numpy.ndarray
            Controller output.

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
    """Run ``plant`` under the controller from :func:`pid.controller.make`.

    The measurement is quantised to the encoder step before it enters the PID.

    Parameters
    ----------
    plant : control.NonlinearIOSystem
        Plant from :func:`plantid.io_system`.
    reference : numpy.ndarray
        Sampled setpoint.
    gains : pid.controller.Gains
        PID gains.
    dt : float
        Sample time in seconds.
    limits : pid.controller.Limits, optional
        Controller limits. Default is constructed by :func:`pid.controller.make`.
    dead_zone : float, optional
        Passed to :func:`pid.controller.make`.
    encoder_step : float or None, optional
        Quantisation step. ``None`` or a non-positive value disables it.
        Default is :data:`pid.ENCODER_STEP`.
    y0 : float, optional
        Initial plant output. Default is 0.

    Returns
    -------
    Trajectory
        Plant output and controller command.
    """
    pid = make(gains, limits=limits, dead_zone=dead_zone)
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
    for an integrator. It is divided by :func:`pid.controller.integral_rate`.

    Parameters
    ----------
    plant : control.NonlinearIOSystem
        Plant whose ``params`` supply ``kind``, ``gain``, and ``fc``.
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


def _quantise(y: float, encoder_step: float | None) -> float:
    """Round a measurement to the encoder grid.

    Parameters
    ----------
    y : float
        Continuous plant output.
    encoder_step : float or None
        Grid size. ``None`` or a non-positive value returns ``y``.

    Returns
    -------
    float
        Quantised measurement.
    """
    if encoder_step is None or encoder_step <= 0.0:
        return float(y)
    return float(np.round(y / encoder_step) * encoder_step)


def _holding_command(plant: control.NonlinearIOSystem, setpoint: float) -> float:
    """Actuator command that keeps ``setpoint`` once the plant has stopped.

    Parameters
    ----------
    plant : control.NonlinearIOSystem
        Plant whose ``params`` supply ``kind``, ``gain``, and ``fc``.
    setpoint : float
        Output to hold.

    Returns
    -------
    float
        ``Fc + abs(setpoint) / K`` for an inertial plant, and ``Fc`` for an
        integrator or a non-positive gain.
    """
    fc = float(plant.params["fc"])
    if plant.params["kind"] == "integrator" or float(plant.params["gain"]) <= 0.0:
        return fc
    return fc + abs(float(setpoint)) / float(plant.params["gain"])
