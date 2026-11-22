"""Discrete PID stepped by the tuner.

Edit this file to match your controller. The shipped :meth:`PID.step` is
``PID.cpp``:

1. ``e = w - ym``, cleared when ``|e| < dead_zone``.
2. ``acc = clamp(e + (1 - smoothing) * acc, ±anti_windup)``.
3. ``p = Kp * e``, ``i = acc * dt * Ki``,
   ``d = Kd * (e - last_error) / dt``. The first sample uses
   ``last_error = 0``.
4. The output is lifted to ``±min_output`` and clipped to ``±max_output``.

Keep :class:`Gains` fields ``kp``, ``ki``, and ``kd``. The tuner searches
those three numbers, builds a controller with :func:`make`, and calls
:meth:`PID.step`. Dead zone, accumulator, and output limits are edited in
this file. Identification and tuning do not know them.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Gains:
    """Proportional, integral and derivative gains the tuner searches.

    Attributes
    ----------
    kp : float
        Proportional gain.
    ki : float
        Integral gain. Default is 0.
    kd : float
        Derivative gain. Default is 0.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0


@dataclass(frozen=True)
class Limits:
    """Limits of this PID, other than the three gains.

    Change these fields when your controller has different limits, and teach
    :func:`make` how to apply them.

    Attributes
    ----------
    max_output : float
        Symmetric output clip. Default is 100.
    min_output : float
        Output whose magnitude is below this is lifted to this magnitude.
        Default is 0.
    anti_windup : float
        Symmetric clamp on the error accumulator. Default is 100.
    dead_zone : float
        Errors whose magnitude is below this are cleared. Default is 0.1,
        the shipped firmware value.
    smoothing : float
        Accumulator leak, ``acc = error + (1 - smoothing) * acc``. Default is 0.
    """

    max_output: float = 100.0
    min_output: float = 0.0
    anti_windup: float = 100.0
    dead_zone: float = 0.1
    smoothing: float = 0.0


@dataclass
class PID:
    """Stateful controller. Replace :meth:`step` to match your PID.

    Attributes
    ----------
    gains : Gains
        ``Kp``, ``Ki``, and ``Kd``.
    limits : Limits
        Output, anti-windup, dead-zone, and smoothing limits.
    acc : float
        Accumulated error. Updated by :meth:`step`.
    last_error : float
        Error used by the derivative on the next sample. The first sample
        treats this as 0.
    """

    gains: Gains
    limits: Limits = field(default_factory=Limits)
    acc: float = 0.0
    last_error: float = 0.0

    def reset(self) -> None:
        """Clear the accumulator and the previous error."""
        self.acc = 0.0
        self.last_error = 0.0

    def step(self, w: float, ym: float, dt: float) -> float:
        """One controller sample.

        ``w`` is the setpoint :math:`w(t)` and ``ym`` is the sensor reading
        :math:`y_m(t)` on the closed-loop drawing. The return value is the
        command :math:`u(t)`.

        Parameters
        ----------
        w : float
            Setpoint :math:`w(t)` for this sample.
        ym : float
            Measured output :math:`y_m(t)`.
        dt : float
            Sample time in seconds. A non-positive value sets the derivative to 0.

        Returns
        -------
        float
            Controller output :math:`u(t)`.
        """
        e = float(w - ym)
        if abs(e) < self.limits.dead_zone:
            e = 0.0
        acc = e + (1.0 - self.limits.smoothing) * self.acc
        acc = float(np.clip(acc, -self.limits.anti_windup, self.limits.anti_windup))
        d_term = 0.0 if dt <= 0.0 else self.gains.kd * (e - self.last_error) / dt
        u = self.gains.kp * e + acc * dt * self.gains.ki + d_term
        if abs(u) < self.limits.min_output:
            u = float(np.sign(u) * self.limits.min_output)
        if abs(u) > self.limits.max_output:
            u = float(np.sign(u) * self.limits.max_output)
        self.acc = acc
        self.last_error = e
        return u


def make(gains: Gains, *, limits: Limits | None = None) -> PID:
    """Build the controller the tuner steps.

    Parameters
    ----------
    gains : Gains
        Gains for this candidate.
    limits : Limits, optional
        Controller limits. Default is :class:`Limits`.

    Returns
    -------
    PID
        Controller ready for the first sample.
    """
    return PID(gains, limits if limits is not None else Limits())
