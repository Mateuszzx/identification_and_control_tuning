"""Controller stepped by the tuner.

Edit :mod:`pid.controller` so :meth:`~pid.controller.PID.step` matches your
PID. The tuner searches ``kp``, ``ki``, and ``kd``, builds the controller
with :func:`~pid.controller.make`, and reads the integral limit from
:func:`~pid.controller.integral_rate`.
"""

from pid.controller import DT, ENCODER_STEP, PID, Gains, Limits, integral_rate, make

__all__ = [
    "DT",
    "ENCODER_STEP",
    "PID",
    "Gains",
    "Limits",
    "integral_rate",
    "make",
]
