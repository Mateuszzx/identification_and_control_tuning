pid
===

The controller the tuner steps. Change :mod:`pid.controller` to match your
PID. Keep ``kp``, ``ki``, and ``kd`` on :class:`~pid.controller.Gains`. The
search calls :func:`~pid.controller.make` and :meth:`~pid.controller.PID.step`.
If the integral limit is not ``anti_windup * dt * Ki``, change
:func:`~pid.controller.integral_rate` too.

.. automodule:: pid
   :no-members:

.. automodule:: pid.controller
   :members:
   :private-members:
