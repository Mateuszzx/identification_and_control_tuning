pid
===

The controller the tuner steps. Change :mod:`pid.controller` to match your
PID. Keep ``kp``, ``ki``, and ``kd`` on :class:`~pid.controller.Gains`. The
search calls :func:`~pid.controller.make` and :meth:`~pid.controller.PID.step`.
Dead zone and accumulator are edited in that module.

.. automodule:: pid
   :no-members:

.. automodule:: pid.controller
   :members:
   :private-members:
