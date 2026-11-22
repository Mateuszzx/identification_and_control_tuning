How to use
==========

The two steps stay independent. Identification writes a plant file. Tuning
reads that file and never refits it.

Configuration
-------------

``config/config.yaml`` is the setup for both steps. A relative path inside
the file starts at the repository root.

.. code-block:: yaml

   identify:
     log: data/fr_prop_pid.csv
     out: model/plant.json

   tune:
     plant: model/plant.json
     dt: 0.015
     dead_zone: 0.1
     hold: 6.0
     levels: [0, -30, 0, -60, 0, -90, 0, 30, 0, 60, 0, 90, 0]
     kp: [1.0, 120.0]
     ki: [0.0, 10.0]
     kd: [0.0, 1.5]
     we: 1.0
     allow_steady_state_error: false
     method: pygad
     generations: 400
     population: 50

``identify.log`` is the CSV described in the :doc:`quickstart`. ``identify.out``
is the plant JSON. Tuning reads ``tune.plant``, which is normally the same file.

``dt`` is the simulation sample time in seconds. ``dead_zone`` is passed to
:func:`pid.controller.make`. The shipped controller clears an error inside it
before the proportional, integral, and derivative terms. ``hold`` is how long
each entry in ``levels`` is held.
``we`` weights the integral of squared setpoint error.

``kp``, ``ki``, and ``kd`` are search intervals ``[low, high]``. A single
number, such as ``ki: 3``, fixes that gain. The search never leaves these
limits.

``allow_steady_state_error: false`` raises the ``Ki`` floor, inside the
interval you set, so the integrator can hold every level. If the whole
interval is below that floor, ``Ki`` stays at the upper end and a shelf
remains. ``true`` searches the whole ``Ki`` interval and may leave that shelf.

``method`` is ``powell`` or ``pygad``. Both minimize the same simulated error.
``powell`` starts from the low, middle, and high end of each free gain, then
keeps the visited point with the smallest error. ``pygad`` searches the same
box genetically. ``generations`` and ``population`` apply only to ``pygad``.

Identify
--------

.. code-block:: bash

   just identify
   uv run python app/identify.py --config config/config.yaml

lmfit minimizes the output error of each candidate in the catalog. The
catalog is inertial plants of order 1–3, with and without Coulomb friction,
then integrators with 0, 1, and 2 lags and friction. The proportional gain
is ``median(p_term / error)`` from the log. It is not a fitted parameter.

The lowest RMSE is saved as a :class:`plantid.plant.Plant`: a transfer function
with a Coulomb dead-zone on the input. ``output/identify.png`` shows the measured
output, the fitted response, and the residual.

Tune
----

.. code-block:: bash

   just tune
   uv run python app/tune.py --config config/config.yaml

Run identification first, so ``tune.plant`` exists. The simulated controller
is :meth:`pid.controller.PID.step`. The shipped body matches ``PID.cpp``:

1. ``error = setpoint - measurement``, cleared when ``|error| < dead_zone``.
2. The accumulator is ``error + (1 - smoothing) * acc``, clamped to
   ``±anti_windup``.
3. ``p = Kp * error``, ``i = acc * dt * Ki``, and
   ``d = Kd * (error - last_error) / dt``. The first sample uses
   ``last_error = 0``.
4. The output is lifted to ``±min_output`` and clipped to ``±max_output``.

To match a different PID, edit ``src/pid/controller.py``. Keep ``kp``, ``ki``,
and ``kd``. Change :func:`pid.controller.integral_rate` when the integral
limit is not ``anti_windup * dt * Ki``.

``output/tune.png`` shows the setpoint, the simulated response, and the
controller output. Printed text reports ``Kp``, ``Ki``, ``Kd``, the integral
of squared error, and the worst end-of-hold error.

Commands
--------

`just <https://github.com/casey/just>`_ runs the common tasks from the
repository root.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Effect
   * - ``just check``
     - Lint with Ruff and run the tests.
   * - ``just identify``
     - Fit the plant and write ``output/identify.png``.
   * - ``just tune``
     - Tune the PID and write ``output/tune.png``.
   * - ``just docs``
     - Build this HTML documentation into ``docs/_build/html``.
   * - ``uv run pytest``
     - Run the tests directly.

Figure files in ``output/`` are gitignored. Identification and tuning stay
off continuous integration because the logs in ``data/`` are not in the
repository.
