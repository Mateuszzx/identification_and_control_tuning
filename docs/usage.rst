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
     kp: 2.0
     models: config/models.yaml
     max_nfev: 100

   tune:
     plant: model/plant.json
     dt: 0.015
     hold: 6.0
     levels: [0, -30, 0, -60, 0, -90, 0, 30, 0, 60, 0, 90, 0]
     kp: [1.0, 120.0]
     ki: [0.0, 10.0]
     kd: [0.0, 1.5]
     we: 1.0
     method: pygad
     generations: 400
     population: 50
     parents: 16
     selection: sss
     crossover: single_point
     mutation: random
     mutation_by_replacement: true
     elitism: 1
     seed: 1
     inertia: 0.7
     cognitive: 1.5
     social: 1.5
     maxfev: 40
     xtol: 1.0e-3
     ftol: 1.0e-3

``identify.log`` is the CSV described in the :doc:`quickstart`. ``identify.out``
is the plant JSON. ``identify.kp`` is the proportional gain of the loop that
recorded the log. ``identify.models`` is the YAML list of structures to fit.
Add a block there to try another plant. ``identify.max_nfev`` is the number of
residual evaluations allowed for each structure. Tuning reads ``tune.plant``, which is
normally the same file.

``dt`` is the simulation sample time in seconds. ``hold`` is how long each
entry in ``levels`` is held. ``we`` weights the integral of squared setpoint
error. The controller dead zone and accumulator are set in
:mod:`pid.controller`, not in this file.

``kp``, ``ki``, and ``kd`` are search intervals ``[low, high]``. A single
number, such as ``ki: 3``, fixes that gain. The search never leaves these
limits.

``method`` is ``powell``, ``pygad``, or ``pso``. All three minimize the same
simulated error. ``powell`` starts from the low, middle, and high end of each
free gain, then keeps the visited point with the smallest error. ``pygad``
searches the same box genetically. ``pso`` is a global-best particle swarm:
each particle moves toward its own best position and the swarm's best
position. ``generations`` and ``population`` are the iteration count and
swarm size for ``pso``, and the generation count and population for
``pygad``. ``parents``, ``selection``, ``crossover``, ``mutation``, and
``elitism`` apply only to ``pygad``. ``selection`` is ``sss``, ``rws``,
``sus``, ``rank``, ``random``, or ``tournament``. ``crossover`` is
``single_point``, ``two_points``, ``uniform``, or ``scattered``.
``mutation`` is ``random``, ``swap``, ``inversion``, or ``scramble``. Omit
``parents`` to mate one third of the population. ``inertia``, ``cognitive``,
and ``social`` apply only to ``pso``. ``seed`` applies to ``pygad`` and
``pso``. ``maxfev``, ``xtol``, and ``ftol`` apply only to ``powell``:
``maxfev`` limits evaluations after the starting points, and ``xtol`` and
``ftol`` are the step and cost tolerances.

Identify
--------

.. code-block:: bash

   just identify
   uv run python app/identify.py --config config/config.yaml

lmfit minimizes the output error of each structure in ``identify.models``.
An interactive terminal shows a bar for each structure and its residual
evaluations.
The shipped file tries inertial orders 1–3, with and without Coulomb friction,
then integrators with 0, 1, and 2 lags and friction. A block can set
``integrators`` to a higher power of ``1/s`` and ``zeros`` to the number of
real derivative factors. The proportional gain is
``identify.kp``. It is not a fitted parameter.

The lowest RMSE is saved as a :class:`plantid.plant.Plant`: a transfer function
with a Coulomb dead-zone on the input. ``output/identify.png`` shows the measured
output, the fitted response, and the residual.

Tune
----

.. code-block:: bash

   just tune
   uv run python app/tune.py --config config/config.yaml

Run identification first, so ``tune.plant`` exists. An interactive terminal
shows Powell evaluations or PyGAD generations. The simulated controller
is :meth:`pid.controller.PID.step`. The shipped body matches ``PID.cpp``:

1. ``e = w - ym``, cleared when ``|e| < dead_zone``. ``w`` is the setpoint
   and ``ym`` is the sensor reading.
2. The accumulator is ``e + (1 - smoothing) * acc``, clamped to
   ``±anti_windup``.
3. ``p = Kp * e``, ``i = acc * dt * Ki``, and
   ``d = Kd * (e - last_error) / dt``. The first sample uses
   ``last_error = 0``.
4. The command ``u`` is lifted to ``±min_output`` and clipped to ``±max_output``.

To match a different PID, edit ``src/pid/controller.py``. Keep ``kp``, ``ki``,
and ``kd``. Dead zone and accumulator stay in that file.

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
