Quick start
===========

Python 3.14 is required. From the repository root:

.. code-block:: bash

   uv sync
   just identify
   just tune

``just identify`` reads the log named in ``config/config.yaml``, fits the
catalog, and writes ``model/plant.json`` and ``output/identify.png``.
``just tune`` loads that plant, searches the PID gains, and writes
``output/tune.png``.

Put the CSV log in ``data/`` before identifying. That directory is gitignored.
The log needs these columns:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``time_s``
     - Sample time, in seconds.
   * - ``set_point``
     - Reference sent to the proportional loop.
   * - ``measure_value``
     - Measured output.
   * - ``error``
     - ``set_point - measure_value``.
   * - ``p_term``
     - Proportional term recorded by the firmware.

Edit ``config/config.yaml`` to point ``identify.log`` at that file and to set
the setpoint levels and the ``Kp``, ``Ki``, and ``Kd`` limits. Paths in the
file are relative to the repository root.

Another config file can be passed explicitly:

.. code-block:: bash

   just identify path/to/other.yaml
   just tune path/to/other.yaml

The same entry points without ``just`` are ``uv run python app/identify.py``
and ``uv run python app/tune.py``. Both accept ``--config`` and default to
``config/config.yaml``. The notebooks ``notebooks/identify.ipynb`` and
``notebooks/tune.ipynb`` follow the same config.
