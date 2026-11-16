# Plant identification and firmware PID tuning

Fit a plant from a closed-loop proportional log, save it as JSON, then tune the firmware PID on that plant. Identification and tuning are separate: `identify.py` does not choose gains, and `tune.py` does not fit a plant.

## Setup

Python 3.14 is required.

```bash
uv sync
```

Logged CSV files belong in `data/`. That directory is gitignored.

## Configuration

Edit `config/config.yaml`. Paths there are relative to the repository root.

`identify.log` is the CSV. `identify.out` is the plant file, by default `model/plant.json`. The log needs `time_s`, `set_point`, `measure_value`, `error`, and `p_term`.

`tune` sets the plant path, the sample time, the firmware dead zone, the setpoint levels, how long each level is held, and the `Kp`, `Ki`, `Kd` limits. A single number fixes that gain. `we` weights the integral of squared setpoint error.

`method` is `powell` or `pygad`. Both minimize that same simulated error and stay inside the limits you set. `generations` and `population` apply to `pygad`.

`allow_steady_state_error: false` raises the `Ki` floor, inside your limits, so the integrator can hold every level. The search never goes above the upper limit you set. `true` searches the whole `Ki` range and may leave a shelf.

## Identify

```bash
uv run python identify.py
```

The same steps are in `notebooks/identify.ipynb`.

lmfit minimizes the output error of each candidate. The catalog is inertial plants of order 1–3, with and without Coulomb friction, and integrators with 0–2 lags and friction. The proportional gain is read from the log as `median(p_term / error)`. It is not fitted. The lowest RMSE is written to `model/plant.json`: a python-control transfer function with a Coulomb dead-zone on the input.

## Tune

```bash
uv run python tune.py
```

The same search is in `notebooks/tune.ipynb`.

The simulated controller follows the firmware law. The error is cleared inside the dead zone, the accumulator is clamped to `±anti_windup`, the derivative on the first sample starts from a previous error of zero, and the output is lifted to `±min_output` and clipped to `±max_output`.

## Commands

[just](https://github.com/casey/just) runs the common tasks from the repository root.

- `just check` lints and runs the tests. Continuous integration runs this on every push and pull request.
- `just identify` fits the plant and writes `output/identify.png`.
- `just tune` tunes the PID and writes `output/tune.png`.

`output/` holds those figures. The image files are gitignored. Identify and tune stay off the CI workflow because the logs in `data/` are not in the repository.

## Tests

```bash
uv run pytest
```
