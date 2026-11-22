# Plant identification and PID tuning

Fit a plant from a closed-loop proportional log, save it as JSON, then tune the PID in `src/pid/controller.py` on that plant. Identification and tuning are separate: `identify.py` does not choose gains, and `tune.py` does not fit a plant. Edit `PID.step` in that module to match your controller. The shipped body matches `PID.cpp`.

## Setup

Python 3.14 is required.

```bash
uv sync
```

Logged CSV files belong in `data/`. That directory is gitignored.

## Configuration

Edit `config/config.yaml`. Paths there are relative to the repository root.

`identify.log` is the CSV. `identify.out` is the plant file, by default `model/plant.json`. `identify.kp` is the proportional gain of the loop that recorded the log. `identify.models` points at `config/models.yaml`, the list of structures to fit. Add a block to that file to try another one. The log needs `time_s`, `set_point`, and `measure_value`. Other columns are ignored.

`tune` sets the plant path, the sample time, the setpoint levels, how long each level is held, and the `Kp`, `Ki`, `Kd` limits. A single number fixes that gain. `we` weights the integral of squared setpoint error. Dead zone and accumulator are edited in `src/pid/controller.py`.

`method` is `powell` or `pygad`. Both minimize that same simulated error and stay inside the limits you set. `generations` and `population` apply to `pygad`.

## Identify

```bash
uv run app/identify.py --config config/config.yaml
```

`--config` is optional and defaults to `config/config.yaml`. A relative path starts at the repository root. `just identify` uses that default; `just identify path/to/other.yaml` selects another file.

The same steps are in `notebooks/identify.ipynb`.

lmfit minimizes the output error of each structure in `config/models.yaml`. The shipped list is inertial plants of order 1–3, with and without Coulomb friction, and integrators with 0–2 lags and friction. The proportional gain is `identify.kp`. It is not fitted. The lowest RMSE is written to `model/plant.json`: a `Plant`, a transfer function with a Coulomb dead-zone on the input.

## Tune

```bash
uv run app/tune.py --config config/config.yaml
```

The same `--config` default applies. `just tune path/to/other.yaml` selects another file.

The same search is in `notebooks/tune.ipynb`.

The tuner calls `pid.make` and `PID.step`. The shipped controller clears the error inside its dead zone, clamps the accumulator to `±anti_windup`, starts the derivative from a previous error of zero, and lifts the output to `±min_output` before clipping it to `±max_output`. Those limits are edited in `src/pid/controller.py`. Identification and tuning do not set them.

## Commands

[just](https://github.com/casey/just) runs the common tasks from the repository root.

- `just check` lints and runs the tests. Continuous integration runs this on every push and pull request.
- `just identify` fits the plant and writes `output/identify.png`.
- `just tune` tunes the PID and writes `output/tune.png`.

`output/` holds those figures. The image files are gitignored. Identify and tune stay off the CI workflow because the logs in `data/` are not in the repository.

## Documentation

```bash
just docs
```

That writes HTML to `docs/_build/html`. On every push to `main`, GitHub Actions builds the same site and publishes it to GitHub Pages:

https://mateuszzx.github.io/identification_and_control_tuning/

In the repository settings, set Pages → Build and deployment → Source to **GitHub Actions**.

## Tests

```bash
uv run pytest
```

## License

Released under the [MIT License](LICENSE).
