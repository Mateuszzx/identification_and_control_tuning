"""Tune the PID in :mod:`pid` for a plant saved by identify.py.

The config defaults to ``config/config.yaml`` and can be overridden with
``--config``. ``method`` selects Powell or PyGAD. The figure is written to
``output/tune.png``. This script does not identify the plant.
"""

import argparse
from pathlib import Path

import numpy as np

import pidtune
import plantid
from plantid.config import repository_root, resolve_config
from plantid.plots import save_tune


def main(argv: list[str] | None = None) -> None:
    """Search the PID gains and save the simulated figure.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments after the program name. ``None`` reads the process arguments.

    Raises
    ------
    SystemExit
        The config file or the plant JSON is missing.
    """
    parser = argparse.ArgumentParser(description="Tune a firmware PID from a saved plant.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/config.yaml",
        help="YAML config, relative to the repository root (default: config/config.yaml)",
    )
    args = parser.parse_args(argv)
    root = repository_root(Path(__file__))
    config_path = resolve_config(args.config, root)
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    cfg = plantid.load_config(config_path, root=root).tune
    if not cfg.plant.is_file():
        raise SystemExit(f"plant file not found: {cfg.plant}  (run identify.py first)")
    plant = pidtune.load_plant(cfg.plant)
    tuned = pidtune.tune(
        plant,
        levels=cfg.levels,
        hold=cfg.hold,
        dt=cfg.dt,
        kp=cfg.kp,
        ki=cfg.ki,
        kd=cfg.kd,
        we=cfg.we,
        method=cfg.method,
        generations=cfg.generations,
        population=cfg.population,
        parents=cfg.parents,
        selection=cfg.selection,
        crossover=cfg.crossover,
        mutation=cfg.mutation,
        mutation_by_replacement=cfg.mutation_by_replacement,
        elitism=cfg.elitism,
        seed=cfg.seed,
    )
    print(tuned)
    figure = root / "output" / "tune.png"
    show_t = np.arange(len(tuned.reference)) * cfg.dt
    save_tune(figure, show_t, tuned.reference, tuned.response, tuned.output)
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
