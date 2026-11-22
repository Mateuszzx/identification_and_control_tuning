"""Identify a plant from a closed-loop P log and save it as JSON.

The config defaults to ``config/config.yaml`` and can be overridden with
``--config``. The figure is written to ``output/identify.png``. This script
does not choose PID gains.
"""

import argparse
from pathlib import Path

import plantid
from plantid.config import repository_root, resolve_config
from plantid.plots import save_identify


def main(argv: list[str] | None = None) -> None:
    """Fit the catalog and save the best plant and its figure.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments after the program name. ``None`` reads the process arguments.

    Raises
    ------
    SystemExit
        The config file or the log is missing.
    """
    parser = argparse.ArgumentParser(description="Identify a plant and save it as JSON.")
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
    cfg = plantid.load_config(config_path, root=root)
    log = cfg.identify.log
    out = cfg.identify.out
    if not log.is_file():
        raise SystemExit(f"log not found: {log}")
    record = plantid.Log.read(log)
    found = record.identify()
    best = found.best
    best.save(out)
    print(best.model.name)
    print(f"P = {best.kp:g}")
    print(f"G(s) = {best.formula()}")
    print(f"Fc = {best.plant.fc:.4g}   RMSE = {best.rmse:.4g}")
    print(f"saved {out}")
    figure = root / "output" / "identify.png"
    save_identify(
        figure,
        record.time,
        record.setpoint,
        record.measurement,
        best.prediction,
        best.model.name,
    )
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
