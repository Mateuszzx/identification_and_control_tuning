"""Identify a plant from a closed-loop P log and save it as JSON.

Paths come from ``config/config.yaml``. Run ``python identify.py``. The figure
is written to ``output/identify.png``. The tuning script reads the plant this
writes. This script does not choose PID gains.
"""

from pathlib import Path

import polars as pl

import plantid
from plantid.plots import save_identify

ROOT = Path(__file__).resolve().parent
CFG = plantid.load_config(ROOT / "config" / "config.yaml", root=ROOT)


def main() -> None:
    log = CFG.identify.log
    out = CFG.identify.out
    if not log.is_file():
        raise SystemExit(f"log not found: {log}")
    frame = pl.read_csv(log).sort("time_s")
    time = frame["time_s"].to_numpy()
    setpoint = frame["set_point"].to_numpy()
    measurement = frame["measure_value"].to_numpy()
    kp = plantid.loop_gain(frame["error"].to_numpy(), frame["p_term"].to_numpy())
    found = plantid.search(time, setpoint, measurement, plantid.default_models(), kp=kp)
    best = found.best
    plantid.save(
        best.plant,
        out,
        model=best.model.name,
        kp=best.kp,
        rmse=best.rmse,
        formula=best.formula(),
    )
    print(best.model.name)
    print(f"P = {best.kp:g}")
    print(f"G(s) = {best.formula()}")
    print(f"Fc = {best.plant.params['fc']:.4g}   RMSE = {best.rmse:.4g}")
    print(f"saved {out}")
    figure = ROOT / "output" / "identify.png"
    save_identify(figure, time, setpoint, measurement, best.prediction, best.model.name)
    print(f"saved {figure}")


if __name__ == "__main__":
    main()
