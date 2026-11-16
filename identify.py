"""Identify a plant from a closed-loop P log and save it as JSON.

Edit the paths below, then run ``python identify.py``. The tuning script reads
the file this writes. This script does not choose PID gains.
"""

from pathlib import Path

import polars as pl

import plantid

# Log collected with a proportional controller. P is read from p_term / error.
LOG = Path("data/fr_turn_pid_ground.csv")
OUT = Path("plant.json")


def main() -> None:
    if not LOG.is_file():
        raise SystemExit(f"log not found: {LOG}")
    frame = pl.read_csv(LOG).sort("time_s")
    time = frame["time_s"].to_numpy()
    setpoint = frame["set_point"].to_numpy()
    measurement = frame["measure_value"].to_numpy()
    kp = plantid.loop_gain(frame["error"].to_numpy(), frame["p_term"].to_numpy())
    found = plantid.search(time, setpoint, measurement, plantid.default_models(), kp=kp)
    best = found.best
    plantid.save(
        best.plant,
        OUT,
        model=best.model.name,
        kp=best.kp,
        rmse=best.rmse,
        formula=best.formula(),
    )
    print(best.model.name)
    print(f"P = {best.kp:g}")
    print(f"G(s) = {best.formula()}")
    print(f"Fc = {best.plant.params['fc']:.4g}   RMSE = {best.rmse:.4g}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
