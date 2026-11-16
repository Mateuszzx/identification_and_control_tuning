"""Tune a firmware PID for a plant saved by identify.py.

The waveform and the gain limits come from ``config/config.yaml``. Run
``python tune.py``. ``method`` in the config selects Powell or PyGAD. This
script does not identify the plant.
"""

from pathlib import Path

import pidtune
import plantid

ROOT = Path(__file__).resolve().parent
CFG = plantid.load_config(ROOT / "config" / "config.yaml", root=ROOT).tune


def main() -> None:
    if not CFG.plant.is_file():
        raise SystemExit(f"plant file not found: {CFG.plant}  (run identify.py first)")
    plant = pidtune.load_plant(CFG.plant)
    tuned = pidtune.tune(
        plant,
        levels=CFG.levels,
        hold=CFG.hold,
        dt=CFG.dt,
        dead_zone=CFG.dead_zone,
        kp=CFG.kp,
        ki=CFG.ki,
        kd=CFG.kd,
        we=CFG.we,
        allow_steady_state_error=CFG.allow_steady_state_error,
        method=CFG.method,
        generations=CFG.generations,
        population=CFG.population,
    )
    print(tuned)


if __name__ == "__main__":
    main()
