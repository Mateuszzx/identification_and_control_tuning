"""Tune a firmware PID for a plant saved by identify.py.

Edit the waveform and the gain limits below, then run ``python tune.py``.
The search uses PyGAD. It does not identify the plant.
"""

from pathlib import Path

import pidtune

# Setpoint waveform. Each level is held for HOLD_S seconds, in this order.
LEVELS = (
    0.0,
    -30.0,
    0.0,
    -60.0,
    0.0,
    -90.0,
    0.0,
    30.0,
    0.0,
    60.0,
    0.0,
    90.0,
    0.0,
)
HOLD_S = 6.0

PLANT = Path("plant.json")
DT = 15e-3
DEAD_ZONE = 0.1
KP = (1.0, 10.0)
KI = (0.0, 7.0)
KD = (0.0, 0.1)
WE = 1.0
ALLOW_STEADY_STATE_ERROR = False
GENERATIONS = 40
POPULATION = 24


def main() -> None:
    if not PLANT.is_file():
        raise SystemExit(f"plant file not found: {PLANT}  (run identify.py first)")
    plant = pidtune.load_plant(PLANT)
    tuned = pidtune.tune(
        plant,
        levels=LEVELS,
        hold=HOLD_S,
        dt=DT,
        dead_zone=DEAD_ZONE,
        kp=KP,
        ki=KI,
        kd=KD,
        we=WE,
        allow_steady_state_error=ALLOW_STEADY_STATE_ERROR,
        generations=GENERATIONS,
        population=POPULATION,
    )
    print(tuned)


if __name__ == "__main__":
    main()
