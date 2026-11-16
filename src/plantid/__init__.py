"""Plant identification from a closed-loop log, and the firmware PID simulation.

PID search lives in ``pidtune``. This package fits the plant and can simulate
the controller from ``PID.cpp``. It does not choose the gains.
"""

from plantid.config import Config, IdentifySetup, TuneSetup, load_config
from plantid.fit import Fit, FitResult, fit_model, loop_gain, search
from plantid.models import Model, default_models, inertial, integrator
from plantid.pid import DT, ENCODER_STEP, FirmwarePID, Gains, Limits, ki_that_holds, simulate, steps
from plantid.plant import io_system, load, save
from plantid.systems import closed_loop

__all__ = [
    "DT",
    "ENCODER_STEP",
    "Config",
    "Fit",
    "FitResult",
    "FirmwarePID",
    "Gains",
    "IdentifySetup",
    "Limits",
    "Model",
    "TuneSetup",
    "closed_loop",
    "default_models",
    "fit_model",
    "inertial",
    "integrator",
    "io_system",
    "load",
    "load_config",
    "save",
    "ki_that_holds",
    "loop_gain",
    "search",
    "simulate",
    "steps",
]
