"""Fit a catalog of models and rank them by output error."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import least_squares

from plantid.models import Model
from plantid.plant import Plant


@dataclass
class FitResult:
    """One fitted model."""

    model: Model
    plant: Plant
    rmse: float
    at_bound: bool
    prediction: np.ndarray
    kp: float

    @property
    def system(self):
        """python-control transfer function of the linear part."""
        return self.plant.system

    @property
    def gain(self) -> float:
        return self.plant.gain

    @property
    def taus(self) -> tuple[float, ...]:
        return self.plant.taus

    @property
    def fc(self) -> float | None:
        return self.plant.fc if self.model.friction else None

    def formula(self) -> str:
        taus = tuple(sorted(self.taus, reverse=True))
        lags = "".join(f"({tau:.4g} s + 1)" for tau in taus)
        if self.model.kind == "integrator":
            if lags:
                return f"{self.gain:.4g} / (s {lags})"
            return f"{self.gain:.4g} / s"
        return f"{self.gain:.4g} / {lags}"


@dataclass
class Fit:
    """Ranked search result. ``best`` is the lowest RMSE."""

    results: list[FitResult]
    kp: float

    @property
    def best(self) -> FitResult:
        return self.results[0]

    @property
    def table(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "model": [item.model.name for item in self.results],
                "P": [item.kp for item in self.results],
                "G(s)": [item.formula() for item in self.results],
                "Fc": [item.fc for item in self.results],
                "rmse": [item.rmse for item in self.results],
                "at_bound": [item.at_bound for item in self.results],
            }
        )


def loop_gain(error: np.ndarray, p_term: np.ndarray) -> float:
    """Proportional gain recorded in the log, ``median(p_term / error)``.

    ``p_term`` is ``Kp * error`` before the output clip, so the ratio stays
    the collection gain on saturated samples too.
    """
    error = np.asarray(error, dtype=float)
    p_term = np.asarray(p_term, dtype=float)
    usable = np.abs(error) > 1e-9
    if not np.any(usable):
        raise ValueError("the log has no nonzero error, so P cannot be read")
    return float(np.median(p_term[usable] / error[usable]))


def fit_model(
    time: np.ndarray,
    setpoint: np.ndarray,
    measurement: np.ndarray,
    model: Model,
    *,
    kp: float,
) -> FitResult:
    """Output-error fit of a single model in a P loop with gain ``kp``."""
    time = np.asarray(time, dtype=float)
    setpoint = np.asarray(setpoint, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    y0 = float(measurement[0])
    lower, upper, x0 = _bounds(model)

    def residual(p: np.ndarray) -> np.ndarray:
        plant = _plant_from_log(model, p)
        return plant.simulate(time, setpoint, kp, y0) - measurement

    result = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=40,
    )
    plant = _plant_from_log(model, result.x)
    prediction = plant.simulate(time, setpoint, kp, y0)
    rmse = float(np.sqrt(np.mean(np.square(prediction - measurement))))
    at_bound = bool(np.any(np.abs(result.x - lower) < 1e-3) or np.any(np.abs(result.x - upper) < 1e-3))
    return FitResult(model, plant, rmse, at_bound, prediction, float(kp))


def search(
    time: np.ndarray,
    setpoint: np.ndarray,
    measurement: np.ndarray,
    models: list[Model],
    *,
    kp: float,
) -> Fit:
    """Fit every model and sort by RMSE, lowest first."""
    if not models:
        raise ValueError("models is empty")
    results = [fit_model(time, setpoint, measurement, model, kp=kp) for model in models]
    results.sort(key=lambda item: item.rmse)
    return Fit(results, kp)


def _bounds(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = model.order
    if model.kind == "inertial":
        gain0 = np.log(20.0 if model.friction else 8.0)
        tau0 = np.log(np.geomspace(30.0 if model.friction else 8.0, 0.1, n))
    else:
        gain0 = np.log(0.8)
        tau0 = np.log(np.geomspace(0.2, 0.05, n)) if n else np.array([])
    x0 = [gain0, *tau0]
    lower = [-2.0, *([-7.0] * n)]
    upper = [8.0, *([8.0] * n)]
    if model.friction:
        x0.append(np.log(10.0))
        lower.append(0.0)
        upper.append(3.5)
    return np.asarray(lower), np.asarray(upper), np.asarray(x0, dtype=float)


def _plant_from_log(model: Model, p: np.ndarray) -> Plant:
    gain = float(np.exp(p[0]))
    taus = tuple(sorted((float(np.exp(v)) for v in p[1 : 1 + model.order]), reverse=True))
    fc = float(np.exp(p[-1])) if model.friction else 0.0
    return Plant(model.kind, gain, taus, fc)
