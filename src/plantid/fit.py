"""Fit a catalog of models and rank them by output error."""

from dataclasses import dataclass

import lmfit
import numpy as np
import polars as pl

import control

from plantid.models import Model
from plantid.plant import io_system, proportional_response


@dataclass
class FitResult:
    """One fitted model."""

    model: Model
    plant: control.NonlinearIOSystem
    rmse: float
    at_bound: bool
    prediction: np.ndarray
    kp: float

    @property
    def gain(self) -> float:
        return float(self.plant.params["gain"])

    @property
    def taus(self) -> tuple[float, ...]:
        return tuple(self.plant.params["taus"])

    @property
    def fc(self) -> float | None:
        if not self.model.friction:
            return None
        return float(self.plant.params["fc"])

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
    """Output-error fit of a single model in a P loop with gain ``kp``.

    lmfit minimizes the squared error between the measured output and the
    simulated P loop. Gain, time constants and friction are fitted in log space.
    """
    time = np.asarray(time, dtype=float)
    setpoint = np.asarray(setpoint, dtype=float)
    measurement = np.asarray(measurement, dtype=float)
    y0 = float(measurement[0])
    params = _parameters(model)

    def residual(pars: lmfit.Parameters) -> np.ndarray:
        plant = _plant_from_log(model, _vector(pars, model))
        return proportional_response(plant, time, setpoint, kp, y0) - measurement

    result = lmfit.minimize(
        residual,
        params,
        method="least_squares",
        max_nfev=40,
        xtol=1e-10,
        ftol=1e-8,
        gtol=1e-8,
    )
    plant = _plant_from_log(model, _vector(result.params, model))
    prediction = proportional_response(plant, time, setpoint, kp, y0)
    rmse = float(np.sqrt(np.mean(np.square(prediction - measurement))))
    return FitResult(model, plant, rmse, _at_bound(result.params), prediction, float(kp))


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


def _parameters(model: Model) -> lmfit.Parameters:
    """Log-space parameters. Exponentiating them keeps gain, τ and Fc positive."""
    n = model.order
    if model.kind == "inertial":
        gain0 = np.log(20.0 if model.friction else 8.0)
        tau0 = np.log(np.geomspace(30.0 if model.friction else 8.0, 0.1, n))
    else:
        gain0 = np.log(0.8)
        tau0 = np.log(np.geomspace(0.2, 0.05, n)) if n else np.array([])
    params = lmfit.Parameters()
    params.add("log_gain", value=float(gain0), min=-2.0, max=8.0)
    for index, value in enumerate(tau0):
        params.add(f"log_tau_{index}", value=float(value), min=-7.0, max=8.0)
    if model.friction:
        params.add("log_fc", value=float(np.log(10.0)), min=0.0, max=3.5)
    return params


def _vector(params: lmfit.Parameters, model: Model) -> np.ndarray:
    values = [params["log_gain"].value]
    values.extend(params[f"log_tau_{index}"].value for index in range(model.order))
    if model.friction:
        values.append(params["log_fc"].value)
    return np.asarray(values, dtype=float)


def _at_bound(params: lmfit.Parameters) -> bool:
    for par in params.values():
        if abs(par.value - par.min) < 1e-3 or abs(par.value - par.max) < 1e-3:
            return True
    return False


def _plant_from_log(model: Model, p: np.ndarray) -> control.NonlinearIOSystem:
    gain = float(np.exp(p[0]))
    taus = tuple(sorted((float(np.exp(v)) for v in p[1 : 1 + model.order]), reverse=True))
    fc = float(np.exp(p[-1])) if model.friction else 0.0
    return io_system(model.kind, gain, taus, fc)
