"""Ranked result of fitting a catalog of models."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from plantid.models import Model
from plantid.plant import Plant


@dataclass
class FitResult:
    """One fitted model.

    Attributes
    ----------
    model : Model
        Structure that was fitted.
    plant : Plant
        Fitted plant, including the Coulomb dead-zone when the model has friction.
    rmse : float
        Root-mean-square output error.
    at_bound : bool
        True when a fitted parameter sits on its bound.
    prediction : numpy.ndarray
        Simulated output on the logged time base.
    kp : float
        Proportional gain used in the simulated loop.
    """

    model: Model
    plant: Plant
    rmse: float
    at_bound: bool
    prediction: np.ndarray
    kp: float

    @property
    def gain(self) -> float:
        """Fitted linear gain.

        Returns
        -------
        float
            ``K`` for an inertial plant, ``Kv`` for an integrator.
        """
        return self.plant.gain

    @property
    def taus(self) -> tuple[float, ...]:
        """Fitted time constants.

        Returns
        -------
        tuple of float
            Real poles stored on the plant.
        """
        return self.plant.taus

    @property
    def fc(self) -> float | None:
        """Fitted Coulomb threshold.

        Returns
        -------
        float or None
            Threshold when the model includes friction, otherwise ``None``.
        """
        if not self.model.friction:
            return None
        return self.plant.fc

    def formula(self) -> str:
        """Text form of the linear transfer function.

        Returns
        -------
        str
            ``G(s)`` with time constants sorted from slowest to fastest.
        """
        num = f"{self.gain:.4g}" + "".join(f"({zero:.4g} s + 1)" for zero in self.plant.zeros)
        lags = "".join(f"({tau:.4g} s + 1)" for tau in sorted(self.taus, reverse=True))
        power = self.plant.integrators
        if power == 0:
            return f"{num} / {lags}"
        pole = "s" if power == 1 else f"s^{power}"
        if lags:
            return f"{num} / ({pole} {lags})"
        return f"{num} / {pole}"

    def save(self, path) -> None:
        """Write this plant as JSON, including the fit metadata.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination. Parent directories are created.
        """
        self.plant.save(
            path,
            model=self.model.name,
            kp=self.kp,
            rmse=self.rmse,
            formula=self.formula(),
        )


@dataclass
class Fit:
    """Ranked search result. ``best`` is the lowest RMSE.

    Attributes
    ----------
    results : list of FitResult
        Fits sorted by RMSE, lowest first.
    kp : float
        Proportional gain shared by every fit.
    """

    results: list[FitResult]
    kp: float

    @classmethod
    def search(
        cls,
        time: np.ndarray,
        setpoint: np.ndarray,
        measurement: np.ndarray,
        models: list[Model],
        *,
        kp: float,
    ) -> "Fit":
        """Fit every model and sort by RMSE, lowest first.

        Parameters
        ----------
        time : numpy.ndarray
            Sample times from the log.
        setpoint : numpy.ndarray
            Logged setpoint :math:`w(t)`.
        measurement : numpy.ndarray
            Logged sensor reading :math:`y_m(t)`.
        models : list of Model
            Structures to fit. Must not be empty.
        kp : float
            Proportional gain of the loop that collected the log.

        Returns
        -------
        Fit
            All fits, sorted by increasing RMSE.

        Raises
        ------
        ValueError
            ``models`` is empty.
        """
        if not models:
            raise ValueError("models is empty")
        results = [model.fit(time, setpoint, measurement, kp=kp) for model in models]
        results.sort(key=lambda item: item.rmse)
        return cls(results, kp)

    @property
    def best(self) -> FitResult:
        """Fit with the lowest RMSE.

        Returns
        -------
        FitResult
            ``results[0]``.
        """
        return self.results[0]

    @property
    def table(self) -> pl.DataFrame:
        """One row per fitted model.

        Returns
        -------
        polars.DataFrame
            Columns ``model``, ``P``, ``G(s)``, ``Fc``, ``rmse``, and ``at_bound``.
        """
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
